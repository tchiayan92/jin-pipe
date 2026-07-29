"""Package stage: write final segment audio + JSON metadata, idempotently.

Writes go through a temp-file-then-rename so a crash between quality
filtering and packaging can't leave a corrupted or half-written file behind
on resume - the rename is atomic on the same filesystem. The manifest is not
incrementally appended (which risks duplicate lines if a run crashes between
appending and marking the segment DONE in the job store); instead it's always
rebuilt from the per-segment JSON files already on disk, which are
themselves idempotent by construction. Rebuilding is cheap since it's just a
directory scan + JSON parse, not re-processing audio.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from jinpipe.config import PackageConfig
from jinpipe.stages.rechunk import Segment


def segment_id_for(video_id: str, idx: int) -> str:
    return f"{video_id}_{idx:05d}"


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def _default_runner(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True)


def slice_segment_audio(
    source_wav: Path,
    out_path: Path,
    start_s: float,
    end_s: float,
    audio_format: str,
    *,
    sample_rate: int | None = None,
    channels: int | None = None,
    runner=_default_runner,
) -> Path:
    """Slice [start_s, end_s) out of source_wav via ffmpeg seek.

    Never holds the whole video's audio in memory - ffmpeg streams the slice
    directly from disk. sample_rate/channels are left unset (native passthrough)
    by default, so this preserves whatever quality the source already has -
    pass them explicitly only to force a resample/downmix.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(out_path.suffix + ".tmp")
    args = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_s:.3f}",
        "-to",
        f"{end_s:.3f}",
        "-i",
        str(source_wav),
    ]
    if sample_rate is not None:
        args += ["-ar", str(sample_rate)]
    if channels is not None:
        args += ["-ac", str(channels)]
    args += ["-f", audio_format, str(tmp_path)]
    result = runner(args)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace") if isinstance(result.stderr, bytes) else str(result.stderr)
        raise RuntimeError(f"ffmpeg slice failed for {out_path} (exit {result.returncode}): {stderr.strip()}")
    tmp_path.replace(out_path)
    return out_path


@dataclass(frozen=True)
class PackagedSegment:
    segment_id: str
    audio_path: Path
    json_path: Path


def package_segment(
    video_id: str,
    source_url: str,
    segment: Segment,
    source_wav: Path,
    output_dir: Path,
    cfg: PackageConfig,
    *,
    dnsmos_ovr: float | None = None,
    language: str | None = None,
    slice_fn=slice_segment_audio,
) -> PackagedSegment:
    seg_id = segment_id_for(video_id, segment.idx)
    audio_path = output_dir / f"{seg_id}.{cfg.audio_format}"
    json_path = output_dir / f"{seg_id}.json"

    if not audio_path.exists():
        slice_fn(
            source_wav,
            audio_path,
            segment.start,
            segment.end,
            cfg.audio_format,
            sample_rate=cfg.sample_rate,
            channels=cfg.channels,
        )

    metadata = {
        "segment_id": seg_id,
        "video_id": video_id,
        "source_url": source_url,
        "start": segment.start,
        "end": segment.end,
        "duration": segment.end - segment.start,
        "text": segment.text,
        "words": [asdict(w) for w in segment.words],
        "speaker": segment.speaker,
        "language": language,
        "dnsmos_ovr": dnsmos_ovr,
        "exceeds_max_duration": segment.exceeds_max_duration,
        "has_overlap": segment.has_overlap,
    }
    _atomic_write_text(json_path, json.dumps(metadata, ensure_ascii=False, indent=2))
    return PackagedSegment(segment_id=seg_id, audio_path=audio_path, json_path=json_path)


def build_manifest(output_dir: Path) -> list[dict]:
    entries = []
    for json_path in sorted(output_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            entries.append(json.load(f))
    return entries


def write_manifest(output_dir: Path, manifest_path: Path) -> int:
    """Rebuild the manifest fresh from ground truth (the per-segment JSON files)."""
    entries = build_manifest(output_dir)
    lines = [json.dumps(e, ensure_ascii=False) for e in entries]
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(manifest_path, "\n".join(lines) + ("\n" if lines else ""))
    return len(entries)


UNKNOWN_SPEAKER = "UNKNOWN"


def dataset_stats(output_dir: Path) -> dict:
    """Aggregate speaker/duration stats from the per-segment JSON files on disk."""
    entries = build_manifest(output_dir)
    videos: set[str] = set()
    per_speaker: dict[str, dict[str, float]] = {}
    total_duration_s = 0.0
    for entry in entries:
        duration = entry.get("duration") or 0.0
        total_duration_s += duration
        videos.add(entry.get("video_id"))
        speaker = entry.get("speaker") or UNKNOWN_SPEAKER
        stat = per_speaker.setdefault(speaker, {"segments": 0, "duration_s": 0.0})
        stat["segments"] += 1
        stat["duration_s"] += duration
    return {
        "num_segments": len(entries),
        "num_videos": len(videos),
        "num_speakers": len(per_speaker),
        "total_duration_s": total_duration_s,
        "per_speaker": per_speaker,
    }


def print_dataset_stats(console, stats: dict) -> None:
    from rich.table import Table

    console.print(
        f"{stats['num_segments']} segment(s) across {stats['num_videos']} video(s), "
        f"{stats['num_speakers']} speaker(s), {stats['total_duration_s'] / 3600:.2f} total speech hour(s)"
    )
    table = Table(title="Speaker breakdown")
    table.add_column("speaker")
    table.add_column("segments", justify="right")
    table.add_column("hours", justify="right")
    for speaker, stat in sorted(stats["per_speaker"].items(), key=lambda kv: kv[1]["duration_s"], reverse=True):
        table.add_row(speaker, str(stat["segments"]), f"{stat['duration_s'] / 3600:.2f}")
    console.print(table)


def filter_speakers(output_dir: Path, keep: set[str], *, dry_run: bool = False) -> dict[str, int]:
    """Delete packaged segments (audio + JSON) whose speaker is not in `keep`.

    Ground truth for a dataset is the per-segment JSON/audio pairs on disk, not
    manifest.jsonl, so this only touches those files - callers must rebuild the
    manifest afterwards (e.g. via write_manifest) to reflect the new state.
    """
    kept = 0
    removed = 0
    for json_path in sorted(output_dir.glob("*.json")):
        with json_path.open("r", encoding="utf-8") as f:
            entry = json.load(f)
        speaker = entry.get("speaker") or UNKNOWN_SPEAKER
        if speaker in keep:
            kept += 1
            continue
        removed += 1
        if not dry_run:
            seg_id = entry["segment_id"]
            for path in output_dir.glob(f"{seg_id}.*"):
                path.unlink()
    return {"kept": kept, "removed": removed}
