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
    runner=_default_runner,
) -> Path:
    """Slice [start_s, end_s) out of the standardized WAV via ffmpeg seek.

    Never holds the whole video's audio in memory - ffmpeg streams the slice
    directly from disk.
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
        "-f",
        audio_format,
        str(tmp_path),
    ]
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
        slice_fn(source_wav, audio_path, segment.start, segment.end, cfg.audio_format)

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
