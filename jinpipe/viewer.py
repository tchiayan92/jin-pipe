"""Gradio viewer: browse packaged chunk audio with transcript/speaker/overlap annotations.

Pure data-prep helpers (below) have no Gradio dependency and are safe to
import/test without the `viewer` extra installed. `build_app`/`launch_viewer`
import `gradio` and `gradio_annotated_audio` lazily, since both are optional
extras (pip install "jinpipe[viewer]") - most pipeline runs never need them.
"""

from __future__ import annotations

from pathlib import Path

from jinpipe.stages.package import UNKNOWN_SPEAKER, build_manifest

ALL = "All"


def load_entries(output_dir: Path) -> dict[str, dict]:
    return {e["segment_id"]: e for e in build_manifest(output_dir)}


def audio_path_for(output_dir: Path, entry: dict) -> Path | None:
    seg_id = entry["segment_id"]
    matches = [p for p in output_dir.glob(f"{seg_id}.*") if p.suffix != ".json"]
    return matches[0] if matches else None


def overlap_ranges(words: list[dict]) -> list[dict]:
    """Merge consecutive overlap-flagged words into contiguous [start, end) ranges."""
    ranges: list[dict] = []
    cur: dict | None = None
    for w in words:
        if w.get("overlap"):
            if cur is None:
                cur = {"start": w["clip_start"], "end": w["clip_end"]}
            else:
                cur["end"] = w["clip_end"]
        elif cur is not None:
            ranges.append(cur)
            cur = None
    if cur is not None:
        ranges.append(cur)
    return ranges


def segment_label(entry: dict) -> str:
    dnsmos = f", dnsmos={entry['dnsmos_ovr']:.2f}" if entry.get("dnsmos_ovr") is not None else ""
    speaker = entry.get("speaker") or UNKNOWN_SPEAKER
    return f"{entry['segment_id']}  |  {speaker}  |  {entry['duration']:.1f}s{dnsmos}"


def video_and_speaker_choices(entries: dict[str, dict]) -> tuple[list[str], list[str]]:
    videos = [ALL] + sorted({e["video_id"] for e in entries.values()})
    speakers = [ALL] + sorted({e.get("speaker") or UNKNOWN_SPEAKER for e in entries.values()})
    return videos, speakers


def segment_choices(entries: dict[str, dict], video: str | None, speaker: str | None) -> list[tuple[str, str]]:
    rows = [
        e
        for e in entries.values()
        if (video in (None, ALL) or e["video_id"] == video)
        and (speaker in (None, ALL) or (e.get("speaker") or UNKNOWN_SPEAKER) == speaker)
    ]
    rows.sort(key=lambda e: e["segment_id"])
    return [(segment_label(e), e["segment_id"]) for e in rows]


def transcript_for(entry: dict) -> list[dict]:
    """Word-level transcript rows (clip-relative) so playback highlighting tracks each word."""
    return [
        {
            "start": w["clip_start"],
            "end": w["clip_end"],
            "text": w["text"],
            "speaker": w.get("speaker") or entry.get("speaker"),
        }
        for w in entry.get("words") or []
    ]


def segment_info_markdown(entry: dict) -> str:
    parts = [
        f"**{entry['segment_id']}**",
        f"video: `{entry['video_id']}`",
        f"speaker: `{entry.get('speaker') or UNKNOWN_SPEAKER}`",
        f"duration: {entry['duration']:.2f}s",
    ]
    if entry.get("dnsmos_ovr") is not None:
        parts.append(f"dnsmos_ovr: {entry['dnsmos_ovr']:.2f}")
    if entry.get("language"):
        parts.append(f"language: {entry['language']}")
    if entry.get("exceeds_max_duration"):
        parts.append("**exceeds_max_duration**")
    return "  \n".join(parts)


def build_app(output_dir: Path):
    """Construct (but don't launch) the Gradio Blocks app for browsing output_dir."""
    import gradio as gr
    from gradio_annotated_audio import AnnotatedAudio

    def on_filter_change(video: str, speaker: str):
        entries = load_entries(output_dir)
        choices = segment_choices(entries, video, speaker)
        value = choices[0][1] if choices else None
        return gr.Dropdown(choices=choices, value=value)

    def on_segment_change(segment_id: str | None):
        if not segment_id:
            return None, "No chunk selected."
        entries = load_entries(output_dir)
        entry = entries.get(segment_id)
        if entry is None:
            return None, f"`{segment_id}` not found - output_dir may have changed."
        audio_path = audio_path_for(output_dir, entry)
        if audio_path is None:
            return None, f"Audio file missing for `{segment_id}`."

        words = entry.get("words") or []
        ranges = overlap_ranges(words)
        categories = [{"key": "overlap", "label": "Overlapping speech", "color": "#ef4444"}] if ranges else None
        annotations = [{"category": "overlap", "kind": "range", **r} for r in ranges] if ranges else None

        player = AnnotatedAudio(
            value=str(audio_path),
            categories=categories,
            annotations=annotations,
            transcript=transcript_for(entry),
            label=entry["segment_id"],
            interactive=False,
        )
        return player, segment_info_markdown(entry)

    entries0 = load_entries(output_dir)
    videos0, speakers0 = video_and_speaker_choices(entries0)
    segments0 = segment_choices(entries0, ALL, ALL)

    with gr.Blocks(title="JinPipe chunk viewer") as demo:
        gr.Markdown(f"## JinPipe chunk viewer — `{output_dir}`")
        with gr.Row():
            video_dd = gr.Dropdown(choices=videos0, value=ALL, label="Video")
            speaker_dd = gr.Dropdown(choices=speakers0, value=ALL, label="Speaker")
        segment_dd = gr.Dropdown(
            choices=segments0, value=segments0[0][1] if segments0 else None, label="Chunk"
        )
        info_md = gr.Markdown()
        player = AnnotatedAudio(label="Chunk audio", interactive=False)

        video_dd.change(on_filter_change, [video_dd, speaker_dd], segment_dd).then(
            on_segment_change, segment_dd, [player, info_md]
        )
        speaker_dd.change(on_filter_change, [video_dd, speaker_dd], segment_dd).then(
            on_segment_change, segment_dd, [player, info_md]
        )
        segment_dd.change(on_segment_change, segment_dd, [player, info_md])
        demo.load(on_segment_change, segment_dd, [player, info_md])

    return demo


def launch_viewer(output_dir: Path, **launch_kwargs) -> None:
    demo = build_app(output_dir)
    demo.launch(**launch_kwargs)
