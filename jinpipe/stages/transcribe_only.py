"""Straight-through transcription for a folder of already-chunked audio files.

Unlike the full pipeline (standardize -> VAD -> ASR -> rechunk -> filter ->
package), this assumes each file directly inside the input folder is already
a single utterance/chunk someone wants transcribed as-is - so it skips VAD,
rechunking, diarization, quality filtering, and audio slicing entirely, and
just writes each file's transcription text straight to a CSV. Used via
`jinpipe transcribe`, never the full `jinpipe run` pipeline.

whisperx.load_audio() decodes+resamples via ffmpeg regardless of the input
file's original sample rate/channel count, so files can be handed to it
directly without a separate standardize pass.
"""

from __future__ import annotations

import csv
from pathlib import Path

from jinpipe.config import AsrConfig
from jinpipe.stages.local import AUDIO_EXTENSIONS, scan_local_audio

CSV_FIELDNAMES = ("no", "filename", "text")


def load_model(cfg: AsrConfig, device: str):
    import whisperx

    from jinpipe.workers.asr_worker import resolve_compute_type

    compute_type = resolve_compute_type(cfg, device)
    return whisperx.load_model(cfg.model_size, device, compute_type=compute_type, language=cfg.language)


def transcribe_file(model, audio_path: Path, cfg: AsrConfig) -> str:
    import whisperx

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, language=cfg.language)
    return " ".join(seg["text"].strip() for seg in result["segments"]).strip()


def transcribe_folder(
    input_dir: Path,
    cfg: AsrConfig,
    *,
    device: str = "cpu",
    model=None,
    transcribe_fn=transcribe_file,
) -> list[dict]:
    """Transcribe every audio file directly inside input_dir. Returns rows: {no, filename, text}."""
    files = scan_local_audio(input_dir, AUDIO_EXTENSIONS)
    if model is None:
        model = load_model(cfg, device)
    rows = []
    for i, path in enumerate(files, start=1):
        text = transcribe_fn(model, path, cfg)
        rows.append({"no": i, "filename": path.name, "text": text})
    return rows


def write_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
