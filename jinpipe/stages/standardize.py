"""Standardize stage: convert raw downloaded audio to mono 16kHz WAV via ffmpeg.

Each call is a single blocking ffmpeg invocation - stateless and cheap enough
that the orchestrator dispatches it through a plain ProcessPoolExecutor rather
than a persistent worker.
"""

from __future__ import annotations

import subprocess
from collections.abc import Callable
from pathlib import Path

RunnerT = Callable[[list[str]], subprocess.CompletedProcess]


def _default_runner(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(args, capture_output=True, stdin=subprocess.DEVNULL)


def standardize_audio(
    raw_path: Path,
    out_path: Path,
    sample_rate: int = 16000,
    *,
    runner: RunnerT = _default_runner,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    args = [
        "ffmpeg",
        "-y",
        "-i",
        str(raw_path),
        "-ac",
        "1",
        "-ar",
        str(sample_rate),
        "-c:a",
        "pcm_s16le",
        str(out_path),
    ]
    result = runner(args)
    if result.returncode != 0:
        stderr = result.stderr.decode(errors="replace") if isinstance(result.stderr, bytes) else str(result.stderr)
        raise RuntimeError(
            f"ffmpeg standardize failed for {raw_path} (exit {result.returncode}): {stderr.strip()}"
        )
    return out_path


def probe_duration(path: Path, *, runner: RunnerT = _default_runner) -> float | None:
    """ffprobe the STANDARDIZED wav's duration, not the raw source container's.

    A pcm_s16le wav's duration is sample-exact from its own header; a
    compressed container's (m4a/opus/webm) format-level duration metadata can
    undercount the actual decoded length, which would otherwise make VAD's
    overlap-padding clamp real trailing speech away as if it were past the
    audio's end.
    """
    args = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    result = runner(args)
    if result.returncode != 0:
        return None
    try:
        return float(result.stdout.decode().strip())
    except (ValueError, AttributeError):
        return None
