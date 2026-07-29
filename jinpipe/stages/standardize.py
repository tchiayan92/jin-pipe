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
