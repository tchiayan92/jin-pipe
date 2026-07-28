"""Coarse VAD segmentation stage: Silero-VAD merges speech regions into
super-chunks capped at a configurable duration, so no single downstream
ASR/diarization call ever has to process more than `max_superchunk_s` of
audio regardless of how long the source video is.

Adjacent super-chunks overlap by a configurable margin. This matters because
the merge step occasionally has to force a split inside a long, pause-free
speech run once it exceeds the cap - without overlap that forced cut could
land mid-sentence or mid-word at the audio level. The overlap is resolved
later by the rechunk stage, which merges word-timestamp streams across
super-chunk boundaries and de-duplicates the overlapping words.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from pathlib import Path

from jinpipe.config import VadConfig

SpeechTimestampsFnT = Callable[[Path, VadConfig], list[dict]]

_model_cache = None


def _get_cached_model():
    global _model_cache
    if _model_cache is None:
        from silero_vad import load_silero_vad

        _model_cache = load_silero_vad()
    return _model_cache


def _default_speech_timestamps(wav_path: Path, cfg: VadConfig) -> list[dict]:
    from silero_vad import get_speech_timestamps, read_audio

    model = _get_cached_model()
    wav = read_audio(str(wav_path))
    return get_speech_timestamps(
        wav,
        model,
        threshold=cfg.threshold,
        min_silence_duration_ms=cfg.min_silence_duration_ms,
        min_speech_duration_ms=cfg.min_speech_duration_ms,
        speech_pad_ms=cfg.speech_pad_ms,
        return_seconds=True,
    )


def merge_speech_regions(regions: list[dict], max_superchunk_s: float) -> list[tuple[float, float]]:
    """Greedily merge adjacent speech regions into chunks capped at max_superchunk_s.

    A single region longer than the cap (a long, pause-free speech run) is
    force-split into equal-length pieces rather than left oversized - this is
    the one case overlap padding exists to protect against.
    """
    if not regions:
        return []

    chunks: list[tuple[float, float]] = []
    cur_start = regions[0]["start"]
    cur_end = regions[0]["end"]
    for region in regions[1:]:
        candidate_end = region["end"]
        if candidate_end - cur_start <= max_superchunk_s:
            cur_end = candidate_end
        else:
            chunks.append((cur_start, cur_end))
            cur_start = region["start"]
            cur_end = region["end"]
    chunks.append((cur_start, cur_end))

    final: list[tuple[float, float]] = []
    for start, end in chunks:
        span = end - start
        if span <= max_superchunk_s or max_superchunk_s <= 0:
            final.append((start, end))
            continue
        n_pieces = math.ceil(span / max_superchunk_s)
        piece_len = span / n_pieces
        for i in range(n_pieces):
            piece_start = start + i * piece_len
            piece_end = end if i == n_pieces - 1 else start + (i + 1) * piece_len
            final.append((piece_start, piece_end))
    return final


def apply_overlap(
    cut_points: list[tuple[float, float]], overlap_s: float, duration_s: float
) -> list[tuple[float, float]]:
    """Pad each internal boundary by overlap_s on both sides, clamped to [0, duration_s]."""
    n = len(cut_points)
    padded = []
    for i, (start, end) in enumerate(cut_points):
        padded_start = max(0.0, start - overlap_s) if i > 0 else start
        padded_end = min(duration_s, end + overlap_s) if i < n - 1 else min(end, duration_s)
        padded.append((padded_start, padded_end))
    return padded


def coarse_segment(
    wav_path: Path,
    cfg: VadConfig,
    duration_s: float,
    *,
    speech_timestamps_fn: SpeechTimestampsFnT = _default_speech_timestamps,
) -> list[tuple[int, float, float]]:
    """Return [(idx, start_s, end_s), ...] super-chunks covering all detected speech."""
    regions = speech_timestamps_fn(wav_path, cfg)
    cut_points = merge_speech_regions(regions, cfg.max_superchunk_s)
    padded = apply_overlap(cut_points, cfg.overlap_s, duration_s)
    return [(i, start, end) for i, (start, end) in enumerate(padded)]
