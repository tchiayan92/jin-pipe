"""End-to-end orchestrator wiring test.

Uses a real JobStore, ResourceGate, and ThreadPoolExecutor (so monkeypatches
apply directly - no multiprocessing pickling involved here, unlike
test_asr_worker.py which specifically needs real process isolation). Only
the heavy/model-backed stage functions (standardize, VAD, filtering) and the
ASR pool are faked; rechunk, packaging, and the job-store barrier logic all
run for real, which is the part most likely to have wiring bugs.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from jinpipe.config import JinPipeConfig, PathsConfig, RechunkConfig, SourcesConfig
from jinpipe.db import JobStore
from jinpipe.orchestrator import _process_video_to_superchunks
from jinpipe.resources import ResourceGate
from jinpipe.stages import filter as filter_stage
from jinpipe.stages import package as package_stage
from jinpipe.stages import standardize as standardize_stage
from jinpipe.stages import vad as vad_stage
from jinpipe.workers.asr_worker import AsrResult


class FakeAsrPool:
    def __init__(self, words_by_idx):
        self.words_by_idx = words_by_idx

    async def submit(self, task):
        return AsrResult(task.task_id, task.video_id, task.superchunk_idx, self.words_by_idx[task.superchunk_idx], None)


def _fake_standardize_audio(raw_path, out_path, sample_rate):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"fake-wav")
    return out_path


def _fake_coarse_segment(std_path, cfg, duration_s):
    return [(0, 0.0, 5.0), (1, 5.0, 10.0)]


def _fake_slice_segment_audio(source, out_path, start, end, audio_format, sample_rate=None, channels=None, runner=None):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(b"fake-audio")
    return out_path


def _fake_filter_segment(audio_path, duration_s, cfg, language=None, has_overlap=False, scorer=None):
    return filter_stage.FilterResult(True, 3.5, 3.5, 4.0, None)


async def test_process_video_to_superchunks_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr(standardize_stage, "standardize_audio", _fake_standardize_audio)
    monkeypatch.setattr(vad_stage, "coarse_segment", _fake_coarse_segment)
    monkeypatch.setattr(package_stage, "slice_segment_audio", _fake_slice_segment_audio)
    monkeypatch.setattr(filter_stage, "filter_segment", _fake_filter_segment)

    cfg = JinPipeConfig(
        sources=SourcesConfig(),
        paths=PathsConfig(
            work_dir=tmp_path / "work", output_dir=tmp_path / "output", db_path=tmp_path / "work" / "jobs.sqlite3"
        ),
        rechunk=RechunkConfig(min_segment_s=0.1, max_segment_s=2.0, silence_fallback_ms=250),
    )
    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    std_dir = cfg.paths.work_dir / "standardized"

    store = JobStore(cfg.paths.db_path)
    store.add_video("vid1", "https://youtu.be/vid1", channel="chan")
    # Mirrors what run_pipeline_async's outer download loop persists before
    # spawning _process_video_to_superchunks - _finalize_segment reads this
    # back to slice the final packaged audio from the original source.
    store.update_video("vid1", raw_path=str(tmp_path / "raw.m4a"))

    gate = ResourceGate(cfg.resources, ram_available_fn=lambda: 10**12)

    # Superchunk boundaries touch exactly (no overlap), so merge_superchunk_words
    # just concatenates - words are already in global time as the real
    # asr_worker would produce them after its local->global shift.
    words_by_idx = {
        0: [
            {"word": "Hello", "start": 0.0, "end": 0.5, "speaker": None},
            {"word": "world.", "start": 0.5, "end": 1.0, "speaker": None},
        ],
        1: [
            {"word": "This", "start": 5.5, "end": 6.0, "speaker": None},
            {"word": "is", "start": 6.0, "end": 6.2, "speaker": None},
            {"word": "Jin.", "start": 6.2, "end": 6.8, "speaker": None},
        ],
    }
    asr_pool = FakeAsrPool(words_by_idx)

    executor = ThreadPoolExecutor(max_workers=2)
    loop = asyncio.get_running_loop()
    try:
        await _process_video_to_superchunks(
            "vid1", tmp_path / "raw.m4a", 10.0, cfg, store, executor, loop, std_dir, gate, asr_pool
        )
    finally:
        executor.shutdown(wait=True)

    video = store.get_video("vid1")
    assert video["status"] == "DONE"

    superchunks = store.get_superchunks("vid1")
    assert [s["status"] for s in superchunks] == ["DONE", "DONE"]

    segments = store.get_segments("vid1")
    assert len(segments) == 2
    assert [s["status"] for s in segments] == ["DONE", "DONE"]
    assert segments[0]["text"] == "Hello world."
    assert segments[1]["text"] == "This is Jin."

    for seg in segments:
        assert Path(seg["output_audio_path"]).exists()
        assert Path(seg["output_json_path"]).exists()

    manifest_path = cfg.paths.output_dir / "manifest.jsonl"
    count = package_stage.write_manifest(cfg.paths.output_dir, manifest_path)
    assert count == 2

    store.close()
