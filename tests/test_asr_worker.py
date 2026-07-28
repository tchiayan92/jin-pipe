"""These tests exercise the real multiprocessing/queue/supervisor plumbing in
AsrWorkerPool - only the model loading and transcription are faked out, so
they run without torch/whisperx installed while still proving the process
lifecycle, crash-detection, and stall-detection logic actually works.

Fakes are module-level functions (not closures) because multiprocessing's
"spawn" context pickles callables by reference and re-imports them in the
child process.
"""

from __future__ import annotations

import asyncio
import os
import time
from pathlib import Path

import pytest

from jinpipe.config import AsrConfig
from jinpipe.workers.asr_worker import AsrWorkerPool, make_task


class _FakeModels:
    pass


def fake_loader(cfg, device, gpu_id):
    return _FakeModels()


def fake_transcribe_simple(models, audio_path, cfg):
    words = []
    for part in audio_path.split("|"):
        text, start, end = part.split(":")
        words.append({"word": text, "start": float(start), "end": float(end), "speaker": None})
    return words


def fake_transcribe_with_error_marker(models, audio_path, cfg):
    if audio_path == "POISON":
        raise RuntimeError("simulated transcribe failure")
    return fake_transcribe_simple(models, audio_path, cfg)


def fake_transcribe_crash_once(models, audio_path, cfg):
    marker = Path(audio_path)
    if not marker.exists():
        marker.touch()
        os._exit(1)  # simulate a hard crash (e.g. CUDA segfault) - no Python cleanup runs
    return [{"word": "ok", "start": 0.0, "end": 0.5, "speaker": None}]


def fake_transcribe_stall_once(models, audio_path, cfg):
    marker = Path(audio_path)
    if not marker.exists():
        marker.touch()
        time.sleep(5)  # simulate a hang; the supervisor should kill us well before this returns
    return [{"word": "ok", "start": 0.0, "end": 0.5, "speaker": None}]


async def test_pool_processes_task_end_to_end(tmp_path):
    cfg = AsrConfig()
    pool = AsrWorkerPool(
        cfg,
        [{"worker_id": "w0", "device": "cpu", "gpu_id": None}],
        model_loader=fake_loader,
        transcribe_fn=fake_transcribe_simple,
    )
    pool.start()
    try:
        task = make_task("vid1", 0, "hello:1.0:1.5|world:1.5:2.0", superchunk_start_s=10.0)
        result = await asyncio.wait_for(pool.submit(task), timeout=10)
        assert result.error is None
        assert result.video_id == "vid1"
        assert result.superchunk_idx == 0
        assert [w["word"] for w in result.words] == ["hello", "world"]
        # Local ASR timestamps must be shifted to global video time by the super-chunk's start offset.
        assert result.words[0]["start"] == pytest.approx(11.0)
        assert result.words[0]["end"] == pytest.approx(11.5)
    finally:
        await pool.stop()


async def test_pool_reports_transcribe_error_without_crashing_worker(tmp_path):
    cfg = AsrConfig()
    pool = AsrWorkerPool(
        cfg,
        [{"worker_id": "w0", "device": "cpu", "gpu_id": None}],
        model_loader=fake_loader,
        transcribe_fn=fake_transcribe_with_error_marker,
    )
    pool.start()
    try:
        bad_task = make_task("vid1", 0, "POISON", superchunk_start_s=0.0)
        result = await asyncio.wait_for(pool.submit(bad_task), timeout=10)
        assert result.error is not None
        assert result.words is None

        good_task = make_task("vid1", 1, "hi:0.0:0.5", superchunk_start_s=0.0)
        result2 = await asyncio.wait_for(pool.submit(good_task), timeout=10)
        assert result2.error is None
        assert result2.words[0]["word"] == "hi"
    finally:
        await pool.stop()


async def test_pool_detects_crash_and_requeues_task(tmp_path):
    cfg = AsrConfig()
    pool = AsrWorkerPool(
        cfg,
        [{"worker_id": "w0", "device": "cpu", "gpu_id": None}],
        model_loader=fake_loader,
        transcribe_fn=fake_transcribe_crash_once,
        heartbeat_poll_s=0.05,
        stall_timeout_s=999,
    )
    pool.start()
    try:
        marker = tmp_path / "task.marker"
        task = make_task("vid1", 0, str(marker), superchunk_start_s=0.0)
        result = await asyncio.wait_for(pool.submit(task), timeout=15)
        assert result.error is None
        assert result.words[0]["word"] == "ok"
    finally:
        await pool.stop()


async def test_pool_detects_stall_and_requeues_task(tmp_path):
    cfg = AsrConfig()
    pool = AsrWorkerPool(
        cfg,
        [{"worker_id": "w0", "device": "cpu", "gpu_id": None}],
        model_loader=fake_loader,
        transcribe_fn=fake_transcribe_stall_once,
        heartbeat_poll_s=0.05,
        stall_timeout_s=0.3,
    )
    pool.start()
    try:
        marker = tmp_path / "task.marker"
        task = make_task("vid1", 0, str(marker), superchunk_start_s=0.0)
        result = await asyncio.wait_for(pool.submit(task), timeout=15)
        assert result.error is None
        assert result.words[0]["word"] == "ok"
    finally:
        await pool.stop()
