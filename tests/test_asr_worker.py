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
from jinpipe.workers.asr_worker import (
    AsrWorkerPool,
    _hf_auth_kwarg,
    assign_speakers,
    compute_overlap_regions,
    make_diarize_task,
    make_task,
    word_in_overlap,
)


class _FakeModels:
    pass


def fake_loader(cfg, device, gpu_id):
    return _FakeModels()


def fake_transcribe_simple(models, audio_path, cfg):
    words = []
    for part in audio_path.split("|"):
        text, start, end = part.split(":")
        words.append({"word": text, "start": float(start), "end": float(end)})
    return words, "en"


def fake_transcribe_with_error_marker(models, audio_path, cfg):
    if audio_path == "POISON":
        raise RuntimeError("simulated transcribe failure")
    return fake_transcribe_simple(models, audio_path, cfg)


def fake_transcribe_crash_once(models, audio_path, cfg):
    marker = Path(audio_path)
    if not marker.exists():
        marker.touch()
        os._exit(1)  # simulate a hard crash (e.g. CUDA segfault) - no Python cleanup runs
    return [{"word": "ok", "start": 0.0, "end": 0.5}], "en"


def fake_transcribe_stall_once(models, audio_path, cfg):
    marker = Path(audio_path)
    if not marker.exists():
        marker.touch()
        time.sleep(5)  # simulate a hang; the supervisor should kill us well before this returns
    return [{"word": "ok", "start": 0.0, "end": 0.5}], "en"


def fake_diarize_fn(models, audio_path):
    return [
        {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
        {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
    ]


def fake_diarize_stall_once(models, audio_path):
    marker = Path(audio_path)
    count_file = marker.with_suffix(".count")
    n = (int(count_file.read_text()) + 1) if count_file.exists() else 1
    count_file.write_text(str(n))
    if n == 1:
        time.sleep(1.5)  # longer than a short asr stall_timeout_s, shorter than diarize_stall_timeout_s
    return [{"start": 0.0, "end": 1.0, "speaker": "SPEAKER_00"}]


# ---------------------------------------------------------------------------
# compute_overlap_regions / word_in_overlap: pure logic over raw diarization
# turns, exercised without pandas/pyannote installed.
# ---------------------------------------------------------------------------


def test_compute_overlap_regions_no_turns_or_single_turn_has_no_overlap():
    assert compute_overlap_regions([]) == []
    assert compute_overlap_regions([{"start": 0.0, "end": 5.0, "speaker": "A"}]) == []


def test_compute_overlap_regions_sequential_turns_have_no_overlap():
    turns = [
        {"start": 0.0, "end": 5.0, "speaker": "A"},
        {"start": 5.0, "end": 10.0, "speaker": "B"},
    ]
    assert compute_overlap_regions(turns) == []


def test_compute_overlap_regions_detects_simultaneous_speakers():
    # B starts talking at 4.0, well before A finishes at 6.0: [4.0, 6.0) is cross-talk.
    turns = [
        {"start": 0.0, "end": 6.0, "speaker": "A"},
        {"start": 4.0, "end": 9.0, "speaker": "B"},
    ]
    regions = compute_overlap_regions(turns)
    assert len(regions) == 1
    assert regions[0] == pytest.approx((4.0, 6.0))


def test_compute_overlap_regions_ignores_same_speaker_double_counted_turn():
    turns = [
        {"start": 0.0, "end": 6.0, "speaker": "A"},
        {"start": 4.0, "end": 9.0, "speaker": "A"},
    ]
    assert compute_overlap_regions(turns) == []


def test_word_in_overlap_checks_intersection_with_regions():
    regions = [(4.0, 6.0)]
    assert word_in_overlap(3.0, 4.5, regions) is True
    assert word_in_overlap(0.0, 2.0, regions) is False
    assert word_in_overlap(4.0, 6.0, []) is False


# ---------------------------------------------------------------------------
# assign_speakers: pure logic, words/turns must already share the same
# (global) time base - exercised without pandas/pyannote installed.
# ---------------------------------------------------------------------------


def test_assign_speakers_picks_turn_with_max_overlap():
    words = [{"start": 0.0, "end": 1.0}, {"start": 5.0, "end": 6.0}]
    turns = [
        {"start": 0.0, "end": 4.0, "speaker": "A"},
        {"start": 4.0, "end": 10.0, "speaker": "B"},
    ]
    assign_speakers(words, turns)
    assert words[0]["speaker"] == "A"
    assert words[1]["speaker"] == "B"


def test_assign_speakers_leaves_speaker_unset_on_zero_overlap_by_default():
    # Falls exactly in the silence gap between the two turns.
    words = [{"start": 4.0, "end": 4.5}]
    turns = [
        {"start": 0.0, "end": 4.0, "speaker": "A"},
        {"start": 5.0, "end": 10.0, "speaker": "B"},
    ]
    assign_speakers(words, turns)
    assert "speaker" not in words[0]


def test_assign_speakers_fill_nearest_guesses_closest_turn_by_midpoint():
    words = [{"start": 4.0, "end": 4.5}]
    turns = [
        {"start": 0.0, "end": 4.0, "speaker": "A"},
        {"start": 5.0, "end": 10.0, "speaker": "B"},
    ]
    assign_speakers(words, turns, fill_nearest=True)
    assert words[0]["speaker"] == "A"  # A's turn-midpoint (2.0) is closer to the word's (4.25) than B's (7.5)


def test_assign_speakers_no_turns_is_a_no_op():
    words = [{"start": 0.0, "end": 1.0}]
    assign_speakers(words, [])
    assert "speaker" not in words[0]


# ---------------------------------------------------------------------------
# _hf_auth_kwarg: picks the right HF-token kwarg across pyannote/whisperx
# versions that renamed use_auth_token -> token.
# ---------------------------------------------------------------------------


def test_hf_auth_kwarg_prefers_token_when_present():
    def ctor(self, token=None, device="cpu"):
        pass

    assert _hf_auth_kwarg(ctor) == "token"


def test_hf_auth_kwarg_falls_back_to_use_auth_token():
    def ctor(self, use_auth_token=None, device="cpu"):
        pass

    assert _hf_auth_kwarg(ctor) == "use_auth_token"


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
        assert result.language == "en"
    finally:
        await pool.stop()


async def test_pool_processes_diarize_task():
    cfg = AsrConfig()
    pool = AsrWorkerPool(
        cfg,
        [{"worker_id": "w0", "device": "cpu", "gpu_id": None}],
        model_loader=fake_loader,
        transcribe_fn=fake_transcribe_simple,
        diarize_fn=fake_diarize_fn,
    )
    pool.start()
    try:
        task = make_diarize_task("vid1", "std.wav")
        result = await asyncio.wait_for(pool.submit(task), timeout=10)
        assert result.error is None
        assert result.words is None
        assert result.turns == [
            {"start": 0.0, "end": 5.0, "speaker": "SPEAKER_00"},
            {"start": 5.0, "end": 10.0, "speaker": "SPEAKER_01"},
        ]
    finally:
        await pool.stop()


async def test_pool_assigns_speakers_from_diarize_turns_in_global_time():
    cfg = AsrConfig()
    pool = AsrWorkerPool(
        cfg,
        [{"worker_id": "w0", "device": "cpu", "gpu_id": None}],
        model_loader=fake_loader,
        transcribe_fn=fake_transcribe_simple,
    )
    pool.start()
    try:
        # Turns are in GLOBAL time; words are chunk-local (start at 1.0/1.5)
        # until shifted by superchunk_start_s=10.0 - the turns must line up
        # with the SHIFTED (global) word times, not the raw chunk-local ones.
        turns = [
            {"start": 10.0, "end": 11.5, "speaker": "SPEAKER_00"},
            {"start": 11.5, "end": 12.0, "speaker": "SPEAKER_01"},
        ]
        task = make_task("vid1", 0, "hello:1.0:1.5|world:1.5:2.0", superchunk_start_s=10.0, diarize_turns=turns)
        result = await asyncio.wait_for(pool.submit(task), timeout=10)
        assert result.error is None
        assert result.words[0]["speaker"] == "SPEAKER_00"
        assert result.words[1]["speaker"] == "SPEAKER_01"
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


async def test_pool_does_not_kill_long_running_diarize_task_under_short_asr_stall_timeout(tmp_path):
    """A whole-video diarize call can legitimately run past the short
    stall_timeout_s tuned for per-super-chunk ASR calls (only 0.3s here); it
    must be judged against diarize_stall_timeout_s instead, so the supervisor
    must NOT kill/requeue it mid-flight."""
    cfg = AsrConfig()
    pool = AsrWorkerPool(
        cfg,
        [{"worker_id": "w0", "device": "cpu", "gpu_id": None}],
        model_loader=fake_loader,
        transcribe_fn=fake_transcribe_simple,
        diarize_fn=fake_diarize_stall_once,
        heartbeat_poll_s=0.05,
        stall_timeout_s=0.3,
        diarize_stall_timeout_s=10.0,
    )
    pool.start()
    try:
        marker = tmp_path / "diarize.marker"
        task = make_diarize_task("vid1", str(marker))
        result = await asyncio.wait_for(pool.submit(task), timeout=15)
        assert result.error is None
        assert result.turns[0]["speaker"] == "SPEAKER_00"
        # If the supervisor had wrongly killed/requeued this task using the
        # short asr stall_timeout_s, fake_diarize_stall_once would have been
        # invoked a second time (count == 2).
        assert marker.with_suffix(".count").read_text() == "1"
    finally:
        await pool.stop()
