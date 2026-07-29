from __future__ import annotations

from pathlib import Path

import pytest

from jinpipe.db import JobStore
from jinpipe.stages.discover import DiscoverError, DiscoveredVideo
from jinpipe.stages.local import local_results, pending_and_fresh_local_files, scan_local_audio


async def _collect(aiter):
    return [item async for item in aiter]


def _touch(path: Path) -> Path:
    path.write_bytes(b"fake-audio")
    return path


# ---------------------------------------------------------------------------
# scan_local_audio
# ---------------------------------------------------------------------------


def test_scan_local_audio_filters_by_extension_and_sorts(tmp_path):
    _touch(tmp_path / "b.wav")
    _touch(tmp_path / "a.mp3")
    _touch(tmp_path / "notes.txt")

    result = scan_local_audio(tmp_path)

    assert [p.name for p in result] == ["a.mp3", "b.wav"]


def test_scan_local_audio_ignores_subdirectories(tmp_path):
    _touch(tmp_path / "top.mp3")
    sub = tmp_path / "sub"
    sub.mkdir()
    _touch(sub / "nested.mp3")

    result = scan_local_audio(tmp_path)

    assert [p.name for p in result] == ["top.mp3"]


def test_scan_local_audio_missing_dir_returns_empty():
    assert scan_local_audio(Path("/no/such/dir")) == []


# ---------------------------------------------------------------------------
# pending_and_fresh_local_files: resume semantics mirroring discover.py
# ---------------------------------------------------------------------------


async def test_pending_and_fresh_local_files_registers_new_files_as_pending(tmp_path):
    _touch(tmp_path / "interview.mp3")
    db_path = tmp_path / "jobs.sqlite3"
    store = JobStore(db_path)
    try:
        results = await _collect(pending_and_fresh_local_files(store, tmp_path))
        assert results == [
            DiscoveredVideo(video_id="interview", channel=None, url=str(tmp_path / "interview.mp3"))
        ]
        video = store.get_video("interview")
        assert video["status"] == "PENDING"
    finally:
        store.close()


async def test_pending_and_fresh_local_files_skips_already_known_videos(tmp_path):
    _touch(tmp_path / "interview.mp3")
    db_path = tmp_path / "jobs.sqlite3"
    store = JobStore(db_path)
    try:
        store.add_video("interview", str(tmp_path / "interview.mp3"), channel=None)
        store.update_video("interview", status="DONE")

        results = await _collect(pending_and_fresh_local_files(store, tmp_path))

        assert results == []
    finally:
        store.close()


async def test_pending_and_fresh_local_files_yields_leftover_pending_before_scanning(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    store = JobStore(db_path)
    try:
        # Simulates a prior interrupted run: registered PENDING but the file
        # is no longer (or never was) on disk under that name.
        store.add_video("stale", "https://old/path.mp3", channel=None)

        results = await _collect(pending_and_fresh_local_files(store, tmp_path))

        assert results == [DiscoveredVideo(video_id="stale", channel=None, url="https://old/path.mp3")]
    finally:
        store.close()


# ---------------------------------------------------------------------------
# local_results: DiscoveredVideo -> DownloadResult/DownloadError
# ---------------------------------------------------------------------------


async def _fake_runner_ok(args, timeout_s):
    return 0, b"42.5\n", b""


async def _fake_runner_fails(args, timeout_s):
    return 1, b"", b"ffprobe blew up"


async def _as_stream(items):
    for item in items:
        yield item


async def test_local_results_probes_duration_for_existing_file(tmp_path):
    path = _touch(tmp_path / "interview.mp3")
    video = DiscoveredVideo(video_id="interview", channel=None, url=str(path))

    results = await _collect(local_results(_as_stream([video]), runner=_fake_runner_ok))

    assert len(results) == 1
    assert results[0].video_id == "interview"
    assert results[0].raw_path == path
    assert results[0].duration_s == pytest.approx(42.5)


async def test_local_results_missing_file_yields_download_error():
    video = DiscoveredVideo(video_id="ghost", channel=None, url="/no/such/file.mp3")

    results = await _collect(local_results(_as_stream([video])))

    assert len(results) == 1
    assert results[0].video_id == "ghost"
    assert "not found" in results[0].error


async def test_local_results_duration_none_when_probe_fails(tmp_path):
    path = _touch(tmp_path / "interview.mp3")
    video = DiscoveredVideo(video_id="interview", channel=None, url=str(path))

    results = await _collect(local_results(_as_stream([video]), runner=_fake_runner_fails))

    assert results[0].duration_s is None


async def test_local_results_skips_discover_errors_passed_through():
    results = await _collect(local_results(_as_stream([DiscoverError(channel="x", error="boom")])))
    assert results == []
