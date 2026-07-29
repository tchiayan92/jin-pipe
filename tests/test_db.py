import time

import pytest

from jinpipe.db import JobStore


@pytest.fixture()
def store(tmp_path):
    s = JobStore(tmp_path / "jobs.sqlite3")
    yield s
    s.close()


def test_add_video_is_idempotent(store):
    assert store.add_video("v1", "https://youtu.be/v1", channel="chan") is True
    assert store.add_video("v1", "https://youtu.be/v1", channel="chan") is False
    video = store.get_video("v1")
    assert video["status"] == "PENDING"
    assert video["channel"] == "chan"


def test_update_video_status(store):
    store.add_video("v1", "https://youtu.be/v1")
    store.update_video("v1", status="RUNNING")
    assert store.get_video("v1")["status"] == "RUNNING"
    store.update_video("v1", status="DONE", raw_path="/tmp/v1.m4a", duration_s=42.0)
    video = store.get_video("v1")
    assert video["status"] == "DONE"
    assert video["raw_path"] == "/tmp/v1.m4a"
    assert video["duration_s"] == 42.0


def test_superchunks_all_done(store):
    store.add_video("v1", "https://youtu.be/v1")
    store.add_superchunk("v1", 0, 0.0, 30.0)
    store.add_superchunk("v1", 1, 27.5, 60.0)

    assert store.superchunks_all_done("v1") is False

    store.update_superchunk("v1", 0, status="DONE", words_json="[]")
    assert store.superchunks_all_done("v1") is False

    store.update_superchunk("v1", 1, status="DONE", words_json="[]")
    assert store.superchunks_all_done("v1") is True


def test_superchunks_all_done_false_when_none_exist(store):
    store.add_video("v1", "https://youtu.be/v1")
    assert store.superchunks_all_done("v1") is False


def test_add_segment_is_idempotent_and_keyed_by_segment_id(store):
    store.add_video("v1", "https://youtu.be/v1")
    inserted = store.add_segment("v1", 0, "v1_00000", 0.0, 5.0, text="hello world")
    assert inserted is True
    inserted_again = store.add_segment("v1", 0, "v1_00000", 0.0, 5.0, text="different text")
    assert inserted_again is False

    segments = store.get_segments("v1")
    assert len(segments) == 1
    assert segments[0]["text"] == "hello world"


def test_recover_stale_running_resets_all_running_rows(store):
    store.add_video("v1", "https://youtu.be/v1")
    store.update_video("v1", status="RUNNING")
    store.add_superchunk("v1", 0, 0.0, 10.0)
    store.update_superchunk("v1", 0, status="RUNNING")

    reset = store.recover_stale_running(max_age_s=0)
    assert reset == 2
    assert store.get_video("v1")["status"] == "PENDING"
    assert store.get_superchunks("v1")[0]["status"] == "PENDING"


def test_recover_stale_running_respects_max_age(store):
    store.add_video("v1", "https://youtu.be/v1")
    store.update_video("v1", status="RUNNING")

    # Not stale yet under a generous max_age.
    reset = store.recover_stale_running(max_age_s=3600)
    assert reset == 0
    assert store.get_video("v1")["status"] == "RUNNING"


def test_reset_video_deletes_only_that_videos_state(store):
    store.add_video("v1", "https://youtu.be/v1")
    store.update_video("v1", status="DONE")
    store.add_superchunk("v1", 0, 0.0, 10.0)
    store.add_segment("v1", 0, "v1_00000", 0.0, 5.0)
    store.add_video("v2", "https://youtu.be/v2")
    store.update_video("v2", status="DONE")

    counts = store.reset_video("v1")

    assert counts == {"segments": 1, "superchunks": 1, "videos": 1}
    assert store.get_video("v1") is None
    assert store.get_superchunks("v1") == []
    assert store.get_segments("v1") == []
    assert store.get_video("v2") is not None


def test_reset_video_no_matching_rows_returns_zero_counts(store):
    counts = store.reset_video("ghost")
    assert counts == {"segments": 0, "superchunks": 0, "videos": 0}


def test_reset_all_deletes_every_video(store):
    store.add_video("v1", "https://youtu.be/v1")
    store.add_superchunk("v1", 0, 0.0, 10.0)
    store.add_segment("v1", 0, "v1_00000", 0.0, 5.0)
    store.add_video("v2", "https://youtu.be/v2")
    store.add_segment("v2", 0, "v2_00000", 0.0, 5.0)

    counts = store.reset_all()

    assert counts == {"segments": 2, "superchunks": 1, "videos": 2}
    assert store.list_videos() == []
    assert store.list_superchunks() == []
    assert store.list_segments() == []


def test_reset_all_on_empty_store_returns_zero_counts(store):
    assert store.reset_all() == {"segments": 0, "superchunks": 0, "videos": 0}


def test_status_counts(store):
    store.add_video("v1", "https://youtu.be/v1")
    store.add_video("v2", "https://youtu.be/v2")
    store.update_video("v2", status="DONE")

    counts = store.status_counts()
    assert counts["videos"]["PENDING"] == 1
    assert counts["videos"]["DONE"] == 1


def test_reopening_store_preserves_schema_and_data(tmp_path):
    db_path = tmp_path / "jobs.sqlite3"
    with JobStore(db_path) as s1:
        s1.add_video("v1", "https://youtu.be/v1")

    with JobStore(db_path) as s2:
        assert s2.get_video("v1") is not None
