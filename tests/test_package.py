import json
import subprocess

import pytest

from jinpipe.config import PackageConfig
from jinpipe.stages.package import (
    build_manifest,
    package_segment,
    segment_id_for,
    slice_segment_audio,
    write_manifest,
)
from jinpipe.stages.rechunk import Segment, Word


def test_segment_id_for_zero_pads_index():
    assert segment_id_for("abc123", 0) == "abc123_00000"
    assert segment_id_for("abc123", 42) == "abc123_00042"


def test_slice_segment_audio_builds_expected_args_and_atomic_rename(tmp_path):
    out_path = tmp_path / "seg.flac"

    def fake_runner(args):
        # The tmp file must exist before rename; simulate ffmpeg producing it.
        tmp_index = args.index("-i")
        assert args[0] == "ffmpeg"
        assert "-ss" in args and "-to" in args
        (tmp_path / "seg.flac.tmp").write_bytes(b"FLAC-DATA")
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=b"")

    result = slice_segment_audio(tmp_path / "source.wav", out_path, 1.0, 2.5, "flac", runner=fake_runner)
    assert result == out_path
    assert out_path.exists()
    assert not (tmp_path / "seg.flac.tmp").exists()


def test_slice_segment_audio_raises_on_ffmpeg_failure(tmp_path):
    def failing_runner(args):
        return subprocess.CompletedProcess(args, returncode=1, stdout=b"", stderr=b"boom")

    with pytest.raises(RuntimeError, match="ffmpeg slice failed"):
        slice_segment_audio(tmp_path / "source.wav", tmp_path / "seg.flac", 0.0, 1.0, "flac", runner=failing_runner)


def make_segment(idx=0, start=0.0, end=2.0, text="Hello world."):
    words = [Word(text="Hello", start=start, end=start + 0.5), Word(text="world.", start=start + 0.5, end=end)]
    return Segment(idx=idx, start=start, end=end, text=text, words=words, speaker="A", exceeds_max_duration=False)


def test_package_segment_writes_json_and_skips_reslice_if_audio_exists(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    source_wav = tmp_path / "source.wav"
    cfg = PackageConfig(audio_format="flac")
    segment = make_segment()

    calls = []

    def fake_slice_fn(source, out_path, start, end, audio_format):
        calls.append(out_path)
        out_path.write_bytes(b"fake-audio")

    packaged = package_segment(
        "vid1", "https://youtu.be/vid1", segment, source_wav, output_dir, cfg, slice_fn=fake_slice_fn
    )
    assert packaged.segment_id == "vid1_00000"
    assert packaged.audio_path.exists()
    assert packaged.json_path.exists()
    assert len(calls) == 1

    metadata = json.loads(packaged.json_path.read_text())
    assert metadata["text"] == "Hello world."
    assert metadata["speaker"] == "A"
    assert metadata["exceeds_max_duration"] is False
    assert len(metadata["words"]) == 2

    # Re-running (as on resume) must not re-slice audio that already exists.
    package_segment("vid1", "https://youtu.be/vid1", segment, source_wav, output_dir, cfg, slice_fn=fake_slice_fn)
    assert len(calls) == 1


def test_build_and_write_manifest_reflects_disk_state(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")

    def fake_slice_fn(source, out_path, start, end, audio_format):
        out_path.write_bytes(b"fake-audio")

    for i in range(3):
        seg = make_segment(idx=i, start=float(i * 3), end=float(i * 3 + 2))
        package_segment("vid1", "https://youtu.be/vid1", seg, tmp_path / "source.wav", output_dir, cfg, slice_fn=fake_slice_fn)

    entries = build_manifest(output_dir)
    assert len(entries) == 3
    assert [e["segment_id"] for e in entries] == ["vid1_00000", "vid1_00001", "vid1_00002"]

    manifest_path = tmp_path / "manifest.jsonl"
    count = write_manifest(output_dir, manifest_path)
    assert count == 3
    lines = manifest_path.read_text().strip().splitlines()
    assert len(lines) == 3
    assert json.loads(lines[0])["segment_id"] == "vid1_00000"

    # Rebuilding after adding a segment must pick it up (idempotent, ground-truth derived).
    seg3 = make_segment(idx=3, start=10.0, end=12.0)
    package_segment("vid1", "https://youtu.be/vid1", seg3, tmp_path / "source.wav", output_dir, cfg, slice_fn=fake_slice_fn)
    count2 = write_manifest(output_dir, manifest_path)
    assert count2 == 4
