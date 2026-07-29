import json
import subprocess

import pytest

from jinpipe.config import PackageConfig
from jinpipe.stages.package import (
    build_manifest,
    dataset_stats,
    filter_speakers,
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


def test_slice_segment_audio_defaults_to_native_passthrough_no_resample_args(tmp_path):
    out_path = tmp_path / "seg.flac"
    captured = {}

    def fake_runner(args):
        captured["args"] = args
        (tmp_path / "seg.flac.tmp").write_bytes(b"FLAC-DATA")
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=b"")

    slice_segment_audio(tmp_path / "source.wav", out_path, 1.0, 2.5, "flac", runner=fake_runner)

    assert "-ar" not in captured["args"]
    assert "-ac" not in captured["args"]


def test_slice_segment_audio_passes_explicit_sample_rate_and_channels(tmp_path):
    out_path = tmp_path / "seg.flac"
    captured = {}

    def fake_runner(args):
        captured["args"] = args
        (tmp_path / "seg.flac.tmp").write_bytes(b"FLAC-DATA")
        return subprocess.CompletedProcess(args, returncode=0, stdout=b"", stderr=b"")

    slice_segment_audio(
        tmp_path / "source.wav", out_path, 1.0, 2.5, "flac", sample_rate=22050, channels=1, runner=fake_runner
    )

    args = captured["args"]
    assert args[args.index("-ar") + 1] == "22050"
    assert args[args.index("-ac") + 1] == "1"


def test_slice_segment_audio_raises_on_ffmpeg_failure(tmp_path):
    def failing_runner(args):
        return subprocess.CompletedProcess(args, returncode=1, stdout=b"", stderr=b"boom")

    with pytest.raises(RuntimeError, match="ffmpeg slice failed"):
        slice_segment_audio(tmp_path / "source.wav", tmp_path / "seg.flac", 0.0, 1.0, "flac", runner=failing_runner)


def make_segment(idx=0, start=0.0, end=2.0, text="Hello world.", speaker="A"):
    words = [Word(text="Hello", start=start, end=start + 0.5), Word(text="world.", start=start + 0.5, end=end)]
    return Segment(idx=idx, start=start, end=end, text=text, words=words, speaker=speaker, exceeds_max_duration=False)


def test_package_segment_writes_json_and_skips_reslice_if_audio_exists(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    source_wav = tmp_path / "source.wav"
    cfg = PackageConfig(audio_format="flac")
    segment = make_segment()

    calls = []

    def fake_slice_fn(source, out_path, start, end, audio_format, sample_rate=None, channels=None):
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


def test_package_segment_word_clip_timestamps_are_relative_to_the_sliced_audio(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")
    # Segment starts partway through the video, so clip-relative timestamps
    # must differ from the video-relative start/end already on each word.
    segment = make_segment(start=100.0, end=102.0)

    def fake_slice_fn(source, out_path, start, end, audio_format, sample_rate=None, channels=None):
        out_path.write_bytes(b"fake-audio")

    packaged = package_segment(
        "vid1", "https://youtu.be/vid1", segment, tmp_path / "source.wav", output_dir, cfg, slice_fn=fake_slice_fn
    )

    metadata = json.loads(packaged.json_path.read_text())
    first, second = metadata["words"]
    assert first["start"] == pytest.approx(100.0)
    assert first["clip_start"] == pytest.approx(0.0)
    assert first["clip_end"] == pytest.approx(0.5)
    assert second["start"] == pytest.approx(100.5)
    assert second["clip_start"] == pytest.approx(0.5)
    assert second["clip_end"] == pytest.approx(2.0)


def test_build_and_write_manifest_reflects_disk_state(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")

    def fake_slice_fn(source, out_path, start, end, audio_format, sample_rate=None, channels=None):
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


def _pack(output_dir, cfg, video_id, idx, start, end, speaker):
    def fake_slice_fn(source, out_path, start, end, audio_format, sample_rate=None, channels=None):
        out_path.write_bytes(b"fake-audio")

    seg = make_segment(idx=idx, start=start, end=end, speaker=speaker)
    return package_segment(video_id, f"https://youtu.be/{video_id}", seg, output_dir / "source.wav", output_dir, cfg, slice_fn=fake_slice_fn)


def test_dataset_stats_aggregates_per_speaker_and_totals(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")

    _pack(output_dir, cfg, "vid1", 0, 0.0, 2.0, "A")
    _pack(output_dir, cfg, "vid1", 1, 2.0, 5.0, "B")
    _pack(output_dir, cfg, "vid2", 0, 0.0, 4.0, "A")

    stats = dataset_stats(output_dir)
    assert stats["num_segments"] == 3
    assert stats["num_videos"] == 2
    assert stats["num_speakers"] == 2
    assert stats["total_duration_s"] == pytest.approx(9.0)
    assert stats["per_speaker"]["A"] == {"segments": 2, "duration_s": pytest.approx(6.0)}
    assert stats["per_speaker"]["B"] == {"segments": 1, "duration_s": pytest.approx(3.0)}


def test_dataset_stats_on_empty_output_dir(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()

    stats = dataset_stats(output_dir)
    assert stats["num_segments"] == 0
    assert stats["num_videos"] == 0
    assert stats["num_speakers"] == 0
    assert stats["total_duration_s"] == 0.0
    assert stats["per_speaker"] == {}


def test_filter_speakers_dry_run_leaves_files_untouched(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")

    _pack(output_dir, cfg, "vid1", 0, 0.0, 2.0, "A")
    _pack(output_dir, cfg, "vid1", 1, 2.0, 5.0, "B")

    result = filter_speakers(output_dir, {"A"}, dry_run=True)
    assert result == {"kept": 1, "removed": 1}
    assert len(list(output_dir.glob("*.json"))) == 2
    assert len(list(output_dir.glob("*.flac"))) == 2


def test_filter_speakers_deletes_audio_and_json_for_excluded_speakers(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")

    _pack(output_dir, cfg, "vid1", 0, 0.0, 2.0, "A")
    _pack(output_dir, cfg, "vid1", 1, 2.0, 5.0, "B")
    _pack(output_dir, cfg, "vid2", 0, 0.0, 4.0, "A")

    result = filter_speakers(output_dir, {"A"}, dry_run=False)
    assert result == {"kept": 2, "removed": 1}

    remaining = build_manifest(output_dir)
    assert sorted(e["segment_id"] for e in remaining) == ["vid1_00000", "vid2_00000"]
    assert not (output_dir / "vid1_00001.json").exists()
    assert not (output_dir / "vid1_00001.flac").exists()


def test_filter_speakers_keeps_everything_when_no_speaker_excluded(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")

    _pack(output_dir, cfg, "vid1", 0, 0.0, 2.0, "A")
    _pack(output_dir, cfg, "vid1", 1, 2.0, 5.0, "B")

    result = filter_speakers(output_dir, {"A", "B"}, dry_run=False)
    assert result == {"kept": 2, "removed": 0}
    assert len(list(output_dir.glob("*.json"))) == 2
