"""Tests for viewer.py's pure data-prep helpers (no Gradio/gradio_annotated_audio import)."""

from jinpipe.config import PackageConfig
from jinpipe.stages.package import package_segment
from jinpipe.stages.rechunk import Segment, Word
from jinpipe.viewer import (
    ALL,
    audio_path_for,
    load_entries,
    overlap_ranges,
    segment_choices,
    segment_info_markdown,
    segment_label,
    transcript_for,
    video_and_speaker_choices,
)


def _pack(output_dir, cfg, video_id, idx, start, end, speaker, words=None, dnsmos_ovr=None):
    def fake_slice_fn(source, out_path, start, end, audio_format, sample_rate=None, channels=None):
        out_path.write_bytes(b"fake-audio")

    if words is None:
        words = [Word(text="hi", start=start, end=end, speaker=speaker)]
    seg = Segment(idx=idx, start=start, end=end, text="hi", words=words, speaker=speaker, exceeds_max_duration=False)
    return package_segment(
        video_id,
        f"https://youtu.be/{video_id}",
        seg,
        output_dir / "source.wav",
        output_dir,
        cfg,
        dnsmos_ovr=dnsmos_ovr,
        slice_fn=fake_slice_fn,
    )


def test_load_entries_and_audio_path_for(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")
    _pack(output_dir, cfg, "vid1", 0, 0.0, 2.0, "A")

    entries = load_entries(output_dir)
    assert set(entries) == {"vid1_00000"}
    audio_path = audio_path_for(output_dir, entries["vid1_00000"])
    assert audio_path == output_dir / "vid1_00000.flac"


def test_audio_path_for_missing_audio_returns_none(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    entry = {"segment_id": "ghost_00000"}
    assert audio_path_for(output_dir, entry) is None


def test_overlap_ranges_merges_consecutive_overlap_words():
    words = [
        {"clip_start": 0.0, "clip_end": 0.5, "overlap": False},
        {"clip_start": 0.5, "clip_end": 1.0, "overlap": True},
        {"clip_start": 1.0, "clip_end": 1.5, "overlap": True},
        {"clip_start": 1.5, "clip_end": 2.0, "overlap": False},
        {"clip_start": 2.0, "clip_end": 2.5, "overlap": True},
    ]
    assert overlap_ranges(words) == [
        {"start": 0.5, "end": 1.5},
        {"start": 2.0, "end": 2.5},
    ]


def test_overlap_ranges_empty_when_no_overlap():
    words = [{"clip_start": 0.0, "clip_end": 0.5, "overlap": False}]
    assert overlap_ranges(words) == []


def test_overlap_ranges_trailing_overlap_word_is_closed_out():
    words = [{"clip_start": 0.0, "clip_end": 0.5, "overlap": True}]
    assert overlap_ranges(words) == [{"start": 0.0, "end": 0.5}]


def test_segment_label_includes_dnsmos_when_present():
    entry = {"segment_id": "vid1_00000", "speaker": "A", "duration": 2.5, "dnsmos_ovr": 3.14159}
    assert segment_label(entry) == "vid1_00000  |  A  |  2.5s, dnsmos=3.14"


def test_segment_label_defaults_unknown_speaker_and_omits_dnsmos():
    entry = {"segment_id": "vid1_00000", "speaker": None, "duration": 2.0, "dnsmos_ovr": None}
    assert segment_label(entry) == "vid1_00000  |  UNKNOWN  |  2.0s"


def test_video_and_speaker_choices_are_sorted_and_prefixed_with_all(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")
    _pack(output_dir, cfg, "vid2", 0, 0.0, 2.0, "B")
    _pack(output_dir, cfg, "vid1", 0, 0.0, 2.0, "A")

    entries = load_entries(output_dir)
    videos, speakers = video_and_speaker_choices(entries)
    assert videos == [ALL, "vid1", "vid2"]
    assert speakers == [ALL, "A", "B"]


def test_segment_choices_filters_by_video_and_speaker(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")
    _pack(output_dir, cfg, "vid1", 0, 0.0, 2.0, "A")
    _pack(output_dir, cfg, "vid1", 1, 2.0, 4.0, "B")
    _pack(output_dir, cfg, "vid2", 0, 0.0, 2.0, "A")
    entries = load_entries(output_dir)

    all_choices = segment_choices(entries, ALL, ALL)
    assert [c[1] for c in all_choices] == ["vid1_00000", "vid1_00001", "vid2_00000"]

    vid1_only = segment_choices(entries, "vid1", ALL)
    assert [c[1] for c in vid1_only] == ["vid1_00000", "vid1_00001"]

    speaker_a_only = segment_choices(entries, ALL, "A")
    assert [c[1] for c in speaker_a_only] == ["vid1_00000", "vid2_00000"]

    both = segment_choices(entries, "vid1", "B")
    assert [c[1] for c in both] == ["vid1_00001"]


def test_transcript_for_builds_word_level_clip_relative_rows(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")
    words = [
        Word(text="Hello", start=100.0, end=100.5, speaker="A"),
        Word(text="world", start=100.5, end=102.0, speaker="A"),
    ]
    packaged = _pack(output_dir, cfg, "vid1", 0, 100.0, 102.0, "A", words=words)
    entry = load_entries(output_dir)[packaged.segment_id]

    transcript = transcript_for(entry)
    assert transcript == [
        {"start": 0.0, "end": 0.5, "text": "Hello", "speaker": "A"},
        {"start": 0.5, "end": 2.0, "text": "world", "speaker": "A"},
    ]


def test_segment_info_markdown_includes_optional_fields(tmp_path):
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    cfg = PackageConfig(audio_format="flac")
    packaged = _pack(output_dir, cfg, "vid1", 0, 0.0, 2.0, "A", dnsmos_ovr=3.5)
    entry = load_entries(output_dir)[packaged.segment_id]
    entry["language"] = "en"

    md = segment_info_markdown(entry)
    assert "vid1_00000" in md
    assert "speaker: `A`" in md
    assert "dnsmos_ovr: 3.50" in md
    assert "language: en" in md
