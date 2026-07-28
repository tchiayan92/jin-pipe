import pytest

from jinpipe.config import VadConfig
from jinpipe.stages.vad import apply_overlap, coarse_segment, merge_speech_regions


def region(start, end):
    return {"start": start, "end": end}


def test_merge_speech_regions_empty():
    assert merge_speech_regions([], max_superchunk_s=90.0) == []


def test_merge_speech_regions_merges_short_regions_under_cap():
    regions = [region(0.0, 10.0), region(12.0, 20.0), region(25.0, 40.0)]
    merged = merge_speech_regions(regions, max_superchunk_s=90.0)
    assert merged == [(0.0, 40.0)]


def test_merge_speech_regions_splits_when_cap_exceeded_by_merge():
    # Each region is short, but merging all three would span 100s > cap of 50s.
    regions = [region(0.0, 20.0), region(25.0, 45.0), region(48.0, 60.0)]
    merged = merge_speech_regions(regions, max_superchunk_s=50.0)
    # First two merge (0->45, span 45 <= 50); adding the third would make span 60 > 50, so it starts a new chunk.
    assert merged == [(0.0, 45.0), (48.0, 60.0)]


def test_merge_speech_regions_force_splits_a_single_overlong_region():
    regions = [region(0.0, 200.0)]
    merged = merge_speech_regions(regions, max_superchunk_s=90.0)
    assert len(merged) == 3  # ceil(200/90) == 3
    # Pieces are contiguous and cover the full span with no gaps or overlaps.
    assert merged[0][0] == 0.0
    assert merged[-1][1] == 200.0
    for i in range(len(merged) - 1):
        assert merged[i][1] == pytest.approx(merged[i + 1][0])
    for start, end in merged:
        assert end - start <= 90.0 + 1e-9


def test_merge_speech_regions_zero_cap_disables_splitting():
    regions = [region(0.0, 200.0)]
    merged = merge_speech_regions(regions, max_superchunk_s=0)
    assert merged == [(0.0, 200.0)]


def test_apply_overlap_pads_only_internal_boundaries():
    cut_points = [(0.0, 30.0), (30.0, 60.0), (60.0, 90.0)]
    padded = apply_overlap(cut_points, overlap_s=2.5, duration_s=90.0)
    assert padded[0] == (0.0, 32.5)  # first chunk: no left pad, right pad added
    assert padded[1] == (27.5, 62.5)  # middle chunk: both sides padded
    assert padded[2] == (57.5, 90.0)  # last chunk: no right pad (clamped to duration)


def test_apply_overlap_clamps_to_audio_bounds():
    cut_points = [(0.0, 5.0), (5.0, 8.0)]
    padded = apply_overlap(cut_points, overlap_s=10.0, duration_s=8.0)
    assert padded[0][0] == 0.0  # never goes negative
    assert padded[-1][1] == 8.0  # never exceeds actual duration


def test_apply_overlap_single_chunk_untouched():
    padded = apply_overlap([(0.0, 10.0)], overlap_s=2.5, duration_s=10.0)
    assert padded == [(0.0, 10.0)]


def test_force_split_pieces_genuinely_overlap_after_padding():
    # Contiguous force-split pieces (no natural gap) must end up with real
    # audio overlap once padded - this is the exact scenario overlap exists for.
    regions = [region(0.0, 200.0)]
    cut_points = merge_speech_regions(regions, max_superchunk_s=90.0)
    padded = apply_overlap(cut_points, overlap_s=2.5, duration_s=200.0)
    for i in range(len(padded) - 1):
        assert padded[i][1] > padded[i + 1][0]  # chunk i's padded end reaches past chunk i+1's padded start


def test_coarse_segment_integration_with_injected_vad(tmp_path):
    fake_regions = [region(0.0, 10.0), region(95.0, 100.0)]

    def fake_speech_timestamps(wav_path, cfg):
        return fake_regions

    cfg = VadConfig(max_superchunk_s=90.0, overlap_s=2.0)
    result = coarse_segment(tmp_path / "audio.wav", cfg, duration_s=100.0, speech_timestamps_fn=fake_speech_timestamps)

    assert [idx for idx, _, _ in result] == [0, 1]
    idx0, start0, end0 = result[0]
    idx1, start1, end1 = result[1]
    assert start0 == 0.0
    assert end1 == 100.0
    # Padding is applied at the internal boundary even though these two speech
    # regions are far apart - it only actually creates overlapping audio when
    # the natural gap is smaller than 2x overlap_s (see the force-split test).
    assert end0 == pytest.approx(12.0)
    assert start1 == pytest.approx(93.0)
