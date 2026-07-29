import pytest

from jinpipe.config import RechunkConfig
from jinpipe.stages.rechunk import (
    Word,
    SuperchunkWords,
    _enforce_min_duration,
    merge_superchunk_words,
    rechunk_video,
)


def W(text, start, end, speaker=None, overlap=False):
    return Word(text=text, start=start, end=end, speaker=speaker, overlap=overlap)


# ---------------------------------------------------------------------------
# merge_superchunk_words: overlap resolution between adjacent super-chunks
# ---------------------------------------------------------------------------


def test_merge_no_superchunks_returns_empty():
    assert merge_superchunk_words([]) == []


def test_merge_single_superchunk_passthrough():
    chunk = SuperchunkWords(idx=0, start=0, end=10, words=[W("a", 1, 1.5), W("b", 2, 2.5)])
    assert merge_superchunk_words([chunk]) == [W("a", 1, 1.5), W("b", 2, 2.5)]


def test_merge_touching_chunks_with_zero_overlap_just_concatenates():
    chunk0 = SuperchunkWords(idx=0, start=0, end=10, words=[W("a", 1, 1.5), W("b", 2, 2.5)])
    chunk1 = SuperchunkWords(idx=1, start=10, end=20, words=[W("c", 11, 11.5)])
    result = merge_superchunk_words([chunk0, chunk1])
    assert [w.text for w in result] == ["a", "b", "c"]


def test_merge_dedups_overlap_preferring_occurrence_farther_from_its_own_edge():
    # "shared" falls in the genuine overlap window [8, 12] and both chunks
    # transcribe it with slightly different (jitter) timestamps.
    chunk0 = SuperchunkWords(idx=0, start=0, end=12, words=[W("early", 1, 1.5), W("shared", 10.5, 11.0)])
    chunk1 = SuperchunkWords(idx=1, start=8, end=20, words=[W("shared", 10.52, 11.02), W("later", 15, 15.5)])

    result = merge_superchunk_words([chunk0, chunk1])

    assert [w.text for w in result] == ["early", "shared", "later"]
    # chunk0's edge_dist for "shared" is min(10.5-0, 12-11.0) = 1.0
    # chunk1's edge_dist for "shared" is min(10.52-8, 20-11.02) = 2.52 -> chunk1 wins
    shared = result[1]
    assert shared.start == pytest.approx(10.52)
    assert shared.end == pytest.approx(11.02)


def test_merge_keeps_words_seen_by_only_one_side():
    chunk0 = SuperchunkWords(idx=0, start=0, end=12, words=[W("early", 1, 1.5), W("onlychunk0", 9, 9.5)])
    chunk1 = SuperchunkWords(idx=1, start=8, end=20, words=[W("onlychunk1", 10.0, 10.5), W("later", 15, 15.5)])

    result = merge_superchunk_words([chunk0, chunk1])

    assert [w.text for w in result] == ["early", "onlychunk0", "onlychunk1", "later"]


def test_merge_sorts_superchunks_by_idx_regardless_of_input_order():
    chunk0 = SuperchunkWords(idx=0, start=0, end=10, words=[W("a", 1, 1.5)])
    chunk1 = SuperchunkWords(idx=1, start=10, end=20, words=[W("b", 11, 11.5)])
    result = merge_superchunk_words([chunk1, chunk0])
    assert [w.text for w in result] == ["a", "b"]


# ---------------------------------------------------------------------------
# rechunk_video: sentence-boundary detection and packing
# ---------------------------------------------------------------------------


def test_rechunk_video_empty_input():
    cfg = RechunkConfig()
    assert rechunk_video([], cfg) == []


def test_rechunk_video_splits_on_punctuation():
    words = [W("Hello", 0, 0.5), W("world.", 0.5, 1.0), W("This", 1.5, 1.8), W("is", 1.8, 2.0), W("Jin.", 2.0, 2.5)]
    chunk = SuperchunkWords(idx=0, start=0, end=3.0, words=words)
    cfg = RechunkConfig(min_segment_s=0.1, max_segment_s=1.2, silence_fallback_ms=250)

    segments = rechunk_video([chunk], cfg)

    assert len(segments) == 2
    assert segments[0].text == "Hello world."
    assert segments[0].start == 0.0
    assert segments[0].end == 1.0
    assert segments[0].exceeds_max_duration is False
    assert segments[1].text == "This is Jin."
    assert segments[1].start == 1.5
    assert segments[1].end == 2.5


def test_rechunk_video_falls_back_to_silence_gap_when_no_punctuation():
    words = [W("hello", 0, 0.5), W("world", 0.5, 1.0), W("this", 1.5, 1.8), W("is", 1.8, 2.0), W("jin", 2.0, 2.5)]
    chunk = SuperchunkWords(idx=0, start=0, end=3.0, words=words)
    cfg = RechunkConfig(min_segment_s=0.1, max_segment_s=1.2, silence_fallback_ms=250)

    segments = rechunk_video([chunk], cfg)

    assert len(segments) == 2
    assert segments[0].text == "hello world"
    assert segments[0].end == 1.0
    assert segments[1].text == "this is jin"
    assert segments[1].start == 1.5


def test_rechunk_video_never_splits_a_sentence_spanning_a_superchunk_boundary():
    # chunk0 covers [0, 32.5] (natural VAD cut at 30, padded +2.5 overlap).
    # chunk1 covers [27.5, 60] (natural start 30, padded -2.5 overlap).
    # The sentence "This spans the boundary." (28.0-30.0) sits entirely inside
    # the overlap window and is transcribed independently by both chunks.
    chunk0 = SuperchunkWords(
        idx=0,
        start=0,
        end=32.5,
        words=[
            W("Prior", 0, 0.5),
            W("sentence.", 0.5, 1.0),
            W("This", 28.0, 28.3),
            W("spans", 28.4, 28.8),
            W("the", 28.9, 29.0),
            W("boundary.", 29.1, 30.0),
        ],
    )
    chunk1 = SuperchunkWords(
        idx=1,
        start=27.5,
        end=60.0,
        words=[
            W("This", 28.0, 28.3),
            W("spans", 28.4, 28.8),
            W("the", 28.9, 29.0),
            W("boundary.", 29.1, 30.0),
            W("Next", 35.0, 35.3),
            W("sentence.", 35.4, 36.0),
        ],
    )
    cfg = RechunkConfig(min_segment_s=0.5, max_segment_s=2.5, silence_fallback_ms=250)

    segments = rechunk_video([chunk0, chunk1], cfg)

    assert len(segments) == 3
    boundary_seg = segments[1]
    assert boundary_seg.text == "This spans the boundary."
    assert boundary_seg.start == 28.0
    assert boundary_seg.end == 30.0
    assert boundary_seg.exceeds_max_duration is False
    # No duplicated words: the boundary sentence's 4 words appear exactly once.
    assert sum(1 for s in segments for w in s.words if w.text == "boundary.") == 1


def test_rechunk_video_keeps_overlong_sentence_whole_and_flags_it():
    words = [
        W("This", 0, 2),
        W("is", 2, 4),
        W("a", 4, 6),
        W("very", 6, 8),
        W("long", 8, 10),
        W("runon.", 10, 15),
    ]
    chunk = SuperchunkWords(idx=0, start=0, end=16.0, words=words)
    cfg = RechunkConfig(min_segment_s=1.0, max_segment_s=5.0, silence_fallback_ms=250)

    segments = rechunk_video([chunk], cfg)

    assert len(segments) == 1
    assert segments[0].text == "This is a very long runon."
    assert segments[0].start == 0.0
    assert segments[0].end == 15.0
    assert segments[0].exceeds_max_duration is True


def test_rechunk_video_assigns_majority_speaker_within_one_speakers_words():
    words = [
        W("Hello.", 0, 0.5, speaker="A"),
        W("Yes.", 0.6, 1.0, speaker="A"),
    ]
    chunk = SuperchunkWords(idx=0, start=0, end=2.0, words=words)
    cfg = RechunkConfig(min_segment_s=0.1, max_segment_s=10.0, silence_fallback_ms=250)

    segments = rechunk_video([chunk], cfg)

    assert len(segments) == 1
    assert segments[0].speaker == "A"


def test_rechunk_video_splits_on_speaker_change_even_without_punctuation_or_silence():
    # No punctuation, no silence gap between "yes" and "ok" - only the speaker
    # change (A -> B) should force a boundary, so speaker A and B never end up
    # sharing one output segment/clip.
    words = [
        W("hello", 0, 0.5, speaker="A"),
        W("yes", 0.6, 1.0, speaker="A"),
        W("ok", 1.05, 1.5, speaker="B"),
        W("sure", 1.55, 2.0, speaker="B"),
    ]
    chunk = SuperchunkWords(idx=0, start=0, end=3.0, words=words)
    cfg = RechunkConfig(min_segment_s=0.1, max_segment_s=10.0, silence_fallback_ms=250, split_on_speaker_change=True)

    segments = rechunk_video([chunk], cfg)

    assert len(segments) == 2
    assert segments[0].text == "hello yes"
    assert segments[0].speaker == "A"
    assert segments[1].text == "ok sure"
    assert segments[1].speaker == "B"


def test_rechunk_video_split_on_speaker_change_disabled_preserves_majority_vote():
    words = [
        W("Hello.", 0, 0.5, speaker="A"),
        W("Yes.", 0.6, 1.0, speaker="A"),
        W("Ok.", 1.1, 1.5, speaker="B"),
    ]
    chunk = SuperchunkWords(idx=0, start=0, end=2.0, words=words)
    cfg = RechunkConfig(
        min_segment_s=0.1, max_segment_s=10.0, silence_fallback_ms=250, split_on_speaker_change=False
    )

    segments = rechunk_video([chunk], cfg)

    assert len(segments) == 1
    assert segments[0].speaker == "A"


def test_rechunk_video_flags_segment_with_overlapping_speech():
    words = [
        W("Hello.", 0, 0.5, speaker="A", overlap=False),
        W("world.", 0.6, 1.0, speaker="A", overlap=True),
    ]
    chunk = SuperchunkWords(idx=0, start=0, end=2.0, words=words)
    cfg = RechunkConfig(min_segment_s=0.1, max_segment_s=10.0, silence_fallback_ms=250)

    segments = rechunk_video([chunk], cfg)

    assert len(segments) == 1
    assert segments[0].has_overlap is True


def test_rechunk_video_no_overlap_flag_when_no_words_overlap():
    words = [W("Hello.", 0, 0.5, speaker="A"), W("world.", 0.6, 1.0, speaker="A")]
    chunk = SuperchunkWords(idx=0, start=0, end=2.0, words=words)
    cfg = RechunkConfig(min_segment_s=0.1, max_segment_s=10.0, silence_fallback_ms=250)

    segments = rechunk_video([chunk], cfg)

    assert segments[0].has_overlap is False


# ---------------------------------------------------------------------------
# _enforce_min_duration: merge-forward, merge-backward, and can't-merge paths
# ---------------------------------------------------------------------------


def test_enforce_min_duration_merges_short_segment_forward():
    segments = [
        [W("Hi.", 0, 0.3)],
        [W("Ok", 0.5, 0.6), W("go.", 0.6, 1.0)],
    ]
    result = _enforce_min_duration(segments, min_segment_s=0.5, max_segment_s=5.0)
    assert len(result) == 1
    assert [w.text for w in result[0]] == ["Hi.", "Ok", "go."]


def test_enforce_min_duration_merges_short_segment_backward():
    segments = [
        [W("First", 0, 1.0), W("sentence.", 1.0, 1.6)],
        [W("Hi.", 2.0, 2.3)],
    ]
    result = _enforce_min_duration(segments, min_segment_s=0.5, max_segment_s=5.0)
    assert len(result) == 1
    assert [w.text for w in result[0]] == ["First", "sentence.", "Hi."]


def test_enforce_min_duration_keeps_unmergeable_short_segment():
    segments = [
        [W("Long", 0, 3.0), W("one.", 3.0, 4.9)],
        [W("Hi.", 5.0, 5.3)],
        [W("Another", 5.5, 8.0), W("long", 8.0, 9.0), W("thing", 9.0, 9.8), W("here.", 9.8, 10.3)],
    ]
    result = _enforce_min_duration(segments, min_segment_s=0.5, max_segment_s=5.0)
    assert len(result) == 3
    assert [w.text for w in result[1]] == ["Hi."]


def test_enforce_min_duration_single_segment_always_kept():
    segments = [[W("Hi.", 0, 0.3)]]
    result = _enforce_min_duration(segments, min_segment_s=0.5, max_segment_s=5.0)
    assert result == segments
