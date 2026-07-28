"""Sentence-safe rechunk stage: the core of JinPipe's "never cut mid-sentence"
guarantee. Pure CPU logic, no model involved.

Operates at the video level, not the super-chunk level: super-chunks from the
VAD stage overlap by design (see stages/vad.py), so before any sentence
splitting happens, word-timestamp streams from all of a video's super-chunks
are merged into one continuous stream and the overlap regions are
de-duplicated. Only then are sentence boundaries computed and words packed
into final output segments - if this were done per-super-chunk instead, any
sentence straddling a super-chunk boundary would be truncated at exactly the
seam the VAD stage's overlap was designed to protect against.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple

from jinpipe.config import RechunkConfig

# Sentence-final punctuation across scripts: Latin/most Latin-alphabet languages,
# CJK full-width forms, Arabic, Devanagari danda, ellipsis.
SENTENCE_END_CHARS = ".!?。!?…؟।"

_WORD_CHARS_RE = re.compile(r"[^\w]", flags=re.UNICODE)


@dataclass(frozen=True)
class Word:
    text: str
    start: float
    end: float
    speaker: str | None = None


@dataclass(frozen=True)
class SuperchunkWords:
    idx: int
    start: float
    end: float
    words: list[Word]


@dataclass(frozen=True)
class Segment:
    idx: int
    start: float
    end: float
    text: str
    words: list[Word]
    speaker: str | None
    exceeds_max_duration: bool


class _Tagged(NamedTuple):
    word: Word
    edge_dist: float


def _normalize(text: str) -> str:
    return _WORD_CHARS_RE.sub("", text).lower()


def _iou(a: Word, b: Word) -> float:
    inter = max(0.0, min(a.end, b.end) - max(a.start, b.start))
    union = max(a.end, b.end) - min(a.start, b.start)
    return inter / union if union > 0 else 0.0


def _tag_words(chunk: SuperchunkWords) -> list[_Tagged]:
    return [
        _Tagged(word=w, edge_dist=min(w.start - chunk.start, chunk.end - w.end)) for w in chunk.words
    ]


def _resolve_overlap(trailing: list[_Tagged], leading: list[_Tagged]) -> list[_Tagged]:
    """Reconcile words from two super-chunks that both cover the same overlap window.

    A trailing word (from the earlier chunk) and a leading word (from the later
    chunk) are treated as the same spoken word if their time spans substantially
    overlap and their normalized text matches. Of a matched pair, keep whichever
    occurrence is farther from its own chunk's edge, since alignment is least
    reliable in the first/last second or two of a chunk. Unmatched words (seen
    by only one side) are kept as-is.
    """
    used_leading: set[int] = set()
    resolved: list[_Tagged] = []
    for t in trailing:
        match_idx = None
        for j, candidate in enumerate(leading):
            if j in used_leading:
                continue
            if _iou(t.word, candidate.word) > 0.4 and _normalize(t.word.text) == _normalize(candidate.word.text):
                match_idx = j
                break
        if match_idx is not None:
            candidate = leading[match_idx]
            used_leading.add(match_idx)
            resolved.append(t if t.edge_dist >= candidate.edge_dist else candidate)
        else:
            resolved.append(t)
    for j, candidate in enumerate(leading):
        if j not in used_leading:
            resolved.append(candidate)
    resolved.sort(key=lambda tw: tw.word.start)
    return resolved


def merge_superchunk_words(superchunks: list[SuperchunkWords]) -> list[Word]:
    """Concatenate a video's super-chunk word streams into one, de-duplicating overlaps."""
    if not superchunks:
        return []

    ordered = sorted(superchunks, key=lambda c: c.idx)
    merged: list[_Tagged] = list(_tag_words(ordered[0]))
    prev_chunk = ordered[0]

    for chunk in ordered[1:]:
        overlap_start = chunk.start
        overlap_end = prev_chunk.end
        chunk_tagged = _tag_words(chunk)

        if overlap_end <= overlap_start:
            merged.extend(chunk_tagged)
            prev_chunk = chunk
            continue

        outside = [t for t in merged if t.word.end <= overlap_start]
        trailing = [t for t in merged if t.word.end > overlap_start]
        leading = [t for t in chunk_tagged if t.word.start < overlap_end]
        remainder = [t for t in chunk_tagged if t.word.start >= overlap_end]

        resolved = _resolve_overlap(trailing, leading)
        merged = outside + resolved + remainder
        prev_chunk = chunk

    return [t.word for t in merged]


def _is_punct_boundary(word: Word) -> bool:
    text = word.text.strip()
    return bool(text) and text[-1] in SENTENCE_END_CHARS


def _find_boundaries(words: list[Word], silence_fallback_ms: int) -> set[int]:
    """Indices i such that a cut is allowed immediately after words[i].

    Punctuation is the primary signal. A large inter-word silence gap is always
    treated as an additional, independent boundary signal (not just a fallback
    for punctuation-less languages) - a real pause is strong evidence a
    sentence ended there even if ASR punctuation missed it.
    """
    boundaries: set[int] = set()
    gap_threshold = silence_fallback_ms / 1000.0
    for i, w in enumerate(words):
        if _is_punct_boundary(w):
            boundaries.add(i)
            continue
        if i + 1 < len(words) and (words[i + 1].start - w.end) >= gap_threshold:
            boundaries.add(i)
    if words:
        boundaries.add(len(words) - 1)
    return boundaries


def _split_into_sentences(words: list[Word], boundaries: set[int]) -> list[list[Word]]:
    sentences: list[list[Word]] = []
    cur: list[Word] = []
    for i, w in enumerate(words):
        cur.append(w)
        if i in boundaries:
            sentences.append(cur)
            cur = []
    if cur:
        sentences.append(cur)
    return sentences


def _pack_sentences(sentences: list[list[Word]], max_segment_s: float) -> list[list[Word]]:
    """Greedily pack sentences into segments, never splitting a sentence.

    A single sentence longer than max_segment_s is kept whole in its own
    segment rather than ever being cut mid-sentence - the hard rule this whole
    module exists to enforce.
    """
    packed: list[list[Word]] = []
    cur: list[Word] = []
    for sentence in sentences:
        if cur:
            proposed_duration = sentence[-1].end - cur[0].start
            if proposed_duration > max_segment_s:
                packed.append(cur)
                cur = list(sentence)
                continue
        cur.extend(sentence)
    if cur:
        packed.append(cur)
    return packed


def _enforce_min_duration(
    segments: list[list[Word]], min_segment_s: float, max_segment_s: float
) -> list[list[Word]]:
    """Merge segments shorter than min_segment_s into a neighbor when it fits within max_segment_s.

    A short segment with no neighbor it can merge into without exceeding
    max_segment_s is left as-is - still never cut mid-sentence, just short.
    """
    result: list[list[Word]] = []
    i = 0
    n = len(segments)
    while i < n:
        seg = segments[i]
        duration = seg[-1].end - seg[0].start
        if duration >= min_segment_s or n == 1:
            result.append(seg)
            i += 1
            continue

        if result:
            merged_duration = seg[-1].end - result[-1][0].start
            if merged_duration <= max_segment_s:
                result[-1] = result[-1] + seg
                i += 1
                continue

        if i + 1 < n:
            merged_duration = segments[i + 1][-1].end - seg[0].start
            if merged_duration <= max_segment_s:
                segments[i + 1] = seg + segments[i + 1]
                i += 1
                continue

        result.append(seg)
        i += 1
    return result


def _build_segment(idx: int, words: list[Word], max_segment_s: float) -> Segment:
    start = words[0].start
    end = words[-1].end
    duration = end - start
    text = " ".join(w.text for w in words).strip()
    speakers = [w.speaker for w in words if w.speaker]
    speaker = Counter(speakers).most_common(1)[0][0] if speakers else None
    exceeds = duration > max_segment_s + 1e-6
    return Segment(idx=idx, start=start, end=end, text=text, words=words, speaker=speaker, exceeds_max_duration=exceeds)


def rechunk_video(superchunks: list[SuperchunkWords], cfg: RechunkConfig) -> list[Segment]:
    merged_words = merge_superchunk_words(superchunks)
    if not merged_words:
        return []
    boundaries = _find_boundaries(merged_words, cfg.silence_fallback_ms)
    sentences = _split_into_sentences(merged_words, boundaries)
    packed = _pack_sentences(sentences, cfg.max_segment_s)
    packed = _enforce_min_duration(packed, cfg.min_segment_s, cfg.max_segment_s)
    return [_build_segment(i, words, cfg.max_segment_s) for i, words in enumerate(packed)]
