"""Local-folder stage: treat pre-existing audio files as already "downloaded",
so a folder of audio (e.g. a 1-hour interview recording) can be processed
without going anywhere near yt-dlp.

Mirrors discover.py + download.py's DiscoveredVideo -> DownloadResult shape so
it slots into the orchestrator's existing download-result loop unchanged:
scan_local_audio finds files, pending_and_fresh_local_files registers/resumes
them with the same PENDING-first-then-fresh semantics
_pending_and_fresh_videos uses for YouTube sources, and local_results turns
each into a DownloadResult (or DownloadError) the rest of the pipeline
already knows how to consume.
"""

from __future__ import annotations

from collections.abc import AsyncIterable, AsyncIterator
from pathlib import Path

from jinpipe.db import JobStore
from jinpipe.stages.discover import DiscoveredVideo, DiscoverError
from jinpipe.stages.download import DownloadError, DownloadResult, SubprocessRunnerT, _probe_duration, _run_subprocess

AUDIO_EXTENSIONS = (".mp3", ".wav", ".m4a", ".flac", ".ogg", ".opus", ".aac", ".mp4", ".webm")


def scan_local_audio(local_dir: Path, extensions: tuple[str, ...] = AUDIO_EXTENSIONS) -> list[Path]:
    """Sorted list of audio files directly inside local_dir (non-recursive)."""
    if not local_dir.is_dir():
        return []
    return sorted(p for p in local_dir.iterdir() if p.is_file() and p.suffix.lower() in extensions)


async def pending_and_fresh_local_files(store: JobStore, local_dir: Path) -> AsyncIterator[DiscoveredVideo]:
    """Leftover PENDING rows from a prior interrupted run first, then freshly
    scanned files - anything already known (DONE, RUNNING, FAILED, or already
    yielded as PENDING above) is skipped, matching discover.py's resume
    semantics for YouTube sources."""
    for video in store.list_videos(status="PENDING"):
        yield DiscoveredVideo(video_id=video["video_id"], channel=video["channel"], url=video["url"])

    for path in scan_local_audio(local_dir):
        video_id = path.stem
        if store.get_video(video_id) is not None:
            continue
        store.add_video(video_id, str(path), channel=None)
        yield DiscoveredVideo(video_id=video_id, channel=None, url=str(path))


async def local_results(
    videos: AsyncIterable[DiscoveredVideo | DiscoverError],
    *,
    runner: SubprocessRunnerT = _run_subprocess,
) -> AsyncIterator[DownloadResult | DownloadError]:
    """Turn each local-file DiscoveredVideo into a DownloadResult, ffprobing
    duration the same way the real download stage does - no yt-dlp involved."""
    async for video in videos:
        if isinstance(video, DiscoverError):
            continue
        path = Path(video.url)
        if not path.is_file():
            yield DownloadError(video_id=video.video_id, url=video.url, error=f"file not found: {path}")
            continue
        duration_s = await _probe_duration(path, runner)
        yield DownloadResult(video_id=video.video_id, channel=video.channel, raw_path=path, duration_s=duration_s)
