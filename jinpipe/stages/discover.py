"""Discover stage: list video ids from YouTube channels/playlists via yt-dlp.

Each channel is resolved concurrently (bounded by a semaphore + a politeness
delay to avoid YouTube throttling), and results are merged into a single
bounded stream of DiscoveredVideo as they arrive - callers don't wait for the
slowest channel before seeing videos from a faster one.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections.abc import AsyncIterator, Awaitable, Callable
from dataclasses import dataclass

from jinpipe.config import DiscoverConfig, SourcesConfig

logger = logging.getLogger(__name__)

_SENTINEL = object()

_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})")


@dataclass(frozen=True)
class DiscoveredVideo:
    video_id: str
    channel: str | None
    url: str


@dataclass(frozen=True)
class DiscoverError:
    channel: str
    error: str


def extract_video_id(url_or_id: str) -> str:
    """Pull an 11-character YouTube video id out of a URL, or pass an id through."""
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url_or_id):
        return url_or_id
    match = _VIDEO_ID_RE.search(url_or_id)
    if not match:
        raise ValueError(f"could not extract a video id from {url_or_id!r}")
    return match.group(1)


async def _run_yt_dlp_flat_playlist(channel_url: str, timeout_s: float = 300.0) -> list[str]:
    proc = await asyncio.create_subprocess_exec(
        "yt-dlp",
        "--flat-playlist",
        "--print",
        "id",
        channel_url,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise RuntimeError(f"yt-dlp --flat-playlist timed out for {channel_url}") from None

    if proc.returncode != 0:
        raise RuntimeError(
            f"yt-dlp --flat-playlist failed for {channel_url} (exit {proc.returncode}): "
            f"{stderr.decode(errors='replace').strip()}"
        )
    return [line.strip() for line in stdout.decode(errors="replace").splitlines() if line.strip()]


RunnerT = Callable[[str], Awaitable[list[str]]]


async def _discover_channel(
    channel_url: str,
    semaphore: asyncio.Semaphore,
    request_delay_s: float,
    runner: RunnerT,
) -> tuple[list[DiscoveredVideo], DiscoverError | None]:
    async with semaphore:
        try:
            ids = await runner(channel_url)
        except Exception as exc:  # noqa: BLE001 - reported to caller, not raised
            logger.warning("discover failed for channel %s: %s", channel_url, exc)
            return [], DiscoverError(channel=channel_url, error=str(exc))
        finally:
            if request_delay_s > 0:
                await asyncio.sleep(request_delay_s)

    videos = [
        DiscoveredVideo(video_id=vid, channel=channel_url, url=f"https://www.youtube.com/watch?v={vid}")
        for vid in ids
    ]
    return videos, None


async def discover_all(
    channels: list[str],
    cfg: DiscoverConfig,
    *,
    runner: RunnerT = _run_yt_dlp_flat_playlist,
    queue_maxsize: int = 256,
) -> AsyncIterator[DiscoveredVideo | DiscoverError]:
    """Yield DiscoveredVideo (or DiscoverError for a failed channel) as they resolve."""
    if not channels:
        return

    semaphore = asyncio.Semaphore(max(1, cfg.max_concurrency))
    queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)

    async def producer(channel_url: str) -> None:
        videos, error = await _discover_channel(channel_url, semaphore, cfg.request_delay_s, runner)
        for video in videos:
            await queue.put(video)
        if error is not None:
            await queue.put(error)

    tasks = [asyncio.create_task(producer(c)) for c in channels]

    async def closer() -> None:
        await asyncio.gather(*tasks)
        await queue.put(_SENTINEL)

    closer_task = asyncio.create_task(closer())

    while True:
        item = await queue.get()
        if item is _SENTINEL:
            break
        yield item

    await closer_task


def ad_hoc_videos(sources: SourcesConfig) -> list[DiscoveredVideo]:
    """Videos listed explicitly by URL/id in config, bypassing channel discovery."""
    videos = []
    for url in sources.video_urls:
        video_id = extract_video_id(url)
        videos.append(DiscoveredVideo(video_id=video_id, channel=None, url=url))
    return videos


async def discover_sources(
    sources: SourcesConfig,
    cfg: DiscoverConfig,
    *,
    runner: RunnerT = _run_yt_dlp_flat_playlist,
) -> AsyncIterator[DiscoveredVideo | DiscoverError]:
    """Combine ad-hoc video URLs with channel discovery into one stream."""
    for video in ad_hoc_videos(sources):
        yield video
    async for item in discover_all(sources.channels, cfg, runner=runner):
        yield item
