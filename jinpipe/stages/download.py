"""Download stage: fetch best-audio for each discovered video via yt-dlp.

Concurrency is bounded by a fixed pool of worker coroutines pulling from a
bounded queue (not one task per video) - this is the direct backpressure
mechanism: if downloads outpace downstream standardize/VAD/ASR stages, the
bounded work queue fills up and the discover stage naturally stalls on
`queue.put()` instead of piling up hundreds of raw audio files on disk.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from jinpipe.config import DownloadConfig
from jinpipe.stages.discover import DiscoverError, DiscoveredVideo

logger = logging.getLogger(__name__)

_SENTINEL = object()


@dataclass(frozen=True)
class DownloadResult:
    video_id: str
    channel: str | None
    raw_path: Path
    duration_s: float | None


@dataclass(frozen=True)
class DownloadError:
    video_id: str
    url: str
    error: str


async def _run_subprocess(args: list[str], timeout_s: float) -> tuple[int, bytes, bytes]:
    proc = await asyncio.create_subprocess_exec(
        *args, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    try:
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout_s)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        raise TimeoutError(f"command timed out after {timeout_s}s: {' '.join(args)}") from None
    return proc.returncode, stdout, stderr


SubprocessRunnerT = Callable[[list[str], float], Awaitable[tuple[int, bytes, bytes]]]


async def _probe_duration(path: Path, runner: SubprocessRunnerT) -> float | None:
    args = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(path),
    ]
    try:
        returncode, stdout, _ = await runner(args, 30.0)
    except Exception:  # noqa: BLE001 - duration is best-effort, never fatal
        return None
    if returncode != 0:
        return None
    try:
        return float(stdout.decode().strip())
    except (ValueError, AttributeError):
        return None


async def download_audio(
    video: DiscoveredVideo,
    output_dir: Path,
    cfg: DownloadConfig,
    *,
    runner: SubprocessRunnerT = _run_subprocess,
) -> DownloadResult:
    output_dir.mkdir(parents=True, exist_ok=True)
    out_template = str(output_dir / f"{video.video_id}.%(ext)s")
    args = [
        "yt-dlp",
        "-f",
        "bestaudio/best",
        "-x",
        "--audio-format",
        cfg.audio_format,
        "--no-keep-video",
        "--no-playlist",
        "-o",
        out_template,
        "--print",
        "after_move:filepath",
    ]
    if cfg.rate_limit:
        args += ["--limit-rate", cfg.rate_limit]
    args.append(video.url)

    returncode, stdout, stderr = await runner(args, float(cfg.timeout_s))
    if returncode != 0:
        raise RuntimeError(
            f"yt-dlp download failed for {video.video_id} (exit {returncode}): "
            f"{stderr.decode(errors='replace').strip()}"
        )

    lines = [line.strip() for line in stdout.decode(errors="replace").splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"yt-dlp produced no output path for {video.video_id}")
    raw_path = Path(lines[-1])
    duration_s = await _probe_duration(raw_path, runner)
    return DownloadResult(video_id=video.video_id, channel=video.channel, raw_path=raw_path, duration_s=duration_s)


async def download_all(
    videos: AsyncIterable[DiscoveredVideo],
    output_dir: Path,
    cfg: DownloadConfig,
    *,
    runner: SubprocessRunnerT = _run_subprocess,
    queue_maxsize: int = 32,
) -> AsyncIterator[DownloadResult | DownloadError]:
    work_queue: asyncio.Queue = asyncio.Queue(maxsize=queue_maxsize)
    result_queue: asyncio.Queue = asyncio.Queue()
    num_workers = max(1, cfg.max_concurrency)

    async def feeder() -> None:
        async for item in videos:
            if isinstance(item, DiscoverError):
                continue
            await work_queue.put(item)
        for _ in range(num_workers):
            await work_queue.put(_SENTINEL)

    async def worker() -> None:
        while True:
            video = await work_queue.get()
            if video is _SENTINEL:
                work_queue.task_done()
                break
            try:
                result: DownloadResult | DownloadError = await download_audio(
                    video, output_dir, cfg, runner=runner
                )
            except Exception as exc:  # noqa: BLE001 - reported to caller, not raised
                logger.warning("download failed for %s: %s", video.video_id, exc)
                result = DownloadError(video_id=video.video_id, url=video.url, error=str(exc))
            await result_queue.put(result)
            work_queue.task_done()

    feeder_task = asyncio.create_task(feeder())
    worker_tasks = [asyncio.create_task(worker()) for _ in range(num_workers)]

    async def closer() -> None:
        await feeder_task
        await asyncio.gather(*worker_tasks)
        await result_queue.put(_SENTINEL)

    closer_task = asyncio.create_task(closer())

    while True:
        item = await result_queue.get()
        if item is _SENTINEL:
            break
        yield item

    await closer_task
