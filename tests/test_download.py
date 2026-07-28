from pathlib import Path

import pytest

from jinpipe.config import DownloadConfig
from jinpipe.stages.discover import DiscoverError, DiscoveredVideo
from jinpipe.stages.download import DownloadError, DownloadResult, download_all, download_audio


def make_video(video_id="v1", url=None, channel="chan"):
    return DiscoveredVideo(video_id=video_id, channel=channel, url=url or f"https://youtu.be/{video_id}")


def make_runner(raw_path: str, duration: float = 12.5, fail: bool = False):
    async def runner(args, timeout_s):
        if args[0] == "yt-dlp":
            if fail:
                return 1, b"", b"some yt-dlp error"
            return 0, f"{raw_path}\n".encode(), b""
        elif args[0] == "ffprobe":
            return 0, f"{duration}\n".encode(), b""
        raise AssertionError(f"unexpected command {args}")

    return runner


async def test_download_audio_success(tmp_path):
    raw_path = tmp_path / "v1.m4a"
    runner = make_runner(str(raw_path))
    cfg = DownloadConfig()
    result = await download_audio(make_video(), tmp_path, cfg, runner=runner)
    assert isinstance(result, DownloadResult)
    assert result.video_id == "v1"
    assert result.raw_path == raw_path
    assert result.duration_s == 12.5


async def test_download_audio_raises_on_nonzero_exit(tmp_path):
    runner = make_runner(str(tmp_path / "v1.m4a"), fail=True)
    cfg = DownloadConfig()
    with pytest.raises(RuntimeError, match="yt-dlp download failed"):
        await download_audio(make_video(), tmp_path, cfg, runner=runner)


async def test_download_audio_duration_probe_failure_is_non_fatal(tmp_path):
    async def runner(args, timeout_s):
        if args[0] == "yt-dlp":
            return 0, f"{tmp_path / 'v1.m4a'}\n".encode(), b""
        return 1, b"", b"ffprobe error"

    cfg = DownloadConfig()
    result = await download_audio(make_video(), tmp_path, cfg, runner=runner)
    assert result.duration_s is None


async def test_download_all_processes_stream_and_reports_errors(tmp_path):
    videos = [make_video("v1"), make_video("v2"), make_video("v3")]

    async def video_stream():
        for v in videos:
            yield v
        yield DiscoverError(channel="bad", error="boom")  # must be skipped, not crash download_all

    async def runner(args, timeout_s):
        if args[0] == "yt-dlp":
            out_index = args.index("-o") + 1
            video_id = Path(args[out_index]).name.split(".")[0]
            if video_id == "v2":
                return 1, b"", b"simulated failure"
            return 0, f"{tmp_path / (video_id + '.m4a')}\n".encode(), b""
        return 0, b"5.0\n", b""

    cfg = DownloadConfig(max_concurrency=2, timeout_s=30)
    results = [r async for r in download_all(video_stream(), tmp_path, cfg, runner=runner)]

    ok = [r for r in results if isinstance(r, DownloadResult)]
    errors = [r for r in results if isinstance(r, DownloadError)]
    assert {r.video_id for r in ok} == {"v1", "v3"}
    assert {r.video_id for r in errors} == {"v2"}
