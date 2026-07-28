import pytest

from jinpipe.config import DiscoverConfig, SourcesConfig
from jinpipe.stages.discover import (
    DiscoverError,
    DiscoveredVideo,
    ad_hoc_videos,
    discover_all,
    discover_sources,
    extract_video_id,
)


@pytest.mark.parametrize(
    "url_or_id,expected",
    [
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ("https://www.youtube.com/watch?v=dQw4w9WgXcQ&t=30s", "dQw4w9WgXcQ"),
    ],
)
def test_extract_video_id(url_or_id, expected):
    assert extract_video_id(url_or_id) == expected


def test_extract_video_id_raises_on_garbage():
    with pytest.raises(ValueError):
        extract_video_id("https://example.com/not-a-youtube-link")


def test_ad_hoc_videos():
    sources = SourcesConfig(video_urls=["https://youtu.be/AAAAAAAAAAA", "BBBBBBBBBBB"])
    videos = ad_hoc_videos(sources)
    assert [v.video_id for v in videos] == ["AAAAAAAAAAA", "BBBBBBBBBBB"]
    assert all(v.channel is None for v in videos)


async def test_discover_all_merges_channels():
    async def fake_runner(channel_url: str) -> list[str]:
        return {"chan-a": ["a1", "a2"], "chan-b": ["b1"]}[channel_url]

    cfg = DiscoverConfig(max_concurrency=2, request_delay_s=0)
    items = [item async for item in discover_all(["chan-a", "chan-b"], cfg, runner=fake_runner)]

    assert all(isinstance(i, DiscoveredVideo) for i in items)
    ids = {i.video_id for i in items}
    assert ids == {"a1", "a2", "b1"}
    for item in items:
        assert item.url.endswith(item.video_id)


async def test_discover_all_reports_channel_error_without_raising():
    async def fake_runner(channel_url: str) -> list[str]:
        if channel_url == "bad-chan":
            raise RuntimeError("network exploded")
        return ["ok1"]

    cfg = DiscoverConfig(max_concurrency=2, request_delay_s=0)
    items = [item async for item in discover_all(["bad-chan", "good-chan"], cfg, runner=fake_runner)]

    errors = [i for i in items if isinstance(i, DiscoverError)]
    videos = [i for i in items if isinstance(i, DiscoveredVideo)]
    assert len(errors) == 1
    assert errors[0].channel == "bad-chan"
    assert [v.video_id for v in videos] == ["ok1"]


async def test_discover_all_empty_channel_list_yields_nothing():
    cfg = DiscoverConfig()

    async def unused_runner(channel_url: str) -> list[str]:
        raise AssertionError("should never be called")

    items = [item async for item in discover_all([], cfg, runner=unused_runner)]
    assert items == []


async def test_discover_sources_combines_ad_hoc_and_channels():
    sources = SourcesConfig(video_urls=["CCCCCCCCCCC"], channels=["chan-a"])

    async def fake_runner(channel_url: str) -> list[str]:
        return ["a1"]

    cfg = DiscoverConfig(request_delay_s=0)
    items = [item async for item in discover_sources(sources, cfg, runner=fake_runner)]
    ids = {i.video_id for i in items}
    assert ids == {"CCCCCCCCCCC", "a1"}
