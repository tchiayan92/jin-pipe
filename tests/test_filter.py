from pathlib import Path

from jinpipe.config import FilterConfig
from jinpipe.stages.filter import filter_segment


def fake_scorer_factory(ovr):
    def scorer(audio_path, model_path):
        return {"sig": 3.0, "bak": 3.0, "ovr": ovr}

    return scorer


def test_filter_disabled_passes_everything():
    cfg = FilterConfig(enabled=False, min_dnsmos_ovr=5.0, min_duration_s=100.0)
    result = filter_segment(Path("x.flac"), duration_s=0.1, cfg=cfg)
    assert result.passed is True


def test_filter_rejects_short_duration():
    cfg = FilterConfig(min_duration_s=2.0, min_dnsmos_ovr=None)
    result = filter_segment(Path("x.flac"), duration_s=1.0, cfg=cfg)
    assert result.passed is False
    assert "duration" in result.reason


def test_filter_rejects_disallowed_language():
    cfg = FilterConfig(allowed_languages=["en", "ta"], min_dnsmos_ovr=None)
    result = filter_segment(Path("x.flac"), duration_s=5.0, cfg=cfg, language="fr")
    assert result.passed is False
    assert "language" in result.reason


def test_filter_allows_language_in_allowlist():
    cfg = FilterConfig(allowed_languages=["en", "ta"], min_dnsmos_ovr=None)
    result = filter_segment(Path("x.flac"), duration_s=5.0, cfg=cfg, language="ta")
    assert result.passed is True


def test_filter_skips_dnsmos_when_threshold_is_none():
    cfg = FilterConfig(min_dnsmos_ovr=None)

    def unused_scorer(audio_path, model_path):
        raise AssertionError("should not be called when min_dnsmos_ovr is None")

    result = filter_segment(Path("x.flac"), duration_s=5.0, cfg=cfg, scorer=unused_scorer)
    assert result.passed is True


def test_filter_rejects_below_dnsmos_threshold():
    cfg = FilterConfig(min_dnsmos_ovr=3.0, min_duration_s=None)
    result = filter_segment(Path("x.flac"), duration_s=5.0, cfg=cfg, scorer=fake_scorer_factory(2.5))
    assert result.passed is False
    assert result.dnsmos_ovr == 2.5
    assert "threshold" in result.reason


def test_filter_passes_above_dnsmos_threshold():
    cfg = FilterConfig(min_dnsmos_ovr=3.0, min_duration_s=None)
    result = filter_segment(Path("x.flac"), duration_s=5.0, cfg=cfg, scorer=fake_scorer_factory(3.5))
    assert result.passed is True
    assert result.dnsmos_ovr == 3.5
    assert result.reason is None


def test_filter_reports_failed_scoring():
    cfg = FilterConfig(min_dnsmos_ovr=3.0, min_duration_s=None)

    def broken_scorer(audio_path, model_path):
        return {"sig": None, "bak": None, "ovr": None}

    result = filter_segment(Path("x.flac"), duration_s=5.0, cfg=cfg, scorer=broken_scorer)
    assert result.passed is False
    assert "failed" in result.reason
