"""Quality filter stage: duration/language gates plus DNSMOS P.835 scoring.

DNSMOS needs Microsoft's sig_bak_ovr.onnx model file, which is not bundled
here - obtain it from the DNS-Challenge repo per its license and point
`filter.dnsmos_model_path` at it. The scorer is injectable so the filtering
*logic* (thresholds, duration/language gates) is fully testable without the
real model; the default feature-extraction implementation follows the
publicly documented DNSMOS local-scoring approach (mel-spectrogram windows,
averaged) but should be checked against whichever model revision you use.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from jinpipe.config import FilterConfig

ScorerFnT = Callable[[Path, str | None], dict]

_session_cache: dict[str, object] = {}


def _get_session(model_path: str):
    if model_path not in _session_cache:
        import onnxruntime as ort

        _session_cache[model_path] = ort.InferenceSession(model_path, providers=["CPUExecutionProvider"])
    return _session_cache[model_path]


def _default_dnsmos_scorer(audio_path: Path, model_path: str | None) -> dict:
    if not model_path:
        raise RuntimeError("filter.min_dnsmos_ovr is set but filter.dnsmos_model_path is not configured")

    import librosa
    import numpy as np

    session = _get_session(model_path)
    sr = 16000
    audio, _ = librosa.load(str(audio_path), sr=sr)

    window_s, hop_s = 9.01, 1.0
    window_len = int(window_s * sr)
    hop_len = int(hop_s * sr)
    if len(audio) < window_len:
        audio = np.pad(audio, (0, window_len - len(audio)))

    scores = {"sig": [], "bak": [], "ovr": []}
    input_name = session.get_inputs()[0].name
    for start in range(0, max(1, len(audio) - window_len + 1), hop_len):
        segment = audio[start : start + window_len]
        mel = librosa.feature.melspectrogram(y=segment, sr=sr, n_fft=512, hop_length=int(sr / 100), n_mels=120)
        log_mel = (librosa.power_to_db(mel, ref=np.max) + 40) / 40
        features = log_mel.T.astype(np.float32)[np.newaxis, ...]
        raw = session.run(None, {input_name: features})[0][0]
        scores["sig"].append(float(raw[0]))
        scores["bak"].append(float(raw[1]))
        scores["ovr"].append(float(raw[2]))

    return {k: (sum(v) / len(v) if v else None) for k, v in scores.items()}


@dataclass(frozen=True)
class FilterResult:
    passed: bool
    dnsmos_sig: float | None
    dnsmos_bak: float | None
    dnsmos_ovr: float | None
    reason: str | None


def filter_segment(
    audio_path: Path,
    duration_s: float,
    cfg: FilterConfig,
    *,
    language: str | None = None,
    has_overlap: bool = False,
    scorer: ScorerFnT = _default_dnsmos_scorer,
) -> FilterResult:
    if not cfg.enabled:
        return FilterResult(True, None, None, None, None)

    if cfg.reject_overlapping_speech and has_overlap:
        return FilterResult(False, None, None, None, "overlapping multi-speaker speech detected")

    if cfg.min_duration_s is not None and duration_s < cfg.min_duration_s:
        return FilterResult(False, None, None, None, f"duration {duration_s:.2f}s below min_duration_s")

    if cfg.allowed_languages and language is not None and language not in cfg.allowed_languages:
        return FilterResult(False, None, None, None, f"language {language!r} not in allowed_languages")

    if cfg.min_dnsmos_ovr is None:
        return FilterResult(True, None, None, None, None)

    scores = scorer(audio_path, cfg.dnsmos_model_path)
    ovr = scores.get("ovr")
    if ovr is None:
        return FilterResult(False, scores.get("sig"), scores.get("bak"), None, "dnsmos scoring failed")
    if ovr < cfg.min_dnsmos_ovr:
        return FilterResult(
            False, scores.get("sig"), scores.get("bak"), ovr, f"dnsmos_ovr {ovr:.2f} below threshold"
        )
    return FilterResult(True, scores.get("sig"), scores.get("bak"), ovr, None)
