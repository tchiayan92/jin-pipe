"""Pydantic configuration schema for the JinPipe pipeline, loaded from YAML."""

from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SourcesConfig(BaseModel):
    channels: list[str] = Field(default_factory=list)
    video_urls: list[str] = Field(default_factory=list)
    # When set, the pipeline processes every audio file directly inside this
    # folder instead of discovering/downloading from YouTube - channels and
    # video_urls are ignored for the run.
    local_dir: Path | None = None


class PathsConfig(BaseModel):
    work_dir: Path
    output_dir: Path
    db_path: Path


class DiscoverConfig(BaseModel):
    max_concurrency: int = 3
    request_delay_s: float = 5.0


class DownloadConfig(BaseModel):
    max_concurrency: int = 5
    audio_format: str = "m4a"
    rate_limit: str | None = None
    timeout_s: int = 900


class StandardizeConfig(BaseModel):
    sample_rate: int = 16000
    workers: int = 4


class VadConfig(BaseModel):
    threshold: float = 0.5
    min_silence_duration_ms: int = 400
    min_speech_duration_ms: int = 250
    speech_pad_ms: int = 100
    max_superchunk_s: float = 90.0
    overlap_s: float = 2.5
    workers: int = 4


class AsrConfig(BaseModel):
    model_size: str = "large-v3"
    device: str = "auto"
    compute_type: str = "auto"
    num_workers: int = 1
    language: str | None = None
    # Word-level timestamps always come from faster-whisper's own native
    # word_timestamps (works for any language Whisper supports). When True,
    # AND WhisperX has a wav2vec2 alignment model for the detected/configured
    # language, an additional forced-alignment pass refines those word
    # boundaries - purely a quality upgrade, applied automatically only when
    # available. Languages WhisperX has no alignment model for (e.g. Malay/
    # "ms") transparently keep faster-whisper's native timestamps either way.
    align: bool = True
    diarize: bool = False
    hf_token: str | None = None


class RechunkConfig(BaseModel):
    min_segment_s: float = 2.0
    max_segment_s: float = 20.0
    boundary_edge_guard_s: float = 1.5
    silence_fallback_ms: int = 250
    split_on_speaker_change: bool = True


class FilterConfig(BaseModel):
    enabled: bool = True
    min_dnsmos_ovr: float | None = 2.5
    min_duration_s: float | None = None
    allowed_languages: list[str] | None = None
    dnsmos_model_path: str | None = None
    reject_overlapping_speech: bool = True
    workers: int = 4


class PackageConfig(BaseModel):
    audio_format: str = "flac"
    # None = keep the source's native sample rate / channel count (no forced
    # resampling or downmixing) - set an explicit value only to shrink output
    # files at the cost of quality. Independent of standardize.sample_rate,
    # which only controls the internal 16kHz-mono copy VAD/ASR need.
    sample_rate: int | None = None
    channels: int | None = None


class ResourceConfig(BaseModel):
    ram_floor_mb: int = 1536
    vram_floor_mb: int = 1024
    # Explicit reservation-accounting capacity. None = derive at runtime from
    # total system RAM / per-GPU total VRAM minus the floor.
    ram_budget_mb: int | None = None
    vram_budget_mb: int | None = None
    poll_interval_s: float = 1.0
    cost_per_audio_second: dict[str, float] = Field(
        default_factory=lambda: {
            "standardize": 2_000_000.0,
            "vad": 3_000_000.0,
            "asr": 60_000_000.0,
            "diarize": 80_000_000.0,
            "filter": 1_000_000.0,
        }
    )


class QueueConfig(BaseModel):
    maxsize_per_stage: int = 32


class JinPipeConfig(BaseModel):
    sources: SourcesConfig
    paths: PathsConfig
    discover: DiscoverConfig = Field(default_factory=DiscoverConfig)
    download: DownloadConfig = Field(default_factory=DownloadConfig)
    standardize: StandardizeConfig = Field(default_factory=StandardizeConfig)
    vad: VadConfig = Field(default_factory=VadConfig)
    asr: AsrConfig = Field(default_factory=AsrConfig)
    rechunk: RechunkConfig = Field(default_factory=RechunkConfig)
    filter: FilterConfig = Field(default_factory=FilterConfig)
    package: PackageConfig = Field(default_factory=PackageConfig)
    resources: ResourceConfig = Field(default_factory=ResourceConfig)
    queues: QueueConfig = Field(default_factory=QueueConfig)


def load_config(path: str | Path) -> JinPipeConfig:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return JinPipeConfig.model_validate(raw)
