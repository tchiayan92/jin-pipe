"""Persistent ASR (+ optional diarization) worker pool.

Each worker is a long-lived multiprocessing.Process (one per GPU, or one for
CPU-only) that loads its WhisperX model(s) once at startup and then loops
pulling super-chunk tasks off a shared task queue - this avoids reloading
multi-GB models per task, which would dominate both throughput and memory
churn if done naively per call.

The pool bridges that sync multiprocessing world into asyncio via a
background thread that resolves asyncio Futures as results arrive, and
supervises worker liveness two ways: process exit-code/is_alive() checks
(catches hard crashes) and a heartbeat timestamp per in-flight task (catches
a worker that's still alive but hung, e.g. a driver-level stall, which
exit-code checks alone would miss). Either failure mode requeues the
in-flight task and spawns a replacement worker process.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import queue as _queue_module
import threading
import time
import uuid
from dataclasses import dataclass, field

import asyncio

from jinpipe.config import AsrConfig

logger = logging.getLogger(__name__)


@dataclass
class AsrTask:
    task_id: str
    video_id: str
    superchunk_idx: int
    audio_path: str
    superchunk_start_s: float


@dataclass
class AsrResult:
    task_id: str
    video_id: str
    superchunk_idx: int
    words: list[dict] | None
    error: str | None


@dataclass
class WorkerModels:
    whisper_model: object
    device: str
    diarize_pipeline: object | None = None
    align_models: dict = field(default_factory=dict)

    def get_align_model(self, language_code: str):
        import whisperx

        if language_code not in self.align_models:
            self.align_models[language_code] = whisperx.load_align_model(
                language_code=language_code, device=self.device
            )
        return self.align_models[language_code]


def compute_overlap_regions(turns: list[dict]) -> list[tuple[float, float]]:
    """Merged time ranges where >=2 distinct speakers' diarization turns intersect.

    pyannote's diarization output is turn-level (one row per speaker per
    contiguous stretch of speech) and, unlike whisperx's word-level
    ``assign_word_speakers`` (which collapses each word to a single nearest
    speaker and so throws this signal away), those turns can genuinely
    overlap in time when two speakers talk at once. A sweep line over the
    turn boundaries finds every sub-interval with >=2 simultaneously active
    speakers and merges adjacent ones.
    """
    if len(turns) < 2:
        return []
    points = sorted({t["start"] for t in turns} | {t["end"] for t in turns})
    regions: list[tuple[float, float]] = []
    region_start = None
    for a, b in zip(points, points[1:]):
        if b <= a:
            continue
        mid = (a + b) / 2
        active_speakers = {t["speaker"] for t in turns if t["start"] <= mid < t["end"]}
        if len(active_speakers) >= 2:
            if region_start is None:
                region_start = a
        elif region_start is not None:
            regions.append((region_start, a))
            region_start = None
    if region_start is not None:
        regions.append((region_start, points[-1]))
    return regions


def word_in_overlap(start: float, end: float, regions: list[tuple[float, float]]) -> bool:
    return any(min(end, r_end) - max(start, r_start) > 0 for r_start, r_end in regions)


def resolve_compute_type(cfg: AsrConfig, device: str) -> str:
    if cfg.compute_type != "auto":
        return cfg.compute_type
    return "float16" if device == "cuda" else "int8"


def build_worker_specs(cfg: AsrConfig, gpu_ids: list[int] | None) -> list[dict]:
    """One spec per persistent worker process: {worker_id, device, gpu_id}."""
    gpu_ids = gpu_ids or []
    use_gpu = gpu_ids and cfg.device in ("auto", "cuda")
    if not use_gpu:
        n = max(1, cfg.num_workers)
        return [{"worker_id": f"cpu-{i}", "device": "cpu", "gpu_id": None} for i in range(n)]
    ids = gpu_ids[: cfg.num_workers] if cfg.num_workers else gpu_ids
    return [{"worker_id": f"gpu-{gid}", "device": "cuda", "gpu_id": gid} for gid in ids]


def _hf_auth_kwarg(ctor) -> str:
    """Name of the HF-auth-token keyword arg a DiarizationPipeline constructor accepts.

    pyannote.audio (and whisperx's thin wrapper around it) renamed
    use_auth_token -> token following huggingface_hub's own deprecation of
    use_auth_token, so which spelling is valid depends on whichever
    pyannote.audio/whisperx version happens to be installed. Inspecting the
    live signature avoids hardcoding either spelling.
    """
    import inspect

    params = inspect.signature(ctor).parameters
    return "token" if "token" in params else "use_auth_token"


def _build_diarize_pipeline(hf_token: str | None, device: str):
    # Import from the whisperx.diarize submodule rather than the top-level
    # package: whether DiarizationPipeline is re-exported from
    # whisperx/__init__.py has changed across whisperx releases, but the
    # submodule itself has stayed put.
    from whisperx.diarize import DiarizationPipeline

    auth_kwarg = _hf_auth_kwarg(DiarizationPipeline.__init__)
    return DiarizationPipeline(device=device, **{auth_kwarg: hf_token})


def _default_model_loader(cfg: AsrConfig, device: str, gpu_id: int | None) -> WorkerModels:
    import os

    # pyannote.audio (pulled in transitively by whisperx) still calls
    # hf_hub_download(..., use_auth_token=...), a kwarg newer huggingface_hub
    # releases dropped in favor of `token`. pyannote's modules do
    # `from huggingface_hub import hf_hub_download` at their own import
    # time, so this must be patched before whisperx is first imported here -
    # patching huggingface_hub.hf_hub_download afterward would miss the
    # local references pyannote already bound.
    import huggingface_hub

    _orig_hf_hub_download = huggingface_hub.hf_hub_download

    def _hf_hub_download_compat(*args, **kwargs):
        if "use_auth_token" in kwargs:
            kwargs["token"] = kwargs.pop("use_auth_token")
        return _orig_hf_hub_download(*args, **kwargs)

    huggingface_hub.hf_hub_download = _hf_hub_download_compat

    import whisperx

    if gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    compute_type = resolve_compute_type(cfg, device)
    # whisperx's bundled pyannote VAD loader calls torch.load() on a
    # trusted HF checkpoint without accounting for PyTorch >=2.6's switch
    # to weights_only=True by default, which rejects the omegaconf config
    # objects baked into that checkpoint. Force weights_only=False for the
    # duration of model loading rather than allowlisting individual globals,
    # since the set of rejected globals isn't stable across releases.
    import torch

    _orig_torch_load = torch.load

    def _weights_only_false_load(*args, **kwargs):
        # Force-overwrite rather than setdefault: lightning_fabric's loader
        # passes weights_only=True explicitly (to match torch's new
        # default), so a mere setdefault never gets a chance to apply.
        kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    torch.load = _weights_only_false_load
    try:
        model = whisperx.load_model(cfg.model_size, device, compute_type=compute_type, language=cfg.language)
        diarize_pipeline = None
        if cfg.diarize:
            # Diarization's segmentation sub-model hits the same
            # weights_only issue as the VAD checkpoint above (tripping on a
            # different disallowed global, e.g. torch.torch_version.TorchVersion),
            # so it also needs to run under the patched torch.load.
            diarize_pipeline = _build_diarize_pipeline(cfg.hf_token, device)
    finally:
        torch.load = _orig_torch_load
    return WorkerModels(whisper_model=model, device=device, diarize_pipeline=diarize_pipeline)


def _whisperx_align_supported(language: str | None) -> bool:
    """Whether WhisperX ships a wav2vec2 CTC alignment model for this language.

    Coverage is far short of Whisper's own ~99 languages (e.g. no Malay/"ms")
    - this gates the *optional* alignment refinement, never whether word
    timestamps exist at all (those always come from faster-whisper directly).
    """
    if not language:
        return False
    from whisperx.alignment import DEFAULT_ALIGN_MODELS_HF, DEFAULT_ALIGN_MODELS_TORCH

    return language in DEFAULT_ALIGN_MODELS_TORCH or language in DEFAULT_ALIGN_MODELS_HF


def _default_transcribe(models: WorkerModels, audio_path: str, cfg: AsrConfig) -> list[dict]:
    """Transcribe one super-chunk. Returns word dicts with LOCAL (chunk-relative) timestamps.

    Word-level timestamps always come from faster-whisper's own native
    word_timestamps=True (Whisper's cross-attention-based alignment, works for
    any language Whisper supports). This deliberately bypasses WhisperX's
    batched FasterWhisperPipeline.transcribe() - that method's segment dicts
    never carry word-level data under any configuration; only
    whisperx.align() adds a "words" key, and that needs a language-specific
    wav2vec2 CTC model that doesn't exist for every language. models.whisper_model
    is that batched pipeline; models.whisper_model.model is the underlying,
    un-overridden faster_whisper.WhisperModel, whose original transcribe()
    we call directly here instead.
    """
    import whisperx

    audio = whisperx.load_audio(audio_path)
    raw_segments, info = models.whisper_model.model.transcribe(
        audio, language=cfg.language, word_timestamps=True
    )
    language = info.language
    result = {
        "language": language,
        "segments": [
            {
                "start": seg.start,
                "end": seg.end,
                "text": seg.text,
                "words": [
                    {"word": (w.word or "").strip(), "start": w.start, "end": w.end} for w in (seg.words or [])
                ],
            }
            for seg in raw_segments
        ],
    }

    if cfg.align and _whisperx_align_supported(language):
        align_model, metadata = models.get_align_model(language)
        result = whisperx.align(
            result["segments"], align_model, metadata, audio, models.device, return_char_alignments=False
        )
    elif cfg.align:
        logger.info(
            "no WhisperX alignment model for language %r - using faster-whisper's native word timestamps",
            language,
        )

    overlap_regions: list[tuple[float, float]] = []
    if models.diarize_pipeline is not None:
        from whisperx.diarize import assign_word_speakers

        diarize_segments = models.diarize_pipeline(audio)
        turns = diarize_segments[["start", "end", "speaker"]].to_dict("records")
        overlap_regions = compute_overlap_regions(turns)
        result = assign_word_speakers(diarize_segments, result)

    words = []
    for seg in result["segments"]:
        for w in seg.get("words", []):
            if "start" not in w or "end" not in w:
                # WhisperX occasionally drops timing for unaligned punctuation-only tokens.
                continue
            words.append(
                {
                    "word": (w.get("word") or "").strip(),
                    "start": w["start"],
                    "end": w["end"],
                    "speaker": w.get("speaker"),
                    "overlap": word_in_overlap(w["start"], w["end"], overlap_regions),
                }
            )
    return words


def _worker_main(task_queue, result_queue, heartbeat_queue, worker_id, device, gpu_id, model_loader, transcribe_fn, cfg):
    models = model_loader(cfg, device, gpu_id)
    while True:
        task = task_queue.get()
        if task is None:
            break
        heartbeat_queue.put((worker_id, task.task_id, "start", time.monotonic()))
        try:
            local_words = transcribe_fn(models, task.audio_path, cfg)
            global_words = [
                {**w, "start": w["start"] + task.superchunk_start_s, "end": w["end"] + task.superchunk_start_s}
                for w in local_words
            ]
            result_queue.put(AsrResult(task.task_id, task.video_id, task.superchunk_idx, global_words, None))
        except Exception as exc:  # noqa: BLE001 - reported to the pool, never fatal to the worker loop
            result_queue.put(AsrResult(task.task_id, task.video_id, task.superchunk_idx, None, str(exc)))
        heartbeat_queue.put((worker_id, task.task_id, "end", time.monotonic()))


class AsrWorkerPool:
    def __init__(
        self,
        cfg: AsrConfig,
        worker_specs: list[dict],
        *,
        model_loader=_default_model_loader,
        transcribe_fn=_default_transcribe,
        stall_timeout_s: float = 300.0,
        heartbeat_poll_s: float = 1.0,
        mp_context: str = "spawn",
    ) -> None:
        self.cfg = cfg
        self.worker_specs = {spec["worker_id"]: spec for spec in worker_specs}
        self.model_loader = model_loader
        self.transcribe_fn = transcribe_fn
        self.stall_timeout_s = stall_timeout_s
        self.heartbeat_poll_s = heartbeat_poll_s

        self._ctx = mp.get_context(mp_context)
        self._task_queue = self._ctx.Queue()
        self._result_queue = self._ctx.Queue()
        self._heartbeat_queue = self._ctx.Queue()

        self._processes: dict[str, object] = {}
        self._last_heartbeat: dict[str, float] = {}
        self._in_flight: dict[str, str | None] = {}  # worker_id -> task_id currently being processed

        self._pending_futures: dict[str, asyncio.Future] = {}
        self._pending_tasks: dict[str, AsrTask] = {}

        self._closed = False
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader_thread: threading.Thread | None = None
        self._heartbeat_thread: threading.Thread | None = None
        self._supervisor_task: asyncio.Task | None = None

    # ---- lifecycle ----

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        for worker_id in self.worker_specs:
            self._spawn(worker_id)

        self._reader_thread = threading.Thread(target=self._read_results_loop, daemon=True)
        self._reader_thread.start()
        self._heartbeat_thread = threading.Thread(target=self._read_heartbeats_loop, daemon=True)
        self._heartbeat_thread.start()
        self._supervisor_task = asyncio.create_task(self._supervise_loop())

    async def stop(self) -> None:
        self._closed = True
        if self._supervisor_task is not None:
            self._supervisor_task.cancel()
        for _ in self._processes:
            self._task_queue.put(None)
        for worker_id, proc in list(self._processes.items()):
            proc.join(timeout=10)
            if proc.is_alive():
                proc.terminate()
        for fut in self._pending_futures.values():
            if not fut.done():
                fut.set_exception(RuntimeError("worker pool stopped before task completed"))

    def _spawn(self, worker_id: str) -> None:
        spec = self.worker_specs[worker_id]
        proc = self._ctx.Process(
            target=_worker_main,
            args=(
                self._task_queue,
                self._result_queue,
                self._heartbeat_queue,
                worker_id,
                spec["device"],
                spec["gpu_id"],
                self.model_loader,
                self.transcribe_fn,
                self.cfg,
            ),
            daemon=True,
        )
        proc.start()
        self._processes[worker_id] = proc
        self._last_heartbeat[worker_id] = time.monotonic()
        self._in_flight[worker_id] = None

    # ---- submission ----

    async def submit(self, task: AsrTask) -> AsrResult:
        fut = self._loop.create_future()
        self._pending_futures[task.task_id] = fut
        self._pending_tasks[task.task_id] = task
        self._task_queue.put(task)
        return await fut

    # ---- background threads ----

    def _read_results_loop(self) -> None:
        while not self._closed:
            try:
                item: AsrResult = self._result_queue.get(timeout=0.5)
            except _queue_module.Empty:
                continue
            self._pending_tasks.pop(item.task_id, None)
            fut = self._pending_futures.pop(item.task_id, None)
            if fut is not None and self._loop is not None:
                self._loop.call_soon_threadsafe(_resolve_future, fut, item)

    def _read_heartbeats_loop(self) -> None:
        while not self._closed:
            try:
                worker_id, task_id, phase, ts = self._heartbeat_queue.get(timeout=0.5)
            except _queue_module.Empty:
                continue
            self._last_heartbeat[worker_id] = ts
            self._in_flight[worker_id] = task_id if phase == "start" else None

    # ---- supervision ----

    async def _supervise_loop(self) -> None:
        try:
            while not self._closed:
                await asyncio.sleep(self.heartbeat_poll_s)
                for worker_id in list(self.worker_specs):
                    self._check_worker(worker_id)
        except asyncio.CancelledError:
            pass

    def _check_worker(self, worker_id: str) -> None:
        proc = self._processes.get(worker_id)
        if proc is None:
            return
        crashed = not proc.is_alive()
        stalled = (
            not crashed
            and self._in_flight.get(worker_id) is not None
            and (time.monotonic() - self._last_heartbeat[worker_id]) > self.stall_timeout_s
        )
        if not crashed and not stalled:
            return

        reason = "crashed" if crashed else "stalled"
        logger.warning("ASR worker %s %s; restarting", worker_id, reason)

        task_id = self._in_flight.get(worker_id)
        if not crashed:
            proc.terminate()
        proc.join(timeout=1)
        self._processes.pop(worker_id, None)

        if task_id is not None:
            task = self._pending_tasks.get(task_id)
            if task is not None:
                self._task_queue.put(task)  # requeue for another worker to pick up

        self._spawn(worker_id)


def _resolve_future(fut: asyncio.Future, result: AsrResult) -> None:
    if not fut.done():
        fut.set_result(result)


def make_task(video_id: str, superchunk_idx: int, audio_path: str, superchunk_start_s: float) -> AsrTask:
    return AsrTask(
        task_id=str(uuid.uuid4()),
        video_id=video_id,
        superchunk_idx=superchunk_idx,
        audio_path=audio_path,
        superchunk_start_s=superchunk_start_s,
    )
