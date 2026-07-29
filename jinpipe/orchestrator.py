"""Top-level asyncio orchestrator wiring every stage together.

The single asyncio event loop in this module's coroutines is the only
scheduler *and* the only SQLite writer (per db.py's single-writer
discipline): worker processes (ProcessPoolExecutor stages and the persistent
ASR pool) only ever return plain data, and this module is the only thing
that calls into JobStore to persist results.

Backpressure composes from the primitives each stage already provides rather
than adding another manual queue on top of them: download_all()'s internal
bounded queue throttles downloads, the ProcessPoolExecutor's own work queue
throttles CPU stages (standardize/VAD/filter/packaging), and AsrWorkerPool's
fixed persistent-worker-per-GPU design bounds actual concurrent model
inference - ResourceGate additionally bounds how many super-chunks/segments
can be in flight (sliced to disk, awaiting a worker) at once.

Rechunking is the one deliberate synchronization point: it only runs for a
video once every one of that video's super-chunks has finished ASR.
"""

from __future__ import annotations

import asyncio
import json
import logging
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from pathlib import Path

from jinpipe.config import JinPipeConfig
from jinpipe.db import JobStore
from jinpipe.resources import ResourceGate
from jinpipe.stages import filter as filter_stage
from jinpipe.stages import package as package_stage
from jinpipe.stages import standardize as standardize_stage
from jinpipe.stages import vad as vad_stage
from jinpipe.stages.discover import DiscoverError, DiscoveredVideo, discover_sources
from jinpipe.stages.download import DownloadError, download_all
from jinpipe.stages.rechunk import Segment, SuperchunkWords, Word, rechunk_video
from jinpipe.workers.asr_worker import AsrWorkerPool, build_worker_specs, make_task

logger = logging.getLogger(__name__)


def detect_gpu_ids() -> list[int]:
    try:
        import torch

        if torch.cuda.is_available():
            return list(range(torch.cuda.device_count()))
    except Exception:  # noqa: BLE001 - torch may not be installed at all
        pass
    return []


async def _pending_and_fresh_videos(store: JobStore, cfg: JinPipeConfig):
    """Yield videos still needing work: leftover PENDING rows from a prior
    interrupted run first, then freshly discovered ones (registered as
    PENDING; anything already known - DONE, RUNNING, FAILED, or already
    yielded as PENDING above - is skipped so re-running is safe)."""
    for video in store.list_videos(status="PENDING"):
        yield DiscoveredVideo(video_id=video["video_id"], channel=video["channel"], url=video["url"])

    async for item in discover_sources(cfg.sources, cfg.discover):
        if isinstance(item, DiscoverError):
            logger.warning("discover error for %s: %s", item.channel, item.error)
            continue
        if store.get_video(item.video_id) is not None:
            continue
        store.add_video(item.video_id, item.url, channel=item.channel)
        yield item


async def run_pipeline_async(cfg: JinPipeConfig, *, gpu_ids: list[int] | None = None) -> None:
    cfg.paths.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = cfg.paths.work_dir / "raw"
    std_dir = cfg.paths.work_dir / "standardized"

    gpu_ids = gpu_ids if gpu_ids is not None else detect_gpu_ids()
    store = JobStore(cfg.paths.db_path)
    gate = ResourceGate(cfg.resources, gpu_ids=gpu_ids)
    worker_specs = build_worker_specs(cfg.asr, gpu_ids)
    asr_pool = AsrWorkerPool(cfg.asr, worker_specs)
    cpu_workers = max(cfg.standardize.workers, cfg.vad.workers, cfg.filter.workers, 1)
    executor = ProcessPoolExecutor(max_workers=cpu_workers)
    loop = asyncio.get_running_loop()

    asr_pool.start()
    background_tasks: set[asyncio.Task] = set()

    def _spawn(coro) -> None:
        task = asyncio.create_task(coro)
        background_tasks.add(task)
        task.add_done_callback(background_tasks.discard)

    try:
        video_stream = _pending_and_fresh_videos(store, cfg)

        async for result in download_all(video_stream, raw_dir, cfg.download):
            if isinstance(result, DownloadError):
                store.update_video(result.video_id, status="FAILED", error=result.error)
                continue
            store.update_video(
                result.video_id, status="RUNNING", raw_path=str(result.raw_path), duration_s=result.duration_s
            )
            _spawn(
                _process_video_to_superchunks(
                    result.video_id,
                    result.raw_path,
                    result.duration_s,
                    cfg,
                    store,
                    executor,
                    loop,
                    std_dir,
                    gate,
                    asr_pool,
                )
            )

        if background_tasks:
            await asyncio.gather(*background_tasks, return_exceptions=True)

        package_stage.write_manifest(cfg.paths.output_dir, cfg.paths.output_dir / "manifest.jsonl")
    finally:
        await asr_pool.stop()
        executor.shutdown(wait=True)
        store.close()


async def _process_video_to_superchunks(
    video_id: str,
    raw_path: Path,
    duration_s: float | None,
    cfg: JinPipeConfig,
    store: JobStore,
    executor: ProcessPoolExecutor,
    loop: asyncio.AbstractEventLoop,
    std_dir: Path,
    gate: ResourceGate,
    asr_pool: AsrWorkerPool,
) -> None:
    std_path = std_dir / f"{video_id}.wav"
    duration = duration_s or 0.0
    try:
        reservation = await gate.acquire("standardize", duration)
        try:
            await loop.run_in_executor(
                executor,
                partial(standardize_stage.standardize_audio, raw_path, std_path, cfg.standardize.sample_rate),
            )
        finally:
            reservation.release()

        reservation = await gate.acquire("vad", duration)
        try:
            superchunks = await loop.run_in_executor(
                executor, partial(vad_stage.coarse_segment, std_path, cfg.vad, duration)
            )
        finally:
            reservation.release()

        for idx, start, end in superchunks:
            store.add_superchunk(video_id, idx, start, end)
        store.update_video(video_id, status="DONE", standardized_path=str(std_path))
    except Exception as exc:  # noqa: BLE001 - reported to the job store, never fatal to the run
        logger.exception("video processing failed for %s", video_id)
        store.update_video(video_id, status="FAILED", error=str(exc))
        return

    asr_tasks = [
        asyncio.create_task(
            _process_superchunk_asr(video_id, idx, start, end, std_path, cfg, store, gate, asr_pool, executor, loop)
        )
        for idx, start, end in superchunks
    ]
    if asr_tasks:
        await asyncio.gather(*asr_tasks, return_exceptions=True)


async def _process_superchunk_asr(
    video_id: str,
    idx: int,
    start: float,
    end: float,
    std_path: Path,
    cfg: JinPipeConfig,
    store: JobStore,
    gate: ResourceGate,
    asr_pool: AsrWorkerPool,
    executor: ProcessPoolExecutor,
    loop: asyncio.AbstractEventLoop,
) -> None:
    duration = end - start
    stage_name = "diarize" if cfg.asr.diarize else "asr"
    store.update_superchunk(video_id, idx, status="RUNNING")
    chunk_path = std_path.parent / f"{video_id}_{idx:05d}.superchunk.wav"

    reservation = await gate.acquire(stage_name, duration)
    try:
        try:
            await loop.run_in_executor(
                executor, partial(package_stage.slice_segment_audio, std_path, chunk_path, start, end, "wav")
            )
            task = make_task(video_id, idx, str(chunk_path), start)
            result = await asr_pool.submit(task)
        except Exception as exc:  # noqa: BLE001 - reported to the job store, never fatal to the run
            store.update_superchunk(video_id, idx, status="FAILED", error=str(exc))
            return
    finally:
        reservation.release()
        chunk_path.unlink(missing_ok=True)

    if result.error is not None:
        store.update_superchunk(video_id, idx, status="FAILED", error=result.error)
        return
    store.update_superchunk(video_id, idx, status="DONE", words_json=json.dumps(result.words))

    if store.superchunks_all_done(video_id):
        await _rechunk_and_finalize(video_id, cfg, store, executor, loop, gate)


async def _rechunk_and_finalize(
    video_id: str,
    cfg: JinPipeConfig,
    store: JobStore,
    executor: ProcessPoolExecutor,
    loop: asyncio.AbstractEventLoop,
    gate: ResourceGate,
) -> None:
    superchunks = [
        SuperchunkWords(
            idx=row["idx"],
            start=row["start_s"],
            end=row["end_s"],
            words=[
                Word(
                    text=w["word"],
                    start=w["start"],
                    end=w["end"],
                    speaker=w.get("speaker"),
                    overlap=w.get("overlap", False),
                )
                for w in json.loads(row["words_json"] or "[]")
            ],
        )
        for row in store.get_superchunks(video_id)
    ]
    segments: list[Segment] = rechunk_video(superchunks, cfg.rechunk)
    video = store.get_video(video_id)

    for seg in segments:
        seg_id = package_stage.segment_id_for(video_id, seg.idx)
        store.add_segment(
            video_id,
            seg.idx,
            seg_id,
            seg.start,
            seg.end,
            text=seg.text,
            words_json=json.dumps([w.__dict__ for w in seg.words]),
            speaker=seg.speaker,
            exceeds_max_duration=int(seg.exceeds_max_duration),
        )

    finalize_tasks = [
        asyncio.create_task(_finalize_segment(video_id, seg, video, cfg, store, executor, loop, gate))
        for seg in segments
    ]
    if finalize_tasks:
        await asyncio.gather(*finalize_tasks, return_exceptions=True)


async def _finalize_segment(
    video_id: str,
    seg: Segment,
    video: dict,
    cfg: JinPipeConfig,
    store: JobStore,
    executor: ProcessPoolExecutor,
    loop: asyncio.AbstractEventLoop,
    gate: ResourceGate,
) -> None:
    seg_id = package_stage.segment_id_for(video_id, seg.idx)
    store.update_segment(video_id, seg.idx, status="RUNNING")
    duration = seg.end - seg.start
    std_path = Path(video["standardized_path"])
    audio_path = cfg.paths.output_dir / f"{seg_id}.{cfg.package.audio_format}"

    reservation = await gate.acquire("filter", duration)
    try:
        await loop.run_in_executor(
            executor,
            partial(package_stage.slice_segment_audio, std_path, audio_path, seg.start, seg.end, cfg.package.audio_format),
        )
        # NOTE: language isn't yet threaded from WhisperX's per-super-chunk
        # detection through to here, so filter.allowed_languages is inert
        # until that's wired up.
        filter_result = await loop.run_in_executor(
            executor,
            partial(
                filter_stage.filter_segment, audio_path, duration, cfg.filter, language=None, has_overlap=seg.has_overlap
            ),
        )
    except Exception as exc:  # noqa: BLE001 - reported to the job store, never fatal to the run
        store.update_segment(video_id, seg.idx, status="FAILED", error=str(exc))
        return
    finally:
        reservation.release()

    if not filter_result.passed:
        audio_path.unlink(missing_ok=True)
        store.update_segment(video_id, seg.idx, status="FAILED", error=f"filtered: {filter_result.reason}")
        return

    packaged = await loop.run_in_executor(
        executor,
        partial(
            package_stage.package_segment,
            video_id,
            video["url"],
            seg,
            std_path,
            cfg.paths.output_dir,
            cfg.package,
            dnsmos_ovr=filter_result.dnsmos_ovr,
            language=None,
        ),
    )
    store.update_segment(
        video_id,
        seg.idx,
        status="DONE",
        dnsmos_ovr=filter_result.dnsmos_ovr,
        output_audio_path=str(packaged.audio_path),
        output_json_path=str(packaged.json_path),
    )


def run_pipeline(cfg: JinPipeConfig) -> None:
    asyncio.run(run_pipeline_async(cfg))
