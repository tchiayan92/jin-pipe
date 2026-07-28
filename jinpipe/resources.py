"""Memory/OOM admission control: budget reservation + OS-level safety brake.

Two independent checks must both pass before a task is admitted into a stage:

1. Reservation accounting - the sum of in-flight reserved bytes for a pool (CPU
   RAM, or a specific GPU's VRAM) must stay under that pool's configured budget.
   This catches bursts of similarly-sized tasks whose *eventual* memory use would
   exceed capacity even before the OS reports any pressure yet.
2. OS-reported available memory must stay above a hard floor. This catches
   estimation error in the per-stage cost model and memory pressure coming from
   outside the pipeline (other processes on the box).

All acquire/release calls are expected to happen on a single asyncio event loop
thread (the orchestrator's main loop, per JinPipe's single-writer/single-scheduler
design), so the reservation counters need no locking.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Callable

import psutil

from jinpipe.config import ResourceConfig


def _default_vram_query(gpu_id: int) -> tuple[int, int]:
    """Return (free_bytes, total_bytes) for a GPU, or (0, 0) if none is available."""
    try:
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free, info.total
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.mem_get_info(gpu_id)
    except Exception:
        pass
    return 0, 0


class ResourceGate:
    def __init__(
        self,
        cfg: ResourceConfig,
        gpu_ids: list[int] | None = None,
        ram_available_fn: Callable[[], int] | None = None,
        vram_query_fn: Callable[[int], tuple[int, int]] | None = None,
    ) -> None:
        self._cfg = cfg
        self._cost_per_second = cfg.cost_per_audio_second
        self._poll_interval = cfg.poll_interval_s
        self._ram_floor = cfg.ram_floor_mb * 1024 * 1024
        self._vram_floor = cfg.vram_floor_mb * 1024 * 1024
        self._ram_available_fn = ram_available_fn or (lambda: psutil.virtual_memory().available)
        self._vram_query_fn = vram_query_fn or _default_vram_query
        self._gpu_ids = list(gpu_ids or [])

        self._ram_budget = (
            cfg.ram_budget_mb * 1024 * 1024
            if cfg.ram_budget_mb is not None
            else max(psutil.virtual_memory().total - self._ram_floor, 0)
        )
        self._ram_reserved = 0

        self._vram_budget: dict[int, int] = {}
        self._vram_reserved: dict[int, int] = {}
        for gpu_id in self._gpu_ids:
            _, total = self._vram_query_fn(gpu_id)
            budget = (
                cfg.vram_budget_mb * 1024 * 1024
                if cfg.vram_budget_mb is not None
                else max(total - self._vram_floor, 0)
            )
            self._vram_budget[gpu_id] = budget
            self._vram_reserved[gpu_id] = 0

    def estimate_cost(self, stage: str, duration_s: float) -> int:
        return int(self._cost_per_second.get(stage, 0.0) * max(duration_s, 0.0))

    async def acquire(self, stage: str, duration_s: float, gpu_id: int | None = None) -> "Reservation":
        cost = self.estimate_cost(stage, duration_s)
        while not self._try_reserve(cost, gpu_id):
            await asyncio.sleep(self._poll_interval)
        return Reservation(gate=self, stage=stage, cost=cost, gpu_id=gpu_id)

    def _try_reserve(self, cost: int, gpu_id: int | None) -> bool:
        if gpu_id is not None:
            if gpu_id not in self._vram_budget:
                raise ValueError(f"unknown gpu_id {gpu_id!r}; not in configured gpu_ids")
            budget = self._vram_budget[gpu_id]
            reserved = self._vram_reserved[gpu_id]
            available, _ = self._vram_query_fn(gpu_id)
            floor = self._vram_floor
        else:
            budget = self._ram_budget
            reserved = self._ram_reserved
            available = self._ram_available_fn()
            floor = self._ram_floor

        if reserved + cost > budget:
            return False
        if available - cost < floor:
            return False

        if gpu_id is not None:
            self._vram_reserved[gpu_id] += cost
        else:
            self._ram_reserved += cost
        return True

    def _release(self, cost: int, gpu_id: int | None) -> None:
        if gpu_id is not None:
            self._vram_reserved[gpu_id] = max(0, self._vram_reserved[gpu_id] - cost)
        else:
            self._ram_reserved = max(0, self._ram_reserved - cost)

    def in_use(self) -> dict:
        return {
            "ram_reserved": self._ram_reserved,
            "ram_budget": self._ram_budget,
            "vram_reserved": dict(self._vram_reserved),
            "vram_budget": dict(self._vram_budget),
        }


@dataclass
class Reservation:
    gate: ResourceGate
    stage: str
    cost: int
    gpu_id: int | None = None
    _released: bool = field(default=False, repr=False)

    async def __aenter__(self) -> "Reservation":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        self.release()

    def release(self) -> None:
        if not self._released:
            self.gate._release(self.cost, self.gpu_id)
            self._released = True
