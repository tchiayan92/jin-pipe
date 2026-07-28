import asyncio

import pytest

from jinpipe.config import ResourceConfig
from jinpipe.resources import ResourceGate


def _cfg(ram_budget_mb=100, vram_budget_mb=50, ram_floor_mb=0, vram_floor_mb=0, poll_interval_s=0.01):
    return ResourceConfig(
        ram_floor_mb=ram_floor_mb,
        vram_floor_mb=vram_floor_mb,
        ram_budget_mb=ram_budget_mb,
        vram_budget_mb=vram_budget_mb,
        poll_interval_s=poll_interval_s,
        cost_per_audio_second={"asr": 10.0 * 1024 * 1024, "vad": 1.0 * 1024 * 1024},
    )


def test_estimate_cost():
    gate = ResourceGate(_cfg(), ram_available_fn=lambda: 10**12)
    cost = gate.estimate_cost("asr", 5.0)
    assert cost == int(10 * 1024 * 1024 * 5.0)


async def test_acquire_release_ram_budget_roundtrip():
    # 100MB budget, each 5s "asr" task costs 50MB -> exactly two fit at once.
    gate = ResourceGate(_cfg(ram_budget_mb=100), ram_available_fn=lambda: 10**12)
    r1 = await asyncio.wait_for(gate.acquire("asr", 5.0), timeout=1)
    r2 = await asyncio.wait_for(gate.acquire("asr", 5.0), timeout=1)

    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(gate.acquire("asr", 5.0), timeout=0.1)

    r1.release()
    r3 = await asyncio.wait_for(gate.acquire("asr", 5.0), timeout=1)
    assert gate.in_use()["ram_reserved"] == 100 * 1024 * 1024
    r2.release()
    r3.release()
    assert gate.in_use()["ram_reserved"] == 0


async def test_acquire_blocks_on_os_floor_even_with_budget_room():
    available = {"value": 10 * 1024 * 1024}  # only 10MB actually available in the OS
    gate = ResourceGate(
        _cfg(ram_budget_mb=1000, ram_floor_mb=0),
        ram_available_fn=lambda: available["value"],
    )
    # 50MB task cost exceeds the 10MB actually available -> must block despite huge budget.
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(gate.acquire("asr", 5.0), timeout=0.1)

    available["value"] = 10 * 1024 * 1024 * 1024
    reservation = await asyncio.wait_for(gate.acquire("asr", 5.0), timeout=1)
    reservation.release()


async def test_gpu_pool_independent_of_ram_pool():
    gate = ResourceGate(
        _cfg(),
        gpu_ids=[0, 1],
        ram_available_fn=lambda: 10**12,
        vram_query_fn=lambda gpu_id: (10**12, 10**12),
    )
    r_ram = await asyncio.wait_for(gate.acquire("asr", 1.0), timeout=1)
    r_gpu0 = await asyncio.wait_for(gate.acquire("asr", 1.0, gpu_id=0), timeout=1)
    usage = gate.in_use()
    assert usage["ram_reserved"] > 0
    assert usage["vram_reserved"][0] > 0
    assert usage["vram_reserved"][1] == 0
    r_ram.release()
    r_gpu0.release()


async def test_reservation_as_context_manager_releases_on_exit():
    gate = ResourceGate(_cfg(), ram_available_fn=lambda: 10**12)
    reservation = await gate.acquire("vad", 1.0)
    async with reservation:
        assert gate.in_use()["ram_reserved"] == reservation.cost
    assert gate.in_use()["ram_reserved"] == 0


def test_unknown_gpu_id_raises():
    gate = ResourceGate(_cfg(), gpu_ids=[0], ram_available_fn=lambda: 10**12)
    with pytest.raises(ValueError):
        gate._try_reserve(1, gpu_id=5)
