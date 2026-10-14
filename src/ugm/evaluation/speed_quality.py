from __future__ import annotations

import statistics
import time
from collections.abc import Callable
from typing import Any

import torch
from torch import nn


class CountingModel(nn.Module):
    """Transparent model wrapper that counts actual forward invocations."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.forward_calls = 0

    def forward(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        cond: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.forward_calls += 1
        return self.model(x, t, cond=cond)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def expected_nfe(
    *,
    sampler_name: str,
    num_steps: int,
    has_condition: bool,
    guidance_scale: float,
) -> int:
    """Expected network evaluations for one sample trajectory."""

    if num_steps <= 0:
        raise ValueError("num_steps must be positive.")
    if sampler_name not in {"ddpm", "ddim", "euler", "heun"}:
        raise ValueError(f"Unknown sampler: {sampler_name}")

    solver_evaluations = 2 if sampler_name == "heun" else 1
    cfg_branches = (
        2
        if has_condition and guidance_scale not in {0.0, 1.0}
        else 1
    )
    return num_steps * solver_evaluations * cfg_branches


def _summary(values: list[float]) -> dict[str, Any]:
    return {
        "values": values,
        "mean": statistics.fmean(values),
        "std_population": statistics.pstdev(values),
        "min": min(values),
        "max": max(values),
        "median": statistics.median(values),
    }


@torch.inference_mode()
def benchmark_sampler(
    *,
    run_once: Callable[[], torch.Tensor],
    counting_model: CountingModel,
    device: torch.device,
    batch_size: int,
    warmup_runs: int,
    repetitions: int,
    expected_forward_calls: int,
) -> dict[str, Any]:
    """Benchmark sampling after warm-up with synchronized repeated timing."""

    if batch_size <= 0:
        raise ValueError("benchmark batch_size must be positive.")
    if warmup_runs < 0:
        raise ValueError("warmup_runs must be non-negative.")
    if repetitions <= 0:
        raise ValueError("repetitions must be positive.")

    for _ in range(warmup_runs):
        run_once()
        synchronize(device)

    elapsed_values: list[float] = []
    throughput_values: list[float] = []
    forward_calls: list[int] = []
    peak_allocated: list[int] = []
    peak_reserved: list[int] = []
    peak_incremental: list[int] = []

    for _ in range(repetitions):
        synchronize(device)
        before_calls = counting_model.forward_calls
        baseline_allocated = 0
        if device.type == "cuda":
            baseline_allocated = torch.cuda.memory_allocated(device)
            torch.cuda.reset_peak_memory_stats(device)

        start = time.perf_counter()
        result = run_once()
        synchronize(device)
        elapsed = time.perf_counter() - start
        if not torch.isfinite(result).all():
            raise FloatingPointError("Benchmark sampling produced non-finite data.")

        calls = counting_model.forward_calls - before_calls
        elapsed_values.append(elapsed)
        throughput_values.append(batch_size / elapsed)
        forward_calls.append(calls)
        if device.type == "cuda":
            allocated = torch.cuda.max_memory_allocated(device)
            reserved = torch.cuda.max_memory_reserved(device)
            peak_allocated.append(int(allocated))
            peak_reserved.append(int(reserved))
            peak_incremental.append(int(max(0, allocated - baseline_allocated)))

    memory: dict[str, Any] = {
        "available": device.type == "cuda",
        "unit": "bytes",
    }
    if device.type == "cuda":
        memory.update(
            {
                "peak_allocated_per_repeat": peak_allocated,
                "peak_reserved_per_repeat": peak_reserved,
                "incremental_peak_allocated_per_repeat": peak_incremental,
                "peak_allocated_max": max(peak_allocated),
                "peak_reserved_max": max(peak_reserved),
                "incremental_peak_allocated_max": max(peak_incremental),
            }
        )

    return {
        "warmup_runs": warmup_runs,
        "repetitions": repetitions,
        "batch_size": batch_size,
        "cuda_synchronized": device.type == "cuda",
        "elapsed_seconds": _summary(elapsed_values),
        "samples_per_second": _summary(throughput_values),
        "nfe_per_sample_trajectory": expected_forward_calls,
        "observed_model_forward_calls_per_repeat": forward_calls,
        "nfe_verified": all(
            calls == expected_forward_calls for calls in forward_calls
        ),
        "cuda_memory": memory,
    }
