"""
Timing Core
===========

Measures wall-clock kernel latency with controlled warmup and repetition
via ``triton.testing.do_bench``. Computes analytical timing metrics.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable

from .models import Err, GpuInfo, Ok, ProfileError, Result, SpikeProfileResult
from .metrics import with_timing_metrics
from .typedecs import (
    Bytes,
    CuptiTier,
    Flops,
    FractionZeroOne,
    KernelSpec,
    Milliseconds,
    TimingBackend,
)

logger = logging.getLogger(__name__)

# ── Production Timing Backend ───────────────────────────────────────


@dataclass(frozen=True, slots=True)
class TritonTimingBackend:
    """Production timing backend using triton.testing.do_bench."""

    warmup: int = 50
    rep: int = 200

    def measure(self, kernel_fn: Callable[..., None], reps: int) -> Milliseconds:
        """Measure kernel execution time using do_bench."""
        try:
            import triton  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError("Triton is required for timing.") from exc

        time_ms = triton.testing.do_bench(
            kernel_fn,
            warmup=self.warmup,
            rep=reps,
        )
        return Milliseconds(float(time_ms))


# ── Stub Timing Backend (for testing) ──────────────────────────────


@dataclass(frozen=True, slots=True)
class StubTimingBackend:
    """Deterministic timing stub for integration tests."""

    fixed_time_ms: Milliseconds

    def measure(self, kernel_fn: Callable[..., None], reps: int) -> Milliseconds:
        return self.fixed_time_ms


# ── Timing Core Function ───────────────────────────────────────────


def time_kernel(
    kernel_fn: Callable[..., None],
    *,
    timing_backend: TimingBackend,
    bench_reps: int = 200,
) -> Result[Milliseconds]:
    """Time a kernel using the provided backend.

    Returns Ok(Milliseconds) on success, Err(ProfileError) on failure.
    """
    try:
        time_ms = timing_backend.measure(kernel_fn, bench_reps)
        return Ok(time_ms)
    except Exception as exc:
        return Err(
            ProfileError(
                config_label="timing",
                error_type="launch_failure",
                message=str(exc),
            )
        )


def build_timing_result(
    spec: KernelSpec,
    time_ms: Milliseconds,
    gpu_info: GpuInfo,
    workload_params: dict[str, Any],
    kernel_params: dict[str, int],
) -> SpikeProfileResult:
    """Build a SpikeProfileResult with timing metrics populated.

    Pure function: constructs immutable result from inputs.
    """
    # Compute byte and flop models
    total_bytes = spec.byte_model(**workload_params)
    total_flops = spec.flop_model(**workload_params)

    # Compute effective flops (accounting for sparsity)
    sparsity = workload_params.get("sparsity", 0.5)
    if spec.supports_sparsity and sparsity > 0:
        effective_flops = Flops(int(total_flops * (1.0 - sparsity)))
    else:
        effective_flops = total_flops

    # Build base result
    base = SpikeProfileResult(
        kernel_name=spec.name,
        kernel_category=spec.category,
        block_l=kernel_params.get("block_l", 0),
        block_n=kernel_params.get("block_n", 0),
        num_warps=kernel_params.get("num_warps", 0),
        num_stages=kernel_params.get("num_stages", 0),
        batch_size=workload_params.get("batch_size", 0),
        num_heads=workload_params.get("num_heads", 0),
        seq_len=workload_params.get("seq_len", 0),
        dim=workload_params.get("dim", 0),
        time_steps=workload_params.get("time_steps", 1),
        sparsity=FractionZeroOne(sparsity),
        tau=FractionZeroOne(workload_params.get("tau", 0.25)),
        v_th=workload_params.get("v_th", 0.5),
        effective_flops=effective_flops,
        bench_reps=200,
    )

    return with_timing_metrics(
        base,
        time_ms=time_ms,
        total_bytes=total_bytes,
        total_flops=total_flops,
        gpu_info=gpu_info,
    )
