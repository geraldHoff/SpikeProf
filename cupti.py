"""
CUPTI Profiling Core
====================

Collects hardware performance counters via Proton's CUPTI backend.
Manages session lifecycle and maps raw counters to result fields.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from models import Err, Ok, ProfileError, Result, SpikeProfileResult
from metrics import with_cupti_metrics
from typedecs import (
    CuptiBackend,
    CuptiTier,
    Milliseconds,
)

logger = logging.getLogger(__name__)

# ── Metric Set Definitions by Tier ──────────────────────────────────

MINIMAL_METRICS: frozenset[str] = frozenset(
    {"dram__bytes_read.sum", "dram__bytes_write.sum", "sm__cycles_active.avg"}
)

STANDARD_METRICS: frozenset[str] = MINIMAL_METRICS | frozenset(
    {
        "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum",
        "lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum",
        "sm__warps_active.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed.sum",
    }
)

FULL_METRICS: frozenset[str] = STANDARD_METRICS | frozenset(
    {
        "smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active",
        "smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active",
        "smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_active",
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active",
        "smsp__inst_executed_op_fp32.sum",
        "smsp__inst_executed_op_integer.sum",
        "smsp__inst_executed_op_fp16.sum",
    }
)

TIER_METRICS: dict[CuptiTier, frozenset[str]] = {
    "minimal": MINIMAL_METRICS,
    "standard": STANDARD_METRICS,
    "full": FULL_METRICS,
}


# ── Counter-to-Field Mapping ───────────────────────────────────────


def map_counters_to_fields(
    raw: Mapping[str, float], kernel_time_ns: float
) -> dict[str, float]:
    """Map raw CUPTI counter values to SpikeProfileResult field names.

    Pure function.
    """
    if kernel_time_ns <= 0:
        kernel_time_ns = 1.0  # Avoid division by zero

    time_s = kernel_time_ns * 1e-9
    result: dict[str, float] = {}

    # DRAM throughput
    dram_read = raw.get("dram__bytes_read.sum", 0.0)
    dram_write = raw.get("dram__bytes_write.sum", 0.0)
    result["dram_read_gbs"] = dram_read / time_s / 1e9
    result["dram_write_gbs"] = dram_write / time_s / 1e9

    # L2 hit rate
    l2_hit = raw.get("lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum", 0.0)
    l2_miss = raw.get("lts__t_sectors_srcunit_tex_op_read_lookup_miss.sum", 0.0)
    if (l2_hit + l2_miss) > 0:
        result["l2_hit_rate"] = l2_hit / (l2_hit + l2_miss)

    # SM occupancy (direct percentage)
    occ = raw.get("sm__warps_active.avg.pct_of_peak_sustained_active", 0.0)
    result["sm_occupancy_pct"] = occ

    # Warp stalls
    result["warp_stall_memory_pct"] = raw.get(
        "smsp__warps_issue_stalled_long_scoreboard.avg.pct_of_peak_sustained_active",
        0.0,
    )
    result["warp_stall_exec_pct"] = raw.get(
        "smsp__warps_issue_stalled_not_selected.avg.pct_of_peak_sustained_active",
        0.0,
    )
    result["warp_stall_sync_pct"] = raw.get(
        "smsp__warps_issue_stalled_wait.avg.pct_of_peak_sustained_active",
        0.0,
    )
    result["warp_stall_other_pct"] = max(
        0.0,
        100.0
        - result.get("warp_stall_memory_pct", 0.0)
        - result.get("warp_stall_exec_pct", 0.0)
        - result.get("warp_stall_sync_pct", 0.0),
    )

    # Instruction mix
    total_inst = raw.get("smsp__inst_executed.sum", 0.0)
    result["inst_executed_total"] = total_inst
    if total_inst > 0:
        result["inst_fp32_pct"] = (
            raw.get("smsp__inst_executed_op_fp32.sum", 0.0) / total_inst * 100.0
        )
        result["inst_int32_pct"] = (
            raw.get("smsp__inst_executed_op_integer.sum", 0.0) / total_inst * 100.0
        )
        result["inst_fp16_pct"] = (
            raw.get("smsp__inst_executed_op_fp16.sum", 0.0) / total_inst * 100.0
        )

    # Tensor core utilization
    result["tensor_core_util_pct"] = raw.get(
        "sm__pipe_tensor_cycles_active.avg.pct_of_peak_sustained_active", 0.0
    )

    return result


# ── Production CUPTI Backend ────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class ProtonCuptiBackend:
    """Production CUPTI backend using Triton Proton."""

    tier: CuptiTier = "standard"
    reps: int = 10
    session_name: str = "spikeprof"

    def collect(
        self, kernel_fn: Callable[..., None], reps: int
    ) -> Mapping[str, float]:
        """Collect CUPTI counters for a kernel."""
        try:
            import proton  # type: ignore[import-not-found]
        except ImportError as exc:
            raise RuntimeError(
                "Triton Proton is required for CUPTI profiling."
            ) from exc

        metrics_to_collect = TIER_METRICS[self.tier]

        proton.start(
            self.session_name,
            context="cupti",
            metrics=list(metrics_to_collect),
        )

        try:
            # Warmup
            for _ in range(10):
                kernel_fn()

            import torch  # type: ignore[import-not-found]
            torch.cuda.synchronize()

            with proton.scope("collection"):
                for _ in range(reps):
                    kernel_fn()

            torch.cuda.synchronize()
        finally:
            proton.finalize()

        # Extract counters from Proton output
        # (Implementation depends on Proton's output format)
        return {}


# ── Stub CUPTI Backend (for testing) ────────────────────────────────


@dataclass(frozen=True, slots=True)
class StubCuptiBackend:
    """Deterministic CUPTI stub returning known counter values."""

    counters: Mapping[str, float]

    def collect(
        self, kernel_fn: Callable[..., None], reps: int
    ) -> Mapping[str, float]:
        return self.counters


# ── CUPTI Core Function ────────────────────────────────────────────


def collect_counters(
    kernel_fn: Callable[..., None],
    *,
    cupti_backend: CuptiBackend,
    cupti_reps: int = 10,
) -> Result[dict[str, float]]:
    """Collect CUPTI counters using the provided backend.

    Returns Ok(counter_dict) on success, Err on failure.
    Graceful degradation: returns Err with error_type='cupti_error'.
    """
    try:
        raw_counters = cupti_backend.collect(kernel_fn, cupti_reps)
        return Ok(dict(raw_counters))
    except Exception as exc:
        logger.warning("CUPTI collection failed: %s", exc)
        return Err(
            ProfileError(
                config_label="cupti",
                error_type="cupti_error",
                message=str(exc),
            )
        )


def apply_cupti_to_result(
    result: SpikeProfileResult,
    counters: dict[str, float],
    kernel_time_ns: float,
) -> SpikeProfileResult:
    """Apply collected counters to a profile result.

    Pure function: returns a new immutable result with CUPTI fields.
    """
    mapped = map_counters_to_fields(counters, kernel_time_ns)
    return with_cupti_metrics(result, counters=mapped)
