"""
Metric Computations — Pure Core
================================

Every function in this module is a pure function: given the same inputs
it returns the same output, with no side effects. This is the testable
core of SpikeProf, verifiable via property-based testing without GPU
hardware.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Literal, Sequence, Any

from models import GpuInfo, SpikeProfileResult
from typedecs import (
    Bytes,
    Flops,
    FractionZeroOne,
    GigabytesPerSec,
    Milliseconds,
    Percentage,
    RooflineClass,
    StallReason,
    TeraflopsPerSec,
    Watts,
)

# ── Timing-Derived Metrics ──────────────────────────────────────────


def compute_bandwidth(total_bytes: Bytes, time_ms: Milliseconds) -> GigabytesPerSec:
    """Compute analytical bandwidth in GB/s from bytes and time.

    Pure function. Returns 0.0 if time_ms <= 0.
    """
    if time_ms <= 0.0:
        return GigabytesPerSec(0.0)
    return GigabytesPerSec(total_bytes / (time_ms * 1e-3) / 1e9)


def compute_gflops(total_flops: Flops, time_ms: Milliseconds) -> float:
    """Compute achieved GFLOP/s from flops and time.

    Pure function. Returns 0.0 if time_ms <= 0.
    """
    if time_ms <= 0.0:
        return 0.0
    return total_flops / (time_ms * 1e-3) / 1e9


def compute_arithmetic_intensity(total_flops: Flops, total_bytes: Bytes) -> float:
    """Compute arithmetic intensity (FLOP/byte).

    Pure function. Returns 0.0 if total_bytes == 0.
    """
    if total_bytes == 0:
        return 0.0
    return total_flops / total_bytes


def compute_bw_utilization(
    analytical_bw: GigabytesPerSec, peak_bw: GigabytesPerSec
) -> Percentage:
    """Compute bandwidth utilization as a percentage.

    Pure function. Returns 0.0 if peak_bw <= 0.
    """
    if peak_bw <= 0.0:
        return Percentage(0.0)
    return Percentage(analytical_bw / peak_bw * 100.0)


# ── Roofline Classification ────────────────────────────────────────


def compute_machine_balance(
    peak_tflops: TeraflopsPerSec, peak_bw_gbs: GigabytesPerSec
) -> float:
    """Compute machine balance point (FLOP/byte).

    The ridge point of the roofline model.
    Pure function.
    """
    if peak_bw_gbs <= 0.0:
        return float("inf")
    # Convert TFLOP/s to GFLOP/s for consistency: TFLOP/s * 1000
    return (peak_tflops * 1000.0) / peak_bw_gbs


def roofline_classify(
    arithmetic_intensity: float, machine_balance: float
) -> RooflineClass:
    """Classify a kernel as memory-bound or compute-bound.

    Pure function. Exhaustive on boundary (boundary → MEMORY_BOUND).
    """
    if arithmetic_intensity < machine_balance:
        return "MEMORY_BOUND"
    return "COMPUTE_BOUND"


def validate_roofline(
    roofline_class: RooflineClass | Literal[""],
    dram_total_gbs: GigabytesPerSec,
    peak_bw_gbs: GigabytesPerSec,
    sm_occupancy_pct: Percentage,
    warp_stall_memory_pct: Percentage,
    warp_stall_exec_pct: Percentage,
) -> tuple[bool, str]:
    """Validate roofline classification against CUPTI measurements.

    Returns (validated: bool, notes: str).
    Pure function.
    """
    if peak_bw_gbs <= 0.0:
        return False, "Cannot validate: peak bandwidth unknown."

    if roofline_class == "":
        return False, "Cannot validate: roofline class not set."

    bw_ratio = dram_total_gbs / peak_bw_gbs

    if roofline_class == "MEMORY_BOUND":
        if bw_ratio > 0.6:
            return True, "VALIDATED: High DRAM throughput confirms memory-bound."
        elif bw_ratio < 0.3:
            return (
                False,
                "LATENCY_BOUND: Low DRAM throughput despite memory-bound AI. "
                "Possible occupancy or cache issue.",
            )
        else:
            return (
                False,
                f"INCONCLUSIVE: DRAM utilization at {bw_ratio:.0%} — "
                "between latency-bound and bandwidth-saturated.",
            )
    else:  # COMPUTE_BOUND
        if sm_occupancy_pct > 50.0 and warp_stall_exec_pct > warp_stall_memory_pct:
            return True, "VALIDATED: High occupancy and execution stalls confirm compute-bound."
        elif warp_stall_memory_pct > warp_stall_exec_pct:
            return (
                False,
                "MISCLASSIFIED: Memory stalls dominate despite high AI. "
                "Actually memory-bound.",
            )
        else:
            return False, "INCONCLUSIVE: Cannot confirm compute-bound classification."


# ── Grid and Wave Metrics ───────────────────────────────────────────


def compute_grid_size(batch_size: int, num_heads: int) -> int:
    """Compute grid size for kernel launch. Pure function."""
    return batch_size * max(1, num_heads)


def compute_wave_count(grid_size: int, num_sm: int) -> float:
    """Compute wave count. Pure function."""
    if num_sm <= 0:
        return 0.0
    return grid_size / num_sm


def compute_tiles_per_seq(seq_len: int, block_l: int) -> int:
    """Compute tiles per sequence. Pure function."""
    if block_l <= 0:
        return 0
    return math.ceil(seq_len / block_l)


# ── Warp Stall Analysis ────────────────────────────────────────────


def determine_dominant_stall(
    memory_pct: Percentage,
    exec_pct: Percentage,
    sync_pct: Percentage,
    other_pct: Percentage,
) -> StallReason:
    """Determine the dominant warp stall reason. Pure function."""
    stalls: dict[str, float] = {
        "memory": float(memory_pct),
        "execution": float(exec_pct),
        "synchronization": float(sync_pct),
        "other": float(other_pct),
    }
    dominant: StallReason = max(stalls, key=stalls.get)  # type: ignore[arg-type]
    return dominant


# ── SNN-Specific Metrics ───────────────────────────────────────────


def compute_sparsity_exploitation(
    time_at_sparsity: Milliseconds,
    time_at_dense: Milliseconds,
) -> FractionZeroOne:
    """Compute sparsity exploitation ratio.

    exploitation = 1 - (time_sparse / time_dense)
    A kernel fully exploiting sparsity: exploitation ≈ sparsity.
    A kernel ignoring sparsity: exploitation ≈ 0.

    Pure function. Result clamped to [0, 1].
    """
    if time_at_dense <= 0.0:
        return FractionZeroOne(0.0)
    ratio = 1.0 - (time_at_sparsity / time_at_dense)
    return FractionZeroOne(max(0.0, min(1.0, ratio)))


def compute_sparsity_exploitation_approx(
    effective_flops: Flops, dense_flops: Flops
) -> FractionZeroOne:
    """Approximate sparsity exploitation from analytical models.

    exploitation_approx = 1 - (effective_flops / dense_flops)
    Pure function.
    """
    if dense_flops <= 0:
        return FractionZeroOne(0.0)
    ratio = 1.0 - (effective_flops / dense_flops)
    return FractionZeroOne(max(0.0, min(1.0, ratio)))


def compute_vmem_pressure(
    batch_size: int,
    num_neurons: int,
    elem_size: int,
    time_ms: Milliseconds,
) -> GigabytesPerSec:
    """Compute membrane-potential memory pressure in GB/s.

    vmem_pressure = (read_U + write_U) / time
    Pure function.
    """
    if time_ms <= 0.0:
        return GigabytesPerSec(0.0)
    vmem_bytes = 2 * batch_size * num_neurons * elem_size  # read + write
    return GigabytesPerSec(vmem_bytes / (time_ms * 1e-3) / 1e9)


def compute_vmem_traffic_fraction(
    vmem_pressure_gbs: GigabytesPerSec,
    dram_total_gbs: GigabytesPerSec,
) -> FractionZeroOne:
    """Fraction of total DRAM traffic that is membrane-potential traffic.

    Pure function. Returns 0.0 if dram_total_gbs <= 0.
    """
    if dram_total_gbs <= 0.0:
        return FractionZeroOne(0.0)
    frac = vmem_pressure_gbs / dram_total_gbs
    return FractionZeroOne(min(1.0, max(0.0, frac)))


def compute_neuronal_compute_frac(
    results: Sequence[SpikeProfileResult],
) -> FractionZeroOne:
    """Compute fraction of total time spent in NEURON-category kernels.

    Pure function on a sequence of immutable results.
    """
    total_time = sum(r.time_ms for r in results)
    if total_time <= 0.0:
        return FractionZeroOne(0.0)
    neuron_time = sum(r.time_ms for r in results if r.kernel_category == "NEURON")
    return FractionZeroOne(neuron_time / total_time)


def compute_edp(
    power_watts: Watts, time_ms: Milliseconds
) -> float:
    """Compute Energy-Delay Product in µJ·ms.

    edp = power × time² (reported in µJ·ms for readability).
    Pure function. Commutative in proportion: edp(2p, t) == 2 * edp(p, t).
    """
    if time_ms <= 0.0 or power_watts <= 0.0:
        return 0.0
    time_s = time_ms * 1e-3
    energy_j = power_watts * time_s
    edp_j_s = energy_j * time_s
    return edp_j_s * 1e6 * 1e3  # → µJ·ms


def estimate_power(
    sm_occupancy_pct: Percentage, tdp_watts: Watts
) -> Watts:
    """Estimate power from occupancy and TDP (last-resort fallback).

    Pure function.
    """
    utilization = max(0.1, sm_occupancy_pct / 100.0)  # Floor at 10%
    return Watts(tdp_watts * utilization)


def compute_per_timestep_cost(
    total_time_ms: Milliseconds, time_steps: int
) -> Milliseconds:
    """Compute per-timestep cost. Pure function."""
    if time_steps <= 0:
        return Milliseconds(0.0)
    return Milliseconds(total_time_ms / time_steps)


def compute_timestep_scaling_factor(
    time_at_t_max: Milliseconds,
    time_at_t_1: Milliseconds,
    t_max: int,
) -> float:
    """Compute timestep scaling factor.

    scaling = measured(T_max) / (T_max × measured(T=1))
    1.0 = perfect linear. >1.0 = super-linear. <1.0 = sub-linear.
    Pure function.
    """
    if t_max <= 0 or time_at_t_1 <= 0.0:
        return 0.0
    return time_at_t_max / (t_max * time_at_t_1)


def compute_bw_divergence(
    analytical_bw: GigabytesPerSec,
    measured_bw: GigabytesPerSec,
) -> Percentage:
    """Compute percentage divergence between analytical and measured BW.

    Pure function.
    """
    if analytical_bw <= 0.0:
        return Percentage(0.0)
    return Percentage(abs(analytical_bw - measured_bw) / analytical_bw * 100.0)


# ── Result Derivation (immutable transforms) ───────────────────────


def with_timing_metrics(
    result: SpikeProfileResult,
    *,
    time_ms: Milliseconds,
    total_bytes: Bytes,
    total_flops: Flops,
    gpu_info: GpuInfo,
) -> SpikeProfileResult:
    """Derive a new result with timing metrics filled in. Pure function."""
    bw = compute_bandwidth(total_bytes, time_ms)
    gflops = compute_gflops(total_flops, time_ms)
    ai = compute_arithmetic_intensity(total_flops, total_bytes)
    bw_util = compute_bw_utilization(bw, gpu_info.peak_bw_gbs)
    balance = compute_machine_balance(gpu_info.peak_fp32_tflops, gpu_info.peak_bw_gbs)
    rc = roofline_classify(ai, balance)

    return dataclasses.replace(
        result,
        time_ms=time_ms,
        total_bytes=total_bytes,
        total_flops=total_flops,
        analytical_bw_gbs=bw,
        bw_util_pct=bw_util,
        analytical_gflops=gflops,
        arithmetic_intensity=ai,
        roofline_class=rc,
        gpu_name=gpu_info.name,
        num_sm=gpu_info.num_sm,
        peak_bw_gbs=gpu_info.peak_bw_gbs,
        peak_fp32_tflops=gpu_info.peak_fp32_tflops,
    )


def with_cupti_metrics(
    result: SpikeProfileResult,
    *,
    counters: dict[str, float],
) -> SpikeProfileResult:
    """Derive a new result with CUPTI counter fields filled in. Pure function."""
    dram_read = GigabytesPerSec(counters.get("dram_read_gbs", 0.0))
    dram_write = GigabytesPerSec(counters.get("dram_write_gbs", 0.0))
    dram_total = GigabytesPerSec(dram_read + dram_write)
    l2_hit = FractionZeroOne(counters.get("l2_hit_rate", 0.0))
    occupancy = Percentage(counters.get("sm_occupancy_pct", 0.0))
    stall_mem = Percentage(counters.get("warp_stall_memory_pct", 0.0))
    stall_exec = Percentage(counters.get("warp_stall_exec_pct", 0.0))
    stall_sync = Percentage(counters.get("warp_stall_sync_pct", 0.0))
    stall_other = Percentage(counters.get("warp_stall_other_pct", 0.0))
    dominant = determine_dominant_stall(stall_mem, stall_exec, stall_sync, stall_other)
    inst_total = int(counters.get("inst_executed_total", 0))
    fp32_pct = Percentage(counters.get("inst_fp32_pct", 0.0))
    int32_pct = Percentage(counters.get("inst_int32_pct", 0.0))
    fp16_pct = Percentage(counters.get("inst_fp16_pct", 0.0))
    tc_util = Percentage(counters.get("tensor_core_util_pct", 0.0))

    bw_div = compute_bw_divergence(result.analytical_bw_gbs, dram_total)

    # Validate roofline if we have a classification
    validated = False
    if result.roofline_class in ("MEMORY_BOUND", "COMPUTE_BOUND"):
        validated, _ = validate_roofline(
            result.roofline_class,
            dram_total,
            result.peak_bw_gbs,
            occupancy,
            stall_mem,
            stall_exec,
        )

    return dataclasses.replace(
        result,
        cupti_collected=True,
        dram_read_gbs=dram_read,
        dram_write_gbs=dram_write,
        dram_total_gbs=dram_total,
        l2_hit_rate=l2_hit,
        sm_occupancy_pct=occupancy,
        warp_stall_memory_pct=stall_mem,
        warp_stall_exec_pct=stall_exec,
        warp_stall_sync_pct=stall_sync,
        warp_stall_other_pct=stall_other,
        warp_stall_dominant=dominant,
        inst_executed_total=inst_total,
        inst_fp32_pct=fp32_pct,
        inst_int32_pct=int32_pct,
        inst_fp16_pct=fp16_pct,
        tensor_core_util_pct=tc_util,
        bw_analytical_vs_measured_pct=bw_div,
        roofline_validated=validated,
    )


def with_snn_metrics(
    result: SpikeProfileResult,
    *,
    vmem_pressure: GigabytesPerSec = GigabytesPerSec(0.0),
    sparsity_exploitation: FractionZeroOne = FractionZeroOne(0.0),
    edp: float = 0.0,
    per_timestep_cost: Milliseconds = Milliseconds(0.0),
    timestep_scaling: float = 0.0,
    neuronal_frac: FractionZeroOne = FractionZeroOne(0.0),
) -> SpikeProfileResult:
    """Derive a new result with SNN-specific metrics filled in. Pure function."""
    return dataclasses.replace(
        result,
        vmem_pressure_gbs=vmem_pressure,
        sparsity_exploitation=sparsity_exploitation,
        edp_estimate_uj_ms=edp,
        per_timestep_cost_ms=per_timestep_cost,
        timestep_scaling_factor=timestep_scaling,
        neuronal_compute_frac=neuronal_frac,
    )


# ── Functional Composition ─────────────────────────────────────────


def compose_results(
    r1: SpikeProfileResult, r2: SpikeProfileResult
) -> SpikeProfileResult:
    """Merge two results for the same kernel config, preferring non-default values.

    Used by ``compose_profiles`` to combine timing-only and CUPTI-deep results.
    Pure function.
    """
    d1 = dataclasses.asdict(r1)
    d2 = dataclasses.asdict(r2)

    # Get defaults for comparison
    defaults = {f.name: f.default for f in dataclasses.fields(SpikeProfileResult)}

    merged: dict[str, Any] = {}
    for key in d1:
        v1 = d1[key]
        v2 = d2[key]
        default = defaults.get(key)
        # Prefer non-default values; if both non-default, prefer r2
        if v2 != default:
            merged[key] = v2
        else:
            merged[key] = v1

    return SpikeProfileResult.from_dict(merged)
