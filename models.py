"""
SpikeProf Data Model
====================

All domain data types are frozen dataclasses. Mutation is replaced by
derivation: pure functions accept an immutable record and return a new
one via ``dataclasses.replace()``.
"""

from __future__ import annotations

import csv
import dataclasses
import io
import json
from dataclasses import dataclass, field, fields
from typing import (
    Any,
    Generic,
    Literal,
    Mapping,
    TypeVar,
    Union,
)

from .typedecs import (
    Bytes,
    FractionZeroOne,
    Flops,
    GigabytesPerSec,
    KernelCategory,
    Milliseconds,
    Percentage,
    RooflineClass,
    StallReason,
    TeraflopsPerSec,
    Watts,
)


# ── GPU Info ────────────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class GpuInfo:
    """Detected GPU properties — immutable after creation."""

    name: str
    compute_capability: tuple[int, int]
    num_sm: int
    max_threads_per_sm: int
    max_regs_per_sm: int
    warp_size: int
    peak_bw_gbs: GigabytesPerSec
    peak_fp32_tflops: TeraflopsPerSec
    peak_fp16_tflops: TeraflopsPerSec
    peak_int32_tflops: TeraflopsPerSec
    tdp_watts: Watts


# ── Profile Result ──────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class SpikeProfileResult:
    """Central data structure: one instance per (kernel, configuration) pair."""

    # ── Identification ──────────────────────────────────────────
    kernel_name: str = ""
    kernel_category: KernelCategory = "CUSTOM"
    proton_scope: str = ""

    # ── Kernel Tuning Parameters ────────────────────────────────
    block_l: int = 0
    block_n: int = 0
    num_warps: int = 0
    num_stages: int = 0

    # ── SNN Workload Parameters ─────────────────────────────────
    batch_size: int = 0
    num_heads: int = 0
    seq_len: int = 0
    dim: int = 0
    channels_per_head: int = 0
    time_steps: int = 1
    time_step_index: int = -1
    layer_index: int = -1
    sparsity: FractionZeroOne = FractionZeroOne(0.5)
    tau: FractionZeroOne = FractionZeroOne(0.25)
    v_th: float = 0.5
    weight_bits: int = 32

    # ── GPU Context ─────────────────────────────────────────────
    gpu_name: str = ""
    num_sm: int = 0
    peak_bw_gbs: GigabytesPerSec = GigabytesPerSec(0.0)
    peak_fp32_tflops: TeraflopsPerSec = TeraflopsPerSec(0.0)

    # ── Launch Geometry ─────────────────────────────────────────
    grid_size: int = 0
    wave_count: float = 0.0
    tiles_per_seq: int = 0

    # ── Timing (from do_bench) ──────────────────────────────────
    time_ms: Milliseconds = Milliseconds(0.0)
    warmup_iters: int = 50
    bench_reps: int = 200

    # ── Analytical Metrics ──────────────────────────────────────
    total_bytes: Bytes = Bytes(0)
    total_flops: Flops = Flops(0)
    effective_flops: Flops = Flops(0)
    analytical_bw_gbs: GigabytesPerSec = GigabytesPerSec(0.0)
    bw_util_pct: Percentage = Percentage(0.0)
    analytical_gflops: float = 0.0
    arithmetic_intensity: float = 0.0
    roofline_class: RooflineClass | Literal[""] = ""

    # ── CUPTI Hardware Counters ─────────────────────────────────
    cupti_collected: bool = False
    dram_read_gbs: GigabytesPerSec = GigabytesPerSec(0.0)
    dram_write_gbs: GigabytesPerSec = GigabytesPerSec(0.0)
    dram_total_gbs: GigabytesPerSec = GigabytesPerSec(0.0)
    l2_hit_rate: FractionZeroOne = FractionZeroOne(0.0)
    sm_occupancy_pct: Percentage = Percentage(0.0)
    warp_stall_memory_pct: Percentage = Percentage(0.0)
    warp_stall_exec_pct: Percentage = Percentage(0.0)
    warp_stall_sync_pct: Percentage = Percentage(0.0)
    warp_stall_other_pct: Percentage = Percentage(0.0)
    warp_stall_dominant: StallReason | Literal[""] = ""
    inst_executed_total: int = 0
    inst_fp32_pct: Percentage = Percentage(0.0)
    inst_int32_pct: Percentage = Percentage(0.0)
    inst_fp16_pct: Percentage = Percentage(0.0)
    tensor_core_util_pct: Percentage = Percentage(0.0)

    # ── SNN-Derived Metrics ─────────────────────────────────────
    sparsity_exploitation: FractionZeroOne = FractionZeroOne(0.0)
    vmem_pressure_gbs: GigabytesPerSec = GigabytesPerSec(0.0)
    neuronal_compute_frac: FractionZeroOne = FractionZeroOne(0.0)
    edp_estimate_uj_ms: float = 0.0
    per_timestep_cost_ms: Milliseconds = Milliseconds(0.0)
    timestep_scaling_factor: float = 0.0
    layer1_edp_dominance: FractionZeroOne = FractionZeroOne(0.0)

    # ── Validation Flags ────────────────────────────────────────
    bw_analytical_vs_measured_pct: Percentage = Percentage(0.0)
    roofline_validated: bool = False

    # ── Error State ─────────────────────────────────────────────
    error_message: str = ""

    # ── Serialization (pure functions) ──────────────────────────

    def to_dict(self) -> dict[str, Any]:
        """Convert to a flat dictionary."""
        return dataclasses.asdict(self)

    def to_csv_row(self) -> dict[str, str]:
        """Convert to a flat dict of strings for CSV writing."""
        d = self.to_dict()
        return {k: str(v) for k, v in d.items()}

    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def field_names(cls) -> tuple[str, ...]:
        """Return ordered field names for CSV headers."""
        return tuple(f.name for f in fields(cls))

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SpikeProfileResult:
        """Reconstruct from a dictionary (e.g., from JSON)."""
        known = {f.name for f in fields(cls)}
        filtered = {k: v for k, v in d.items() if k in known}
        return cls(**filtered)

    @classmethod
    def from_json(cls, s: str) -> SpikeProfileResult:
        """Reconstruct from a JSON string."""
        return cls.from_dict(json.loads(s))


# ── Inference Pass Result ───────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class InferencePassResult:
    """Aggregated result for multi-kernel inference pass profiling."""

    snn_config: dict[str, Any]
    gpu_info: GpuInfo
    total_time_ms: Milliseconds
    total_edp_estimate: float
    per_timestep: tuple[tuple[SpikeProfileResult, ...], ...]  # (T,)(L,)
    component_time_frac: Mapping[KernelCategory, FractionZeroOne]
    component_energy_frac: Mapping[KernelCategory, FractionZeroOne]
    dominant_kernel: KernelCategory
    dominant_layer: int


# ── Algebraic Error Handling ────────────────────────────────────────

T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class ProfileError:
    """Structured error information for failed profiling configurations."""

    config_label: str
    error_type: Literal["launch_failure", "oom", "cupti_error", "timeout"]
    message: str


@dataclass(frozen=True, slots=True)
class Ok(Generic[T]):
    """Success variant of Result."""

    value: T


@dataclass(frozen=True, slots=True)
class Err:
    """Error variant of Result.

    Always carries a ProfileError — this keeps Result as a single-
    parameter generic: ``Result[T]`` is ``Ok[T] | Err``.
    """

    error: ProfileError


# Result is parameterised over the success type only.
# Usage: ``Result[SpikeProfileResult]`` means ``Ok[SpikeProfileResult] | Err``.
Result = Union[Ok[T], Err]


# ── CSV Batch Export ────────────────────────────────────────────────


def results_to_csv(results: list[SpikeProfileResult]) -> str:
    """Serialize a list of results to a CSV string (pure function)."""
    if not results:
        return ""
    buf = io.StringIO()
    header = SpikeProfileResult.field_names()
    writer = csv.DictWriter(buf, fieldnames=header)
    writer.writeheader()
    for r in results:
        writer.writerow(r.to_csv_row())
    return buf.getvalue()


def results_to_json(results: list[SpikeProfileResult]) -> str:
    """Serialize a list of results to a JSON array string (pure function)."""
    return json.dumps([r.to_dict() for r in results], indent=2, default=str)
