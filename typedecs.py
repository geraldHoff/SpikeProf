"""
SpikeProf Type Foundation
=========================

All semantic newtypes, literal enumerations, typed-dict configurations,
and protocol definitions used across the codebase.

NewType aliases are zero-cost at runtime but checked statically by
mypy/pyright to prevent accidental mixing of incompatible quantities.
"""

from __future__ import annotations

from typing import (
    Any,
    Callable,
    Literal,
    Mapping,
    NewType,
    Protocol,
    Sequence,
    TypedDict,
    TypeVar,
    runtime_checkable,
)

# ── Physical Quantities ─────────────────────────────────────────────

GigabytesPerSec = NewType("GigabytesPerSec", float)   # GB/s — memory bandwidth
TeraflopsPerSec = NewType("TeraflopsPerSec", float)    # TFLOP/s — compute throughput
Milliseconds    = NewType("Milliseconds", float)       # ms — kernel execution time
Watts           = NewType("Watts", float)               # W — power consumption
Bytes           = NewType("Bytes", int)                 # B — memory traffic
Flops           = NewType("Flops", int)                 # FLOPs — floating-point operations

# ── Dimensionless Ratios ────────────────────────────────────────────

Percentage      = NewType("Percentage", float)          # 0.0–100.0 scale
FractionZeroOne = NewType("FractionZeroOne", float)     # 0.0–1.0 scale

# ── Literal Types for Fixed Enumerations ────────────────────────────

KernelCategory = Literal[
    "MATMUL", "NEURON", "ATTENTION", "ENCODE", "REDUCE", "CUSTOM"
]
RooflineClass = Literal["MEMORY_BOUND", "COMPUTE_BOUND"]
CuptiTier = Literal["minimal", "standard", "full"]
StallReason = Literal["memory", "execution", "synchronization", "other"]
ErrorType = Literal["launch_failure", "oom", "cupti_error", "timeout"]

# ── Metric Name Literal Types ──────────────────────────────────────

TimingMetric = Literal[
    "time_ms",
    "analytical_bw_gbs",
    "bw_util_pct",
    "analytical_gflops",
    "arithmetic_intensity",
]
CuptiMetric = Literal[
    "dram_read_gbs",
    "dram_write_gbs",
    "l2_hit_rate",
    "sm_occupancy_pct",
    "warp_stalls",
    "instruction_mix",
    "tensor_core_util_pct",
]
SnnMetric = Literal[
    "sparsity_exploitation",
    "vmem_pressure_gbs",
    "per_timestep_cost_ms",
    "timestep_scaling_factor",
    "edp_estimate_uj_ms",
]

# Union of all metric names
MetricName = TimingMetric | CuptiMetric | SnnMetric

# ── Convenience Metric Bundles (frozensets) ─────────────────────────

TIMING_ALL: frozenset[MetricName] = frozenset(
    {
        "time_ms",
        "analytical_bw_gbs",
        "bw_util_pct",
        "analytical_gflops",
        "arithmetic_intensity",
    }
)

CUPTI_ALL: frozenset[MetricName] = frozenset(
    {
        "dram_read_gbs",
        "dram_write_gbs",
        "l2_hit_rate",
        "sm_occupancy_pct",
        "warp_stalls",
        "instruction_mix",
        "tensor_core_util_pct",
    }
)

SNN_ALL: frozenset[MetricName] = frozenset(
    {
        "sparsity_exploitation",
        "vmem_pressure_gbs",
        "per_timestep_cost_ms",
        "timestep_scaling_factor",
        "edp_estimate_uj_ms",
    }
)

FULL: frozenset[MetricName] = TIMING_ALL | CUPTI_ALL | SNN_ALL

# SNN metrics that require CUPTI counters for accurate computation
SNN_METRICS_REQUIRING_CUPTI: frozenset[MetricName] = frozenset(
    {"vmem_pressure_gbs", "edp_estimate_uj_ms"}
)

# ── TypedDict Definitions for Sweep Grids ───────────────────────────


class KernelGrid(TypedDict, total=False):
    """Kernel tuning parameter grid for parametric sweeps."""

    block_l: list[int]
    block_n: list[int]
    num_warps: list[int]
    num_stages: list[int]


class SnnGrid(TypedDict, total=False):
    """SNN workload parameter grid for parametric sweeps."""

    batch_size: list[int]
    num_heads: list[int]
    seq_len: list[int]
    dim: list[int]
    time_steps: list[int]
    sparsity: list[float]
    tau: list[float]
    v_th: list[float]


# ── Protocol Types for Extensibility ────────────────────────────────


@runtime_checkable
class ByteModel(Protocol):
    """Pure function: workload parameters → total bytes transferred."""

    def __call__(self, **params: int) -> Bytes: ...


@runtime_checkable
class FlopModel(Protocol):
    """Pure function: workload parameters → total FLOPs (dense)."""

    def __call__(self, **params: int) -> Flops: ...


@runtime_checkable
class KernelSpec(Protocol):
    """Structural contract for kernel specifications.

    Any object providing these read-only attributes satisfies the
    contract — no inheritance required. Properties enforce read-only
    compatibility with frozen dataclasses.
    """

    @property
    def name(self) -> str: ...
    @property
    def category(self) -> KernelCategory: ...
    @property
    def kernel_fn(self) -> Callable[..., None]: ...
    @property
    def byte_model(self) -> ByteModel: ...
    @property
    def flop_model(self) -> FlopModel: ...
    @property
    def default_block_params(self) -> Mapping[str, int]: ...
    @property
    def snn_param_keys(self) -> Sequence[str]: ...
    @property
    def supports_sparsity(self) -> bool: ...


@runtime_checkable
class TimingBackend(Protocol):
    """Protocol for timing measurement backends (production + stubs)."""

    def measure(self, kernel_fn: Callable[..., None], reps: int) -> Milliseconds: ...


@runtime_checkable
class CuptiBackend(Protocol):
    """Protocol for CUPTI counter collection backends (production + stubs)."""

    def collect(
        self, kernel_fn: Callable[..., None], reps: int
    ) -> Mapping[str, float]: ...


# ── Generic TypeVar for Result and Sweep ────────────────────────────

R = TypeVar("R", bound="object")
