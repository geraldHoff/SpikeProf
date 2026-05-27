"""
Decorator API
=============

The primary user-facing interface. ``spike_profile(metrics=[...])``
returns a decorator that wraps a Triton kernel function, preserving
its call signature while adding ``.profile()`` and ``.sweep()`` methods.

Built on higher-order functions: ``spike_profile(metrics, ...)`` is a
partial application that captures configuration, then returns a
one-argument function (the decorator) that closes over them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping

from .models import (
    Err,
    GpuInfo,
    Ok,
    ProfileError,
    Result,
    SpikeProfileResult,
)
from .metrics import compose_results
from .registry import KernelRegistry, TritonKernelSpec
from .typedecs import (
    ByteModel,
    Bytes,
    CUPTI_ALL,
    CuptiTier,
    FlopModel,
    Flops,
    KernelCategory,
    KernelGrid,
    MetricName,
    SNN_METRICS_REQUIRING_CUPTI,
    SnnGrid,
)

# ── Module-level Registry ───────────────────────────────────────────

_module_registry = KernelRegistry()


# ── CUPTI Resolution ───────────────────────────────────────────────


def _needs_cupti(metrics: frozenset[MetricName]) -> bool:
    """Pure: True iff any requested metric requires CUPTI hardware counters."""
    return bool(metrics & (CUPTI_ALL | SNN_METRICS_REQUIRING_CUPTI))


# ── ProfiledKernel ──────────────────────────────────────────────────


@dataclass(frozen=True)
class ProfiledKernel:
    """Wraps a Triton kernel with profiling capabilities.

    Callable — delegates to the raw kernel at call time.
    Provides ``.profile()`` for single-point and ``.sweep()`` for
    parametric profiling.
    """

    spec: TritonKernelSpec
    metrics: frozenset[MetricName]
    cupti_tier: CuptiTier
    warmup: int
    bench_reps: int
    cupti_reps: int

    def __call__(self, *args: Any, **kwargs: Any) -> None:
        """Delegate to the raw kernel — the decorator is transparent."""
        return self.spec.kernel_fn(*args, **kwargs)

    def profile(
        self, **workload_params: int | float
    ) -> Result[SpikeProfileResult]:
        """Profile a single configuration.

        Delegates to ``benchmark_kernel`` from the profiling module.
        """
        from profiler import benchmark_kernel

        return benchmark_kernel(
            self.spec,
            cupti=_needs_cupti(self.metrics),
            cupti_tier=self.cupti_tier,
            warmup=self.warmup,
            bench_reps=self.bench_reps,
            cupti_reps=self.cupti_reps,
            timing_backend=None,
            cupti_backend=None,
            gpu_info=None,
            **workload_params,
        )

    def sweep(
        self,
        kernel_grid: KernelGrid,
        snn_grid: SnnGrid,
        *,
        proton_session_name: str = "",
    ) -> list[Result[SpikeProfileResult]]:
        """Parametric sweep over kernel and SNN parameter grids.

        Delegates to ``profile_sweep`` from the profiling module.
        """
        from profiler import profile_sweep

        return profile_sweep(
            self.spec,
            kernel_grid=kernel_grid,
            snn_grid=snn_grid,
            cupti=_needs_cupti(self.metrics),
            cupti_tier=self.cupti_tier,
            warmup=self.warmup,
            bench_reps=self.bench_reps,
            cupti_reps=self.cupti_reps,
            proton_session_name=proton_session_name or self.spec.name + "_sweep",
        )


# ── spike_profile Decorator Factory ────────────────────────────────


def spike_profile(
    metrics: Iterable[MetricName] | frozenset[MetricName],
    category: KernelCategory,
    byte_model: ByteModel | None = None,
    flop_model: FlopModel | None = None,
    supports_sparsity: bool = False,
    cupti_tier: CuptiTier = "standard",
    warmup: int = 50,
    bench_reps: int = 200,
    cupti_reps: int = 10,
) -> Callable[[Callable[..., None]], ProfiledKernel]:
    """Higher-order function: (MetricSpec, KernelMeta) → (KernelFn → ProfiledKernel).

    Returns a decorator that wraps a Triton kernel callable, preserving
    its original call signature while adding ``.profile()`` and
    ``.sweep()`` methods.

    Usage::

        @spike_profile(
            metrics=['time_ms', 'vmem_pressure_gbs'],
            category='NEURON',
            byte_model=my_byte_model,
            flop_model=my_flop_model,
        )
        def my_kernel(U, S, W, tau, v_th, BLOCK_N: tl.constexpr):
            ...

        result = my_kernel.profile(batch_size=32, num_neurons=1024)
    """
    resolved_metrics: frozenset[MetricName] = frozenset(metrics)

    # Default byte/flop models if not provided
    _byte_model: ByteModel = byte_model or (lambda **_: Bytes(0))
    _flop_model: FlopModel = flop_model or (lambda **_: Flops(0))

    def decorator(kernel_fn: Callable[..., None]) -> ProfiledKernel:
        spec = TritonKernelSpec(
            name=kernel_fn.__name__,
            category=category,
            kernel_fn=kernel_fn,
            byte_model=_byte_model,
            flop_model=_flop_model,
            default_block_params={},
            snn_param_keys=(),
            supports_sparsity=supports_sparsity,
        )

        # Register in module-level registry
        if spec.name not in _module_registry:
            _module_registry.register(spec)

        return ProfiledKernel(
            spec=spec,
            metrics=resolved_metrics,
            cupti_tier=cupti_tier,
            warmup=warmup,
            bench_reps=bench_reps,
            cupti_reps=cupti_reps,
        )

    return decorator


# ── Profile Composition ─────────────────────────────────────────────


def compose_profiles(
    r1: Result[SpikeProfileResult],
    r2: Result[SpikeProfileResult],
) -> Result[SpikeProfileResult]:
    """Merge two profiling results from different passes.

    Used to combine timing-only and CUPTI-deep results for the same
    kernel and configuration. Pure function on immutable records.
    """
    if isinstance(r1, Err):
        return r1
    if isinstance(r2, Err):
        return r2
    return Ok(compose_results(r1.value, r2.value))
