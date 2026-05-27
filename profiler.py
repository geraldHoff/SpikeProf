"""
Profiler — Orchestration Layer
==============================

Top-level profiling functions that compose the timing core, CUPTI core,
SNN metrics, and sweep engine. This is the effectful shell — all side
effects (GPU launches, CUPTI sessions) are confined here.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

from .models import (
    Err,
    GpuInfo,
    InferencePassResult,
    Ok,
    ProfileError,
    Result,
    SpikeProfileResult,
)
from .metrics import (
    compute_edp,
    compute_neuronal_compute_frac,
    compute_per_timestep_cost,
    compute_vmem_pressure,
    estimate_power,
    with_snn_metrics,
)
from .cupti import (
    StubCuptiBackend,
    apply_cupti_to_result,
    collect_counters,
)
from .sweep import (
    SweepExecutor,
    build_scope_label,
    default_progress,
    select_cupti_tier,
    count_sweep_configs,
)
from .timing import (
    StubTimingBackend,
    TritonTimingBackend,
    build_timing_result,
    time_kernel,
)
from .typedecs import (
    CuptiBackend,
    CuptiTier,
    FractionZeroOne,
    GigabytesPerSec,
    KernelCategory,
    KernelGrid,
    KernelSpec,
    Milliseconds,
    SnnGrid,
    TimingBackend,
    Watts,
)

logger = logging.getLogger(__name__)


# ── Single-Kernel Benchmark ────────────────────────────────────────


def benchmark_kernel(
    spec: KernelSpec,
    *,
    cupti: bool = False,
    cupti_tier: CuptiTier = "standard",
    warmup: int = 50,
    bench_reps: int = 200,
    cupti_reps: int = 10,
    timing_backend: TimingBackend | None = None,
    cupti_backend: CuptiBackend | None = None,
    gpu_info: GpuInfo | None = None,
    **workload_params: int | float,
) -> Result[SpikeProfileResult]:
    """Profile a single kernel at one configuration.

    Returns Ok(SpikeProfileResult) on success, Err(ProfileError) on failure.
    """
    # Get or detect GPU info
    if gpu_info is None:
        try:
            from gpu_detector import detect_gpu

            gpu_info = detect_gpu()
        except RuntimeError as exc:
            return Err(
                ProfileError(
                    config_label=spec.name,
                    error_type="launch_failure",
                    message=f"GPU detection failed: {exc}",
                )
            )

    # Separate kernel params from workload params
    kernel_param_names = {"block_l", "block_n", "num_warps", "num_stages"}
    kernel_params = {
        k: int(v) for k, v in workload_params.items() if k in kernel_param_names
    }
    wl_params: dict[str, Any] = {
        k: v for k, v in workload_params.items() if k not in kernel_param_names
    }
    # Ensure int conversion for byte/flop model params
    wl_int: dict[str, int] = {}
    for k, v in wl_params.items():
        try:
            wl_int[k] = int(v)
        except (ValueError, TypeError):
            pass

    # Build kernel launch closure
    def kernel_launch() -> None:
        spec.kernel_fn()

    # Time the kernel
    tb = timing_backend or StubTimingBackend(Milliseconds(1.0))
    timing_result = time_kernel(kernel_launch, timing_backend=tb, bench_reps=bench_reps)

    if isinstance(timing_result, Err):
        return timing_result

    time_ms = timing_result.value

    # Build result with timing metrics
    result = build_timing_result(spec, time_ms, gpu_info, wl_params, kernel_params)

    # Collect CUPTI counters if requested
    if cupti:
        cb = cupti_backend or StubCuptiBackend(counters={})
        counter_result = collect_counters(kernel_launch, cupti_backend=cb, cupti_reps=cupti_reps)

        if isinstance(counter_result, Ok) and counter_result.value:
            kernel_time_ns = time_ms * 1e6  # ms → ns
            result = apply_cupti_to_result(result, counter_result.value, kernel_time_ns)

    # Compute SNN-specific metrics
    # Vmem pressure for NEURON kernels
    vmem = GigabytesPerSec(0.0)
    if spec.category == "NEURON":
        batch = int(wl_params.get("batch_size", wl_params.get("B", 0)))
        neurons = int(wl_params.get("num_neurons", wl_params.get("N", wl_params.get("dim", 0))))
        vmem = compute_vmem_pressure(batch, neurons, 4, time_ms)

    # EDP estimate
    power = estimate_power(result.sm_occupancy_pct, gpu_info.tdp_watts)
    edp = compute_edp(power, time_ms)

    # Per-timestep cost
    ts = int(wl_params.get("time_steps", 1))
    per_ts = compute_per_timestep_cost(time_ms, ts)

    result = with_snn_metrics(
        result,
        vmem_pressure=vmem,
        edp=edp,
        per_timestep_cost=per_ts,
    )

    return Ok(result)


# ── Parametric Sweep ────────────────────────────────────────────────


def profile_sweep(
    spec: KernelSpec,
    *,
    kernel_grid: KernelGrid,
    snn_grid: SnnGrid,
    cupti: bool = False,
    cupti_tier: CuptiTier | None = None,
    warmup: int = 50,
    bench_reps: int = 200,
    cupti_reps: int = 10,
    timing_backend: TimingBackend | None = None,
    cupti_backend: CuptiBackend | None = None,
    gpu_info: GpuInfo | None = None,
    proton_session_name: str = "",
) -> list[Result[SpikeProfileResult]]:
    """Execute a parametric sweep across kernel × SNN grids.

    Returns a list of Result values — failed configurations preserved as Err.
    """
    num_configs = count_sweep_configs(kernel_grid, snn_grid)
    tier = select_cupti_tier(num_configs, cupti_tier)

    def profile_fn(
        kc: dict[str, int], sc: dict[str, Any]
    ) -> Result[SpikeProfileResult]:
        all_params: dict[str, int | float] = {}
        all_params.update(kc)
        all_params.update(sc)
        return benchmark_kernel(
            spec,
            cupti=cupti,
            cupti_tier=tier,
            warmup=warmup,
            bench_reps=bench_reps,
            cupti_reps=cupti_reps,
            timing_backend=timing_backend,
            cupti_backend=cupti_backend,
            gpu_info=gpu_info,
            **all_params,
        )

    executor: SweepExecutor[SpikeProfileResult] = SweepExecutor(profile_fn)
    return executor.execute(
        kernel_grid,
        snn_grid,
        progress_callback=default_progress,
    )


# ── Multi-Kernel Inference Pass ─────────────────────────────────────


def profile_inference_pass(
    *,
    kernels: Sequence[KernelSpec],
    layer_configs: Sequence[dict[str, int]],
    snn_config: dict[str, Any],
    cupti: bool = False,
    cupti_tier: CuptiTier = "standard",
    timing_backend: TimingBackend | None = None,
    cupti_backend: CuptiBackend | None = None,
    gpu_info: GpuInfo | None = None,
) -> Result[InferencePassResult]:
    """Profile a multi-kernel, multi-timestep inference pass.

    Profiles all layers across all time-steps and produces an
    InferencePassResult with component breakdowns.
    """
    if gpu_info is None:
        try:
            from gpu_detector import detect_gpu

            gpu_info = detect_gpu()
        except RuntimeError as exc:
            return Err(
                ProfileError(
                    config_label="inference_pass",
                    error_type="launch_failure",
                    message=f"GPU detection failed: {exc}",
                )
            )

    time_steps = int(snn_config.get("time_steps", 1))
    all_results: list[list[SpikeProfileResult]] = []  # (T, L)

    for t in range(time_steps):
        timestep_results: list[SpikeProfileResult] = []
        for layer_idx, (kernel, layer_cfg) in enumerate(
            zip(kernels, layer_configs)
        ):
            params: dict[str, int | float] = dict(snn_config)
            params.update(layer_cfg)
            params["time_step_index"] = t
            params["layer_index"] = layer_idx

            result = benchmark_kernel(
                kernel,
                cupti=cupti,
                cupti_tier=cupti_tier,
                warmup=50,
                bench_reps=200,
                cupti_reps=10,
                timing_backend=timing_backend,
                cupti_backend=cupti_backend,
                gpu_info=gpu_info,
                **params,
            )

            if isinstance(result, Err):
                return result
            timestep_results.append(result.value)

        all_results.append(timestep_results)

    # Flatten for aggregate computation
    flat_results = [r for ts in all_results for r in ts]
    total_time = Milliseconds(sum(r.time_ms for r in flat_results))
    neuronal_frac = compute_neuronal_compute_frac(flat_results)

    # Component breakdown by category
    category_times: dict[KernelCategory, float] = {}
    for r in flat_results:
        category_times[r.kernel_category] = (
            category_times.get(r.kernel_category, 0.0) + r.time_ms
        )

    total_t = sum(category_times.values()) or 1.0
    component_time: dict[KernelCategory, FractionZeroOne] = {
        cat: FractionZeroOne(t / total_t) for cat, t in category_times.items()
    }

    # EDP estimates
    total_edp = sum(r.edp_estimate_uj_ms for r in flat_results)
    category_edp: dict[KernelCategory, float] = {}
    for r in flat_results:
        category_edp[r.kernel_category] = (
            category_edp.get(r.kernel_category, 0.0) + r.edp_estimate_uj_ms
        )
    total_e = sum(category_edp.values()) or 1.0
    component_energy: dict[KernelCategory, FractionZeroOne] = {
        cat: FractionZeroOne(e / total_e) for cat, e in category_edp.items()
    }

    # Dominant kernel and layer
    dominant_kernel = max(category_times, key=category_times.get)  # type: ignore[arg-type]
    dominant_layer = max(
        range(len(layer_configs)),
        key=lambda i: sum(all_results[t][i].time_ms for t in range(time_steps)),
    )

    # Build per_timestep tuples (immutable)
    per_ts = tuple(tuple(ts) for ts in all_results)

    return Ok(
        InferencePassResult(
            snn_config=dict(snn_config),
            gpu_info=gpu_info,
            total_time_ms=total_time,
            total_edp_estimate=total_edp,
            per_timestep=per_ts,
            component_time_frac=component_time,
            component_energy_frac=component_energy,
            dominant_kernel=dominant_kernel,
            dominant_layer=dominant_layer,
        )
    )
