"""
Sweep Engine
============

Executes parametric sweeps across the joint space of kernel tuning
parameters and SNN workload parameters. Generic over the result type.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import Any, Callable, Generic, Sequence, TypeVar, cast

from models import (
    Err,
    GpuInfo,
    Ok,
    ProfileError,
    Result,
    SpikeProfileResult,
)
from typedecs import (
    CuptiTier,
    KernelGrid,
    KernelSpec,
    Milliseconds,
    SnnGrid,
)

logger = logging.getLogger(__name__)

R = TypeVar("R")


# ── Grid Expansion ──────────────────────────────────────────────────


def expand_grid(grid: dict[str, list[Any]]) -> list[dict[str, Any]]:
    """Expand a parameter grid into a list of configurations.

    Computes the Cartesian product of all parameter lists.
    Pure function.

    >>> expand_grid({'a': [1, 2], 'b': [3]})
    [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]
    """
    if not grid:
        return [{}]

    keys = list(grid.keys())
    values = [grid[k] for k in keys]

    configs: list[dict[str, Any]] = []
    for combo in itertools.product(*values):
        configs.append(dict(zip(keys, combo)))
    return configs


def count_sweep_configs(
    kernel_grid: KernelGrid, snn_grid: SnnGrid
) -> int:
    """Count total configurations in a sweep (pure function)."""
    kg = {k: cast(list[Any], v) for k, v in kernel_grid.items() if v}
    sg = {k: cast(list[Any], v) for k, v in snn_grid.items() if v}

    kernel_count = max(1, math.prod(len(v) for v in kg.values())) if kg else 1
    snn_count = max(1, math.prod(len(v) for v in sg.values())) if sg else 1

    return kernel_count * snn_count


def select_cupti_tier(num_configs: int, requested: CuptiTier | None = None) -> CuptiTier:
    """Auto-select CUPTI tier based on sweep size. Pure function."""
    if requested is not None:
        return requested
    if num_configs < 10:
        return "full"
    elif num_configs <= 100:
        return "standard"
    else:
        return "minimal"


# ── Scope Label Builder ────────────────────────────────────────────


def build_scope_label(
    kernel_name: str,
    kernel_params: dict[str, int],
    snn_params: dict[str, Any],
) -> str:
    """Build a Proton scope label from config parameters. Pure function."""
    parts = [kernel_name]
    parts.append(f"B{snn_params.get('batch_size', 0)}")
    parts.append(f"H{snn_params.get('num_heads', 0)}")
    parts.append(f"L{snn_params.get('seq_len', 0)}")
    parts.append(f"D{snn_params.get('dim', 0)}")
    parts.append(f"T{snn_params.get('time_steps', 1)}")
    parts.append(f"BL{kernel_params.get('block_l', 0)}")
    parts.append(f"W{kernel_params.get('num_warps', 0)}")
    parts.append(f"sp{snn_params.get('sparsity', 0.5)}")
    return "_".join(parts)


# ── Generic Sweep Executor ──────────────────────────────────────────


class SweepExecutor(Generic[R]):
    """Generic sweep executor over kernel_grid × snn_grid.

    Drives single-kernel sweeps (Sweep[SpikeProfileResult]) and
    inference-pass sweeps (Sweep[InferencePassResult]).
    """

    def __init__(
        self,
        profile_fn: Callable[
            [dict[str, int], dict[str, Any]], Result[R]
        ],
    ) -> None:
        self._profile_fn = profile_fn

    def execute(
        self,
        kernel_grid: KernelGrid,
        snn_grid: SnnGrid,
        *,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[Result[R]]:
        """Execute the sweep across all grid configurations.

        Failed configurations are preserved as Err values, not dropped.
        """
        kg = {k: cast(list[Any], v) for k, v in kernel_grid.items() if v}
        sg = {k: cast(list[Any], v) for k, v in snn_grid.items() if v}

        kernel_configs = expand_grid(kg)
        snn_configs = expand_grid(sg)

        total = len(kernel_configs) * len(snn_configs)
        results: list[Result[R]] = []

        for i, (kc, sc) in enumerate(
            itertools.product(kernel_configs, snn_configs)
        ):
            if progress_callback is not None:
                progress_callback(i + 1, total)

            try:
                result = self._profile_fn(kc, sc)
                results.append(result)
            except Exception as exc:
                label = build_scope_label("unknown", kc, sc)
                results.append(
                    Err(
                        ProfileError(
                            config_label=label,
                            error_type="launch_failure",
                            message=str(exc),
                        )
                    )
                )

        return results


# ── Convenience Sweep Function ──────────────────────────────────────


def default_progress(current: int, total: int) -> None:
    """Default progress callback: prints to logger."""
    if current % max(1, total // 10) == 0 or current == total:
        logger.info("Sweep progress: %d/%d (%.0f%%)", current, total, 100 * current / total)
