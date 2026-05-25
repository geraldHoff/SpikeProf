"""
Data Synthesizer
================

Generates correctly-shaped spike, weight, and membrane-potential
tensors for each configuration point in a sweep.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from typedecs import FractionZeroOne

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class SynthesizedData:
    """Container for synthesized tensors. Immutable record."""

    spike: Any       # torch.Tensor (binary)
    weights: Any     # torch.Tensor (float)
    membrane: Any    # torch.Tensor (float)
    output: Any      # torch.Tensor (float, pre-allocated)
    sparsity: FractionZeroOne
    shape_info: dict[str, int]


class DataSynthesizer:
    """Generates tensors for profiling on the specified CUDA device.

    Supports static sparsity (fixed fraction) and dynamic sparsity
    (loaded from a per-layer profile file).
    """

    def __init__(
        self,
        device: str = "cuda:0",
        dtype_str: str = "float32",
        seed: int | None = None,
    ) -> None:
        self._device = device
        self._dtype_str = dtype_str
        self._seed = seed

    def synthesize(
        self,
        *,
        batch_size: int,
        in_features: int,
        out_features: int,
        sparsity: float = 0.5,
        init_membrane: str = "zeros",
    ) -> SynthesizedData:
        """Generate spike, weight, membrane, and output tensors.

        Args:
            batch_size: Batch dimension.
            in_features: Input feature dimension (M).
            out_features: Output feature dimension (N).
            sparsity: Fraction of spike elements that are zero (0.0–1.0).
            init_membrane: 'zeros' or 'random' for membrane initialization.

        Returns:
            SynthesizedData with all tensors on the configured device.
        """
        try:
            import torch  # type: ignore[import]
        except ImportError as exc:
            raise RuntimeError("PyTorch required for data synthesis.") from exc

        if self._seed is not None:
            torch.manual_seed(self._seed)

        dtype = getattr(torch, self._dtype_str, torch.float32)
        device = self._device

        # Spike tensor: binary with controlled density
        density = 1.0 - sparsity
        spike = (torch.rand(batch_size, in_features, device=device) < density).to(
            dtype
        )

        # Weight tensor: normal distribution
        weights = torch.randn(in_features, out_features, device=device, dtype=dtype)

        # Membrane potential
        if init_membrane == "random":
            membrane = torch.rand(
                batch_size, out_features, device=device, dtype=dtype
            )
        else:
            membrane = torch.zeros(
                batch_size, out_features, device=device, dtype=dtype
            )

        # Pre-allocated output
        output = torch.zeros(batch_size, out_features, device=device, dtype=dtype)

        return SynthesizedData(
            spike=spike,
            weights=weights,
            membrane=membrane,
            output=output,
            sparsity=FractionZeroOne(sparsity),
            shape_info={
                "batch_size": batch_size,
                "in_features": in_features,
                "out_features": out_features,
            },
        )

    def check_memory(
        self,
        *,
        batch_size: int,
        in_features: int,
        out_features: int,
        elem_size: int = 4,
        safety_factor: float = 0.9,
    ) -> bool:
        """Check if the GPU has enough memory for a configuration.

        Returns True if estimated usage is within ``safety_factor`` of
        available memory.
        """
        try:
            import torch  # type: ignore[import]
        except ImportError:
            return True  # Can't check, assume OK

        if not torch.cuda.is_available():
            return True

        estimated_bytes = (
            batch_size * in_features * elem_size  # spike (as float)
            + in_features * out_features * elem_size  # weights
            + batch_size * out_features * elem_size  # membrane
            + batch_size * out_features * elem_size  # output
        )
        free, _ = torch.cuda.mem_get_info()
        return estimated_bytes < free * safety_factor

    @staticmethod
    def free_tensors(data: SynthesizedData) -> None:
        """Explicitly free tensors and clear CUDA cache."""
        try:
            import torch  # type: ignore[import]

            del data  # Remove reference
            torch.cuda.empty_cache()
        except ImportError:
            pass
