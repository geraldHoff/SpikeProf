"""
GPU Detector
============

Detects the current GPU's properties and computes peak theoretical
specifications. Runs once at profiling startup.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from .models import GpuInfo
from .typedecs import (
    Bytes,
    GigabytesPerSec,
    TeraflopsPerSec,
    Watts,
)

logger = logging.getLogger(__name__)

# ── GPU Lookup Table ────────────────────────────────────────────────


@dataclass(frozen=True, slots=True)
class _GpuTableEntry:
    peak_bw_gbs: float
    peak_fp32_tflops: float
    peak_fp16_tflops: float
    peak_int32_tflops: float
    tdp_watts: float


_GPU_TABLE: dict[str, _GpuTableEntry] = {
    "A100-SXM": _GpuTableEntry(2039, 19.5, 312.0, 19.5, 400),
    "A100-PCIe": _GpuTableEntry(1555, 19.5, 312.0, 19.5, 300),
    "A100": _GpuTableEntry(2039, 19.5, 312.0, 19.5, 400),
    "H100-SXM": _GpuTableEntry(3352, 51.2, 989.4, 51.2, 700),
    "H100-PCIe": _GpuTableEntry(2000, 51.2, 989.4, 51.2, 350),
    "H100": _GpuTableEntry(3352, 51.2, 989.4, 51.2, 700),
    "L4": _GpuTableEntry(300, 30.3, 242.0, 30.3, 72),
    "RTX 4090": _GpuTableEntry(1008, 82.6, 165.2, 82.6, 450),
    "RTX 4080": _GpuTableEntry(717, 48.7, 97.5, 48.7, 320),
    "RTX 3090": _GpuTableEntry(936, 35.6, 71.0, 35.6, 350),
    "RTX 3080": _GpuTableEntry(760, 29.8, 59.6, 29.8, 320),
    "V100": _GpuTableEntry(900, 14.0, 28.0, 14.0, 300),
}


def _lookup_gpu(name: str) -> _GpuTableEntry | None:
    """Try to match the GPU name against the lookup table."""
    name_upper = name.upper()
    for key, entry in _GPU_TABLE.items():
        if key.upper() in name_upper:
            return entry
    return None


# ── Detection ───────────────────────────────────────────────────────


def detect_gpu(device_index: int = 0) -> GpuInfo:
    """Detect GPU properties from the current CUDA device.

    Falls back to estimation from device properties if the GPU is not
    in the lookup table. Raises RuntimeError if no CUDA device is
    available.
    """
    try:
        import torch  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for GPU detection. "
            "Install with: pip install torch"
        ) from exc

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA device available for GPU detection.")

    props = torch.cuda.get_device_properties(device_index)
    name: str = props.name
    cc = (props.major, props.minor)

    entry = _lookup_gpu(name)

    if entry is not None:
        peak_bw = entry.peak_bw_gbs
        peak_fp32 = entry.peak_fp32_tflops
        peak_fp16 = entry.peak_fp16_tflops
        peak_int32 = entry.peak_int32_tflops
        tdp = entry.tdp_watts
    else:
        # Estimate from device properties
        logger.warning(
            "GPU '%s' not in lookup table — estimating specs from device properties.",
            name,
        )
        # Memory bandwidth: bus_width * clock * 2 (DDR) / 8 (bits→bytes) / 1e9
        mem_clock_ghz = props.mem_clock_rate / 1e6  # kHz → GHz
        bus_width_bytes = props.memory_bus_width / 8
        peak_bw = mem_clock_ghz * bus_width_bytes * 2  # DDR factor

        # FP32 TFLOP/s: SM_count * clock * FMA_per_clock / 1e12
        core_clock_ghz = props.clock_rate / 1e6  # kHz → GHz
        # Rough estimate: 128 FP32 cores per SM for modern GPUs
        fp32_cores_per_sm = 128
        peak_fp32 = (
            props.multi_processor_count
            * core_clock_ghz
            * fp32_cores_per_sm
            * 2  # FMA = 2 ops
            / 1e3  # → TFLOP/s
        )
        peak_fp16 = peak_fp32 * 2  # FP16 typically 2× FP32
        peak_int32 = peak_fp32
        tdp = 300.0  # Conservative default

    return GpuInfo(
        name=name,
        compute_capability=cc,
        num_sm=props.multi_processor_count,
        max_threads_per_sm=props.max_threads_per_multi_processor,
        max_regs_per_sm=props.regs_per_multiprocessor,
        max_smem_per_sm=Bytes(props.max_shared_memory_size_per_multiprocessor),
        warp_size=props.warp_size if hasattr(props, "warp_size") else 32,
        peak_bw_gbs=GigabytesPerSec(peak_bw),
        peak_fp32_tflops=TeraflopsPerSec(peak_fp32),
        peak_fp16_tflops=TeraflopsPerSec(peak_fp16),
        peak_int32_tflops=TeraflopsPerSec(peak_int32),
        tdp_watts=Watts(tdp),
    )


def make_gpu_info(
    *,
    name: str = "Unknown GPU",
    compute_capability: tuple[int, int] = (8, 0),
    num_sm: int = 108,
    max_threads_per_sm: int = 2048,
    max_regs_per_sm: int = 65536,
    max_smem_per_sm: int = 167936,
    warp_size: int = 32,
    peak_bw_gbs: float = 2039.0,
    peak_fp32_tflops: float = 19.5,
    peak_fp16_tflops: float = 312.0,
    peak_int32_tflops: float = 19.5,
    tdp_watts: float = 400.0,
) -> GpuInfo:
    """Factory function for GpuInfo — handles NewType wrapping."""
    return GpuInfo(
        name=name,
        compute_capability=compute_capability,
        num_sm=num_sm,
        max_threads_per_sm=max_threads_per_sm,
        max_regs_per_sm=max_regs_per_sm,
        max_smem_per_sm=Bytes(max_smem_per_sm),
        warp_size=warp_size,
        peak_bw_gbs=GigabytesPerSec(peak_bw_gbs),
        peak_fp32_tflops=TeraflopsPerSec(peak_fp32_tflops),
        peak_fp16_tflops=TeraflopsPerSec(peak_fp16_tflops),
        peak_int32_tflops=TeraflopsPerSec(peak_int32_tflops),
        tdp_watts=Watts(tdp_watts),
    )
