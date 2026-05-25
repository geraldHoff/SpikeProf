"""
Kernel Registry
===============

Typed catalog of SNN kernel specifications. Uses Protocol-based
structural typing — kernels need only satisfy the ``KernelSpec``
interface, no inheritance required.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping, Sequence

from typedecs import (
    ByteModel,
    Bytes,
    FlopModel,
    Flops,
    KernelCategory,
    KernelSpec,
)

# ── Concrete Kernel Specification ───────────────────────────────────


@dataclass(frozen=True, slots=True)
class TritonKernelSpec:
    """Concrete, immutable kernel specification satisfying KernelSpec Protocol."""

    name: str
    category: KernelCategory
    kernel_fn: Callable[..., None]
    byte_model: ByteModel
    flop_model: FlopModel
    default_block_params: Mapping[str, int]
    snn_param_keys: tuple[str, ...]  # tuple for hashability and immutability
    supports_sparsity: bool


# ── Registry ────────────────────────────────────────────────────────


class KernelRegistry:
    """Registry of kernel specifications, keyed by name.

    Maintains a typed catalog. Registration validates that the spec
    satisfies the KernelSpec protocol at runtime.
    """

    def __init__(self) -> None:
        self._specs: dict[str, KernelSpec] = {}

    def register(self, spec: KernelSpec) -> None:
        """Register a kernel specification.

        Raises:
            TypeError: If spec does not satisfy KernelSpec Protocol.
            ValueError: If a spec with the same name is already registered.
        """
        if not isinstance(spec, KernelSpec):
            raise TypeError(
                f"Spec must satisfy KernelSpec protocol, got {type(spec).__name__}"
            )
        if spec.name in self._specs:
            raise ValueError(
                f"Kernel '{spec.name}' is already registered. "
                "Unregister it first or use a different name."
            )
        self._specs[spec.name] = spec

    def unregister(self, name: str) -> None:
        """Remove a kernel specification by name."""
        if name not in self._specs:
            raise KeyError(f"Kernel '{name}' is not registered.")
        del self._specs[name]

    def __getitem__(self, name: str) -> KernelSpec:
        """Look up a kernel specification by name."""
        if name not in self._specs:
            raise KeyError(
                f"Kernel '{name}' not found. "
                f"Registered kernels: {list(self._specs.keys())}"
            )
        return self._specs[name]

    def __contains__(self, name: str) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    def list_kernels(self) -> list[str]:
        """Return sorted list of registered kernel names."""
        return sorted(self._specs.keys())

    def kernels_by_category(self, category: KernelCategory) -> list[KernelSpec]:
        """Return all specs matching a given category."""
        return [s for s in self._specs.values() if s.category == category]


# ── Built-in Byte and FLOP Models ──────────────────────────────────


def matmul_byte_model(
    *, B: int = 0, M: int = 0, N: int = 0, elem: int = 4, **_: int
) -> Bytes:
    """Byte model for spike-driven MatMul: read S, read W, write output."""
    s_bytes = B * M * 1  # Binary spike tensor (1 byte per element)
    w_bytes = M * N * elem  # Weight matrix
    o_bytes = B * N * elem  # Output
    return Bytes(s_bytes + w_bytes + o_bytes)


def matmul_flop_model(*, B: int = 0, M: int = 0, N: int = 0, **_: int) -> Flops:
    """FLOP model for dense spike-driven MatMul: 2·B·M·N."""
    return Flops(2 * B * M * N)


def neuron_byte_model(
    *, B: int = 0, N: int = 0, elem: int = 4, **_: int
) -> Bytes:
    """Byte model for LIF/IF neuron: read/write U, read MAC input, write spikes."""
    u_read = B * N * elem
    u_write = B * N * elem
    mac_input = B * N * elem
    spike_out = B * N * 1  # Binary output
    return Bytes(u_read + u_write + mac_input + spike_out)


def neuron_flop_model(*, B: int = 0, N: int = 0, **_: int) -> Flops:
    """FLOP model for LIF neuron: multiply(leak) + add(integrate) + compare(fire)."""
    return Flops(3 * B * N)


def encode_byte_model(
    *, B: int = 0, N: int = 0, elem: int = 4, **_: int
) -> Bytes:
    """Byte model for spike encoding: read input, write binary output."""
    return Bytes(B * N * elem + B * N * 1)


def encode_flop_model(*, B: int = 0, N: int = 0, **_: int) -> Flops:
    """FLOP model for spike encoding: minimal arithmetic (random + compare)."""
    return Flops(2 * B * N)


def reduce_byte_model(
    *, B: int = 0, N: int = 0, T: int = 1, elem: int = 4, **_: int
) -> Bytes:
    """Byte model for membrane reduction: read T potentials, write 1."""
    return Bytes(B * N * T * elem + B * N * elem)


def reduce_flop_model(*, B: int = 0, N: int = 0, T: int = 1, **_: int) -> Flops:
    """FLOP model for membrane reduction: T-1 additions per element."""
    return Flops(max(0, T - 1) * B * N)
