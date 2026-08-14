"""EXL3 fused dequantize + GEMM as a PyTorch custom operator.

This module defines a proper torch custom operator for the EXL3 quantized
linear layer's core computation: dequantize EXL3-packed weights and multiply
by the (Hadamard-transformed) input.

The operator is registered via ``torch.library`` so it composes correctly with
``torch.compile``, CUDA graph capture, and other PyTorch subsystems.

The current implementation uses the same reconstruct+hgemm fallback as the
existing ROCm path. Future work: replace the Python kernel with a Triton
fused dequant+GEMM kernel for dramatic speedup.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .ext import exllamav3_ext as ext

_EXL3_LIBRARY_NAME = "exl3"

# Register the library and operator (idempotent)
_lib = torch.library.Library(_EXL3_LIBRARY_NAME, "DEF")
try:
    _lib.define(
        "exl3_gemm("
        "Tensor x, Tensor(a!) y, Tensor trellis, Tensor suh, Tensor svh, "
        "int K, bool mcg, bool mul1, int in_features, int out_features"
        ") -> ()"
    )
except RuntimeError:
    pass  # Already defined (re-import)


# ---------------------------------------------------------------------------
# FakeTensor (meta) kernel — required for torch.compile / dynamo support
# ---------------------------------------------------------------------------

@torch.library.register_fake(f"{_EXL3_LIBRARY_NAME}::exl3_gemm")
def _exl3_gemm_fake(x, y, trellis, suh, svh, K, mcg, mul1, in_features, out_features):
    # y is mutated in-place; no new tensors are returned
    return


# ---------------------------------------------------------------------------
# had_r_128 custom op (registered via TORCH_LIBRARY in bindings_stable.cpp)
# ---------------------------------------------------------------------------

# Register FakeTensor (Meta) kernel for torch.compile support
@torch.library.register_fake("exl3_ops::had_r_128")
def _had_r_128_fake(input, output, pre_scale, post_scale, scale):
    # output is mutated in-place; no return value
    return


def had_r_128(
    input: torch.Tensor,
    output: torch.Tensor,
    pre_scale: torch.Tensor | None = None,
    post_scale: torch.Tensor | None = None,
    scale: float = 1.0,
) -> None:
    """128-element row Hadamard transform with optional sign scaling.

    Wraps the exl3_ops::had_r_128 custom operator registered in
    bindings_stable.cpp via TORCH_LIBRARY. Mutates `output` in-place.
    """
    torch.ops.exl3_ops.had_r_128(input, output, pre_scale, post_scale, scale)


# ---------------------------------------------------------------------------
# Python implementation — uses Triton fused dequant+GEMM
# ---------------------------------------------------------------------------

def _exl3_gemm_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    K: int,
    mcg: bool,
    mul1: bool,
    in_features: int,
    out_features: int,
) -> None:
    """Dequantize EXL3 weights and compute x @ W using Triton fused kernel."""
    from .exl3_gemm_triton import exl3_gemm as triton_exl3_gemm

    original_shape = x.shape
    x_flat = x.view(-1, in_features)
    rows = x_flat.shape[0]

    x_half = x_flat if x_flat.dtype == torch.half else x_flat.to(torch.half)

    result = triton_exl3_gemm(
        x_half,
        trellis,
        suh,
        svh,
        K,
        mcg,
        mul1,
        in_features,
        out_features,
        trellis.device,
        y.dtype,
    )
    y.copy_(result.view(rows, out_features))


# Register the Python implementation for all device types
@torch.library.impl(f"{_EXL3_LIBRARY_NAME}::exl3_gemm", "default")
def _exl3_gemm_registered(
    x: torch.Tensor,
    y: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    K: int,
    mcg: bool,
    mul1: bool,
    in_features: int,
    out_features: int,
) -> None:
    _exl3_gemm_impl(x, y, trellis, suh, svh, K, mcg, mul1, in_features, out_features)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def exl3_gemm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    K: int,
    mcg: bool,
    mul1: bool,
    in_features: int,
    out_features: int,
    device: torch.device,
    out_dtype: torch.dtype = torch.half,
) -> torch.Tensor:
    """Compute EXL3 fused dequant+GEMM.

    Allocates the output tensor and calls the registered custom op.
    """
    original_shape = x.shape
    x_flat = x.view(-1, in_features)
    rows = x_flat.shape[0]

    y = torch.empty(
        (rows, out_features),
        dtype=out_dtype,
        device=device,
    )

    x_half = x_flat if x_flat.dtype == torch.half else x_flat.to(torch.half)

    torch.ops.exl3.exl3_gemm(
        x_half,
        y,
        trellis,
        suh,
        svh,
        K,
        mcg,
        mul1,
        in_features,
        out_features,
    )

    return y.view(original_shape[:-1] + (out_features,))
