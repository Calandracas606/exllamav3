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

# ---------------------------------------------------------------------------
# Operator definition
# ---------------------------------------------------------------------------

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
# Python implementation (the actual computation)
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
    """Dequantize EXL3 weights and compute x @ W.

    Mirrors the current ``reconstruct_hgemm`` fallback path:
    1. Hadamard-transform the input (with suh sign correction)
    2. Dequantize the weight matrix from trellis to fp16
    3. fp16 matmul
    4. Hadamard-transform the output (with svh sign correction)

    All operations use existing extension kernels that already work on ROCm.
    """
    rows = x.shape[0]

    # Phase 1: Hadamard-transform input
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)

    # Phase 2: Dequantize weights + matmul
    MAX_SLICE_N = 32768
    if out_features <= MAX_SLICE_N:
        w = torch.empty(
            (in_features, out_features),
            dtype=torch.half,
            device=trellis.device,
        )
        ext.reconstruct(w, trellis, K, mcg, mul1)
        ext.hgemm(xh, w, y)
    else:
        numel_slice = in_features * MAX_SLICE_N
        w_buf = torch.empty((numel_slice,), dtype=torch.half, device=trellis.device)
        for n_start in range(0, out_features, MAX_SLICE_N):
            n_end = min(n_start + MAX_SLICE_N, out_features)
            n = n_end - n_start
            w = w_buf[: in_features * n].view(in_features, n)
            ext.reconstruct_slice(w, trellis, K, mcg, mul1, n_start)
            ext.hgemm(xh, w, y[:, n_start:n_end])

    # Phase 3: Hadamard-transform output (with svh sign correction)
    ext.had_r_128(y, y, None, svh, 1.0)


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

    Args:
        x: Input tensor, shape ``[batch..., in_features]``, will be
            flattened to ``[batch, in_features]``.
        trellis: EXL3-packed weight tensor.
        suh: Sign/unscale Hadamard vector for the input (row) dimension.
        svh: Sign/unscale Hadamard vector for the output (column) dimension.
        K: EXL3 bitrate parameter.
        mcg: Whether to use the MCG codebook.
        mul1: Whether to use the mul1 codebook.
        in_features: Input dimension (rows of the weight matrix).
        out_features: Output dimension (columns of the weight matrix).
        device: Device for the output tensor.
        out_dtype: Output dtype (fp16 or fp32).

    Returns:
        Output tensor of shape ``[batch..., out_features]``.
    """
    original_shape = x.shape
    x_flat = x.view(-1, in_features)
    rows = x_flat.shape[0]

    y = torch.empty(
        (rows, out_features),
        dtype=out_dtype,
        device=device,
    )

    # Cast x to half if needed (the kernel operates in fp16)
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
