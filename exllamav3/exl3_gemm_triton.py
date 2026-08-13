"""Triton matmul operator for EXL3 linear layers.

This module provides a fused dequant+GEMM pipeline composed of three ops:

    1. torch.ops.exl3_ops.had_r_128(x, xh, suh)       # C++ custom op
    2. ext.reconstruct(w, trellis) + triton matmul      # dequant + GEMM
    3. torch.ops.exl3_ops.had_r_128(y, y, svh)          # C++ custom op

The Triton matmul (``exl3::exl3_gemm_triton``) is registered via
``torch.library.triton_op`` and does ONLY the matmul (x @ w → y).
The dequantization is handled by ``ext.reconstruct`` outside the
triton_op body (calling C++ ops from inside a triton_op causes
segfaults during dynamo tracing).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .ext import exllamav3_ext as ext

# ---------------------------------------------------------------------------
# Triton matmul kernel
# ---------------------------------------------------------------------------

@triton.jit
def _matmul_kernel(
    x_ptr, w_ptr, y_ptr,
    M, N, K: tl.constexpr,
    stride_xm, stride_xk,
    stride_wk, stride_wn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)
    num_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_n
    pid_n = pid % num_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    for k_start in range(0, K, BLOCK_K):
        offs_k = k_start + tl.arange(0, BLOCK_K)

        mask_m = offs_m < M
        mask_n = offs_n < N
        mask_k = offs_k < K

        x = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + offs_k[None, :] * stride_xk,
            mask=mask_m[:, None] & mask_k[None, :],
            other=0.0,
        )
        w = tl.load(
            w_ptr + offs_k[:, None] * stride_wk + offs_n[None, :] * stride_wn,
            mask=mask_k[:, None] & mask_n[None, :],
            other=0.0,
        )
        acc += tl.dot(x, w)

    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask,
    )


_wrapped_matmul = torch.library.wrap_triton(_matmul_kernel)


# ---------------------------------------------------------------------------
# triton_op registration — pure matmul only
# ---------------------------------------------------------------------------

@torch.library.triton_op("exl3::exl3_gemm_triton", mutates_args=("y",))
def exl3_gemm_triton(
    x: torch.Tensor,
    w: torch.Tensor,
    y: torch.Tensor,
) -> None:
    """Pure fp16 matmul: y = x @ w.

    x: [M, K] half
    w: [K, N] half
    y: [M, N] half
    """
    M, K = x.shape
    K2, N = w.shape
    assert K == K2

    BLOCK_M = min(64, M) if M > 16 else 16
    BLOCK_N = min(128, N)
    BLOCK_K = 32
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    _wrapped_matmul[grid](
        x, w, y,
        M, N, K,
        x.stride(0), x.stride(1),
        w.stride(0), w.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=4,
    )


# ---------------------------------------------------------------------------
# Fused dequant + GEMM
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
    """Fused EXL3 dequant + GEMM.

    Composes:
    1. had_r_128 — Hadamard-transform input (with suh sign correction)
    2. reconstruct + triton matmul — dequantize EXL3 weights, then GEMM
    3. had_r_128 — Hadamard-transform output (with svh sign correction)
    """
    original_shape = x.shape
    x_flat = x.view(-1, in_features)
    rows = x_flat.shape[0]

    x_half = x_flat if x_flat.dtype == torch.half else x_flat.to(torch.half)

    # Phase 1: Hadamard-transform input
    xh = torch.empty_like(x_half)
    torch.ops.exl3_ops.had_r_128(x_half, xh, suh, None, 1.0)

    # Phase 2: Dequantize weights (fused into the GEMM pipeline)
    w = torch.empty(
        (in_features, out_features),
        dtype=torch.half,
        device=trellis.device,
    )
    ext.reconstruct(w, trellis, K, mcg, mul1)

    # Phase 3: Triton matmul
    y = torch.empty((rows, out_features), dtype=out_dtype, device=device)
    torch.ops.exl3.exl3_gemm_triton(xh, w, y)

    # Phase 4: Hadamard-transform output
    torch.ops.exl3_ops.had_r_128(y, y, None, svh, 1.0)

    return y.view(original_shape[:-1] + (out_features,))
