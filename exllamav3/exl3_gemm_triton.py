"""Triton matmul operator for EXL3 linear layers.

This module provides a fused dequant+GEMM pipeline composed of three ops:

    1. torch.ops.exl3_ops.had_r_128(x, xh, suh)       # C++ custom op
    2. dequant_trellis + triton matmul                   # pure-PyTorch dequant + GEMM
    3. torch.ops.exl3_ops.had_r_128(y, y, svh)          # C++ custom op

The Triton matmul (``exl3::exl3_gemm_triton``) is registered via
``torch.library.triton_op`` and does ONLY the matmul (x @ w -> y).
The dequantization is done in pure PyTorch (no C++ ext.reconstruct call)
using a precomputed decode LUT and bitstream window extraction.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

from .ext import exllamav3_ext as ext

# ---------------------------------------------------------------------------
# EXL3 dequantization in pure PyTorch
# ---------------------------------------------------------------------------

_TENSOR_CORE_PERM = None
_TENSOR_CORE_PERM_I = None

def _get_perm(device):
    global _TENSOR_CORE_PERM, _TENSOR_CORE_PERM_I
    if _TENSOR_CORE_PERM is None or _TENSOR_CORE_PERM.device != device:
        perm = [0] * 256
        for t in range(32):
            r0 = (t % 4) * 2; r1 = r0 + 1; r2 = r0 + 8; r3 = r0 + 9
            c0 = t // 4; c1 = c0 + 8
            perm[t*8+0] = r0*16+c0; perm[t*8+1] = r1*16+c0
            perm[t*8+2] = r2*16+c0; perm[t*8+3] = r3*16+c0
            perm[t*8+4] = r0*16+c1; perm[t*8+5] = r1*16+c1
            perm[t*8+6] = r2*16+c1; perm[t*8+7] = r3*16+c1
        perm_i = [0]*256
        for i, p in enumerate(perm):
            perm_i[p] = i
        _TENSOR_CORE_PERM = torch.tensor(perm, device=device, dtype=torch.long)
        _TENSOR_CORE_PERM_I = torch.tensor(perm_i, device=device, dtype=torch.long)
    return _TENSOR_CORE_PERM_I


_DQ_CACHE = {}

def _dq_indices(bits: int, device):
    key = (bits, str(device))
    if key not in _DQ_CACHE:
        e = torch.arange(256, device=device, dtype=torch.int64)
        lane = e // 8; r = e % 8; n = bits * 256 // 32
        if bits in (5, 6, 8):
            t = (e // 4) * 4; j = e % 4
            b0 = (t + 257) * bits - 16; b2 = (t + 260) * bits
            i0 = b0 // 32; i2 = (b2 - 1) // 32; s2 = (i2 + 1) * 32 - b2
            low_idx = i2 % n; high_idx = i0 % n; shift = s2 + (3 - j) * bits
        elif bits == 7:
            t = (e // 4) * 4; i = e % 4 // 2; j = e % 4 % 2
            b0 = (t + 2*i + 257) * bits - 16; b2 = (t + 2*i + 258) * bits
            i0 = b0 // 32; i2 = (b2 - 1) // 32; s2 = (i2 + 1) * 32 - b2
            low_idx = i2 % n; high_idx = i0 % n; shift = s2 + (1 - j) * bits
        elif bits == 3:
            t_offset = lane * 8
            b1 = (t_offset + 257) * bits; b0 = b1 - 16; b2 = b1 + bits * 7
            i0 = b0 // 32; i2 = (b2 - 1) // 32; s2 = (i2 + 1) * 32 - b2
            low_idx = i2 % n; high_idx = i0 % n; shift = s2 + (7 - r) * bits
        elif bits == 4:
            q = lane; low_idx = q; high_idx = (q + 31) % 32; shift = (7 - r) * 4
        elif bits == 2:
            q16 = e // 16; i1 = q16; i0 = (i1 + 15) % 16
            shift0 = ((~(lane * 8)) & 8) * 2
            low_idx = i1; high_idx = i0; shift = shift0 + (7 - r) * 2
        elif bits == 1:
            q32 = e // 32; i1 = q32; i0 = (i1 + 7) % 8
            shift0 = (~(lane * 8)) & 24
            low_idx = i1; high_idx = i0; shift = shift0 + (7 - r)
        else:
            raise ValueError(f"Unsupported bits={bits}")
        _DQ_CACHE[key] = (low_idx, high_idx, shift)
    return _DQ_CACHE[key]


_LUT_CACHE = {}

def _decode_lut(cb: int, device) -> torch.Tensor:
    key = (cb, str(device))
    if key not in _LUT_CACHE:
        x = torch.arange(65536, device=device, dtype=torch.int64)
        M = 0xFFFFFFFF
        if cb == 0:
            x = (x * 89226354) & M; x = (x + 64248484) & M
            x = 0x3b603b60 ^ (x & 0x8fff8fff)
            lo = (x & 0xFFFF).to(torch.int16).view(torch.float16)
            hi = ((x >> 16) & 0xFFFF).to(torch.int16).view(torch.float16)
            lut = lo + hi
        elif cb == 1:
            x = (x * 0xCBAC1FED) & M
            x = 0x3b603b60 ^ (x & 0x8fff8fff)
            lo = (x & 0xFFFF).to(torch.int16).view(torch.float16)
            hi = ((x >> 16) & 0xFFFF).to(torch.int16).view(torch.float16)
            lut = lo + hi
        elif cb == 2:
            x = (x * 0x83DCD12D) & M
            acc = torch.full_like(x, 0x6400)
            s = (acc + (x & 0xFF) + ((x >> 8) & 0xFF) + ((x >> 16) & 0xFF) + ((x >> 24) & 0xFF)) & 0xFFFF
            sum_h = s.to(torch.int16).view(torch.float16)
            k_inv = torch.tensor([0x1eee], dtype=torch.int16, device=device).view(torch.float16)
            k_bias_data = torch.tensor([0xc931], dtype=torch.int32, device=device).to(torch.int16).view(torch.float16)
            lut = sum_h * k_inv + k_bias_data
        _LUT_CACHE[key] = lut
    return _LUT_CACHE[key]


def dequant_trellis(trellis: torch.Tensor, K: int, mcg: bool = False, mul1: bool = False) -> torch.Tensor:
    """Pure PyTorch EXL3 dequantization. Replaces ext.reconstruct."""
    device = trellis.device
    bits = K
    rows, cols = trellis.shape[0], trellis.shape[1]
    n_u32 = bits * 256 // 32

    cb = 1 if mcg else (2 if mul1 else 0)
    lut = _decode_lut(cb, device)

    u16 = (trellis.to(torch.int32) & 0xFFFF).to(torch.int64)
    ptr = u16[..., 0::2] | (u16[..., 1::2] << 16)

    low_idx, high_idx, shift = _dq_indices(bits, device)

    b = ptr[:, :, low_idx]
    a = ptr[:, :, high_idx]
    merged = ((a << 32) | b) >> shift
    w0 = (merged & 0xFFFF).to(torch.int64)

    decoded = lut[w0]

    perm_i = _get_perm(device)
    decoded = decoded[:, :, perm_i]
    decoded = decoded.view(rows, cols, 16, 16).permute(0, 2, 1, 3).contiguous()
    return decoded.view(rows * 16, cols * 16)


# ---------------------------------------------------------------------------
# Triton matmul kernel
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fused dequant + GEMM Triton kernel
# ---------------------------------------------------------------------------

@triton.jit
def _fused_dequant_gemm_kernel(
    x_ptr, y_ptr,
    trellis_ptr,
    lut_ptr,
    perm_i_ptr,
    M, N, K_dim,
    tiles_n,
    packed_size,
    n_u32,
    stride_xm, stride_xk,
    stride_tk, stride_tn, stride_tp,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    K_BITS: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n_tile = tl.program_id(1)  # one 16-column tile

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, 16)
    offs_n = pid_n_tile * 16 + tl.arange(0, 16)

    mask_m = offs_m < M

    acc = tl.zeros((BLOCK_M, 16), dtype=tl.float32)

    n_tiles_k = K_dim // 16

    for k_tile in range(n_tiles_k):
        # Load x block: [BLOCK_M, 16]
        k_offset = k_tile * 16 + offs_k
        x_block = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_offset[None, :] * stride_xk,
            mask=mask_m[:, None],
            other=0.0,
        )

        tile_base = k_tile * stride_tk + pid_n_tile * stride_tn

        # Element index within 16x16 tile via inverse perm
        local_k = tl.arange(0, 16)[:, None]  # [16, 1]
        local_n = tl.arange(0, 16)[None, :]  # [1, 16]
        elem_flat = local_k * 16 + local_n   # [16, 16]
        elem_idx = tl.load(perm_i_ptr + elem_flat).to(tl.int32)  # [16, 16]

        # Decode indices (K_BITS-specific)
        if K_BITS == 4:
            lane = elem_idx // 8
            r = elem_idx % 8
            word_low_idx = lane
            word_high_idx = (lane + 31) % 32
            shift = (7 - r) * 4
        elif K_BITS == 2:
            q16 = elem_idx // 16
            i1 = q16
            i0 = (i1 + 15) % 16
            r = elem_idx % 8
            shift0 = ((~(elem_idx // 8 * 8)) & 8) * 2
            word_low_idx = i1
            word_high_idx = i0
            shift = shift0 + (7 - r) * 2
        elif K_BITS == 1:
            q32 = elem_idx // 32
            i1 = q32
            i0 = (i1 + 7) % 8
            r = elem_idx % 8
            shift0 = (~(elem_idx // 8 * 8)) & 24
            word_low_idx = i1
            word_high_idx = i0
            shift = shift0 + (7 - r)
        elif K_BITS == 3:
            # dq8<bits=3>
            t_offset = elem_idx // 8 * 8
            r = elem_idx % 8
            b1 = (t_offset + 257) * K_BITS
            b0 = b1 - 16
            b2 = b1 + K_BITS * 7
            i0 = b0 // 32
            i2 = (b2 - 1) // 32
            s2 = (i2 + 1) * 32 - b2
            word_low_idx = i2 % (K_BITS * 256 // 32)
            word_high_idx = i0 % (K_BITS * 256 // 32)
            shift = s2 + (7 - r) * K_BITS
        else:
            # Generic for K in {5,6,8}: dq4-style extraction
            t = (elem_idx // 4) * 4
            j = elem_idx % 4
            b0 = (t + 257) * K_BITS - 16
            b2 = (t + 260) * K_BITS
            i0 = b0 // 32
            i2 = (b2 - 1) // 32
            s2 = (i2 + 1) * 32 - b2
            word_low_idx = i2 % (K_BITS * 256 // 32)
            word_high_idx = i0 % (K_BITS * 256 // 32)
            shift = s2 + (3 - j) * K_BITS

        # Gather uint32 words using computed indices.
        # trellis is int16; promote to int64 and mask to uint16 *before*
        # combining, so the high bit of a u16 half never sign-extends into
        # the 64-bit funnel-shift value below.
        word_base = tile_base
        low_u16_0 = tl.load(trellis_ptr + word_base + word_low_idx * 2).to(tl.int64) & 0xFFFF
        low_u16_1 = tl.load(trellis_ptr + word_base + word_low_idx * 2 + 1).to(tl.int64) & 0xFFFF
        low_u32 = (low_u16_0 | (low_u16_1 << 16)) & 0xFFFFFFFF

        high_u16_0 = tl.load(trellis_ptr + word_base + word_high_idx * 2).to(tl.int64) & 0xFFFF
        high_u16_1 = tl.load(trellis_ptr + word_base + word_high_idx * 2 + 1).to(tl.int64) & 0xFFFF
        high_u32 = (high_u16_0 | (high_u16_1 << 16)) & 0xFFFFFFFF

        # Funnel shift and mask
        combined = (high_u32 << 32) | low_u32
        windows = (combined >> shift) & 0xFFFF

        # LUT lookup
        w_tile = tl.load(lut_ptr + windows)  # [16, 16] fp16

        # Dot product
        acc += tl.dot(x_block, w_tile)

    # Store output
    mask_n = offs_n < N
    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


_wrapped_fused_kernel = torch.library.wrap_triton(_fused_dequant_gemm_kernel)


@torch.library.triton_op("exl3::exl3_gemm_triton", mutates_args=("y",))
def exl3_gemm_triton(
    x: torch.Tensor,
    trellis: torch.Tensor,
    y: torch.Tensor,
    lut: torch.Tensor,
    perm_i: torch.Tensor,
    K_bits: int,
    tiles_n: int,
) -> None:
    """Fused EXL3 dequant + fp16 matmul. Does NOT materialize the weight matrix."""
    M, K_dim = x.shape
    N = y.shape[1]
    packed_size = trellis.shape[-1]

    BLOCK_M = min(64, M) if M > 16 else 16
    grid = (triton.cdiv(M, BLOCK_M), tiles_n)

    _wrapped_fused_kernel[grid](
        x, y,
        trellis,
        lut,
        perm_i,
        M, N, K_dim,
        tiles_n,
        packed_size,
        packed_size // 2,  # n_u32
        x.stride(0), x.stride(1),
        trellis.stride(0), trellis.stride(1), trellis.stride(2),
        y.stride(0), y.stride(1),
        BLOCK_M=BLOCK_M,
        K_BITS=K_bits,
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
    """Fused EXL3 dequant + GEMM. Does NOT materialize the weight matrix.

    Dequantization happens tile-by-tile inside the Triton kernel.
    """
    original_shape = x.shape
    x_flat = x.view(-1, in_features)
    rows = x_flat.shape[0]

    x_half = x_flat if x_flat.dtype == torch.half else x_flat.to(torch.half)

    # Phase 1: Hadamard-transform input
    xh = torch.empty_like(x_half)
    torch.ops.exl3_ops.had_r_128(x_half, xh, suh, None, 1.0)

    # Phase 2+3: Fused dequant + Triton matmul (no weight matrix materialization)
    cb = 1 if mcg else (2 if mul1 else 0)
    lut = _decode_lut(cb, device)
    perm_i = _get_perm(device)
    tiles_n = trellis.shape[1]

    y = torch.empty((rows, out_features), dtype=out_dtype, device=device)
    torch.ops.exl3.exl3_gemm_triton(xh, trellis, y, lut, perm_i, K, tiles_n)

    # Phase 4: Hadamard-transform output
    torch.ops.exl3_ops.had_r_128(y, y, None, svh, 1.0)

    return y.view(original_shape[:-1] + (out_features,))
