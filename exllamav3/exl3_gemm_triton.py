"""Triton operators for EXL3 linear layers.

The ops live in the ``exl3_ops::`` namespace:

    exl3_ops::had_r_128_triton(x, y, suh)          # Triton: row Hadamard transform
    exl3_ops::exl3_gemm_triton(xh, t, y, ...)      # Triton: fused dequant + GEMM
    exl3_ops::LinearEXL3_triton(x, y, xh, ...)     # composition of the two, below

``LinearEXL3_triton`` is the complete EXL3 linear forward —
hadamard -> fused-dequant-gemm -> hadamard (+ optional bias) — writing into
caller-provided buffers so it composes with CUDA graph capture (see bc_rocm.py).
It depends only on Triton: every kernel it launches is a Triton kernel.
Callers outside this pipeline use the C++ kernel via the pybind
``ext.had_r_128`` (quant/hadamard.cu).
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Triton Hadamard transform (128-element rows, radix-2 butterfly)
#
# Mirrors the C++ had_hf/had_ff_r_128 kernels exactly: the transform is
# evaluated in fp32 regardless of I/O dtype (with a single round to half at
# the very end for half output), r_scale = scale / sqrt(128) is applied in
# fp32 after the transform, and pre/post scales are applied in the I/O
# dtype. The butterfly runs sequentially over masks 1..64, which reproduces
# the C++ expression tree (4-point transform in registers, then 32-lane
# xor-shuffles) term-for-term, so results are bit-identical.
# ---------------------------------------------------------------------------

@triton.jit
def _had_stage(v, BLOCK_R: tl.constexpr, SPAN: tl.constexpr):
    """One radix-2 butterfly stage over a [BLOCK_R, 128] fp32 tile.

    Elements whose bit log2(SPAN) is 0 receive a+b, those with the bit set
    receive a-b, where b is the partner element at distance SPAN.
    """
    G: tl.constexpr = 128 // (2 * SPAN)
    pair = tl.permute(v.reshape(BLOCK_R, G, 2, SPAN), (0, 1, 3, 2))
    lo, hi = tl.split(pair)
    pair = tl.join(lo + hi, lo - hi)
    return tl.permute(pair, (0, 1, 3, 2)).reshape(BLOCK_R, 128)


@triton.jit
def _had_r_128_kernel(
    x_ptr, y_ptr, scale_ptr,
    n_rows,
    stride_xr, stride_yr,
    r_scale,
    IO_FP32: tl.constexpr,
    PRE_SCALED: tl.constexpr,
    POST_SCALED: tl.constexpr,
    BLOCK_R: tl.constexpr,
):
    # One program transforms a [BLOCK_R, 128] tile: pid_m rows, pid_c the
    # 128-column block within each row. Scales are indexed by flat column
    # position (row-independent), matching the C++ kernel.
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_m * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = rows < n_rows
    col = tl.arange(0, 128)

    x = tl.load(
        x_ptr + rows[:, None] * stride_xr + (pid_c * 128 + col)[None, :],
        mask=mask_r[:, None], other=0.0,
    )

    # Pre-scale, applied in the I/O dtype (half multiply for the half path,
    # exactly like the C++ __hmul2 version)
    if PRE_SCALED:
        pre = tl.load(scale_ptr + pid_c * 128 + col)
        if IO_FP32:
            x = x * pre.to(tl.float32)
        else:
            x = x * pre

    # fp32 radix-2 butterfly over bit masks 1..64, unrolled via a constexpr
    # helper. Each stage pairs element j with j^span: the bit-0 element gets
    # a+b, the bit-1 element a-b — the same expression tree as the C++
    # register H4 + xor-shuffle network, so the result is bit-identical.
    v = x.to(tl.float32)
    v = _had_stage(v, BLOCK_R, 1)
    v = _had_stage(v, BLOCK_R, 2)
    v = _had_stage(v, BLOCK_R, 4)
    v = _had_stage(v, BLOCK_R, 8)
    v = _had_stage(v, BLOCK_R, 16)
    v = _had_stage(v, BLOCK_R, 32)
    v = _had_stage(v, BLOCK_R, 64)
    v = v * r_scale

    # Post-scale. The C++ half kernel rounds the scaled transform to half
    # first and then multiplies in half; reproduce that ordering exactly.
    if POST_SCALED:
        post = tl.load(scale_ptr + pid_c * 128 + col)
        if IO_FP32:
            out = v * post.to(tl.float32)
        else:
            out = v.to(x_ptr.dtype.element_ty) * post
    else:
        out = v

    tl.store(
        y_ptr + rows[:, None] * stride_yr + (pid_c * 128 + col)[None, :],
        out.to(y_ptr.dtype.element_ty),
        mask=mask_r[:, None],
    )

# ---------------------------------------------------------------------------
# had_r_128_triton: Triton row Hadamard transform
#
# A Triton twin of the C++ pybind ``ext.had_r_128`` (quant/hadamard.cu),
# bit-identical by construction. Used ONLY inside LinearEXL3_triton so that
# the whole EXL3 linear path depends on Triton alone; every other caller
# keeps using the C++ kernel through pybind.
#
# Scope note: its kernel time matches or beats the C++ kernel at every shape
# (measured via CUDA-graph replay), but an EAGER call from Python pays the
# torch.ops dispatch + Triton launch path — roughly 2-3x the pybind call's
# cost. It exists for the graph-captured path, where launch overhead does not
# apply; don't adopt it on uncaptured hot paths.
# ---------------------------------------------------------------------------

_RSCALE_128 = 0.088388347648  # 1/sqrt(128), matches the C++ literal

_LIB = torch.library.Library("exl3_ops", "FRAGMENT")
_LIB.define(
    "had_r_128_triton("
    "Tensor input, Tensor(a!) output, Tensor? pre_scale, "
    "Tensor? post_scale, float scale"
    ") -> ()"
)


@torch.library.register_fake("exl3_ops::had_r_128_triton")
def _had_r_128_triton_fake(input, output, pre_scale, post_scale, scale):
    # output is mutated in-place; no new tensors are returned
    return


@torch.library.impl("exl3_ops::had_r_128_triton", "default")
def _had_r_128_triton_impl(
    input: torch.Tensor,
    output: torch.Tensor,
    pre_scale: torch.Tensor | None,
    post_scale: torch.Tensor | None,
    scale: float,
) -> None:
    """y = (x.view(-1, 128) @ H128) * (pre|post)_scale, scaled by scale/sqrt(128).

    Matches the C++ ``had_r_128`` contract: input/output must be 2D, the same
    dtype (half or float), contiguous in the last dim, with last dim a
    multiple of 128; scales are half tensors with one element per column
    (flat, row-independent). Pre-scale multiplies before the transform in the
    I/O dtype; post-scale multiplies after it (for half output, after the
    round to half), like the C++ kernels.
    """
    assert input.dtype == output.dtype, "had_r_128_triton: input/output dtype mismatch"
    assert input.dtype in (torch.half, torch.float), \
        f"had_r_128_triton: unsupported dtype {input.dtype}"
    assert input.dim() == 2 and input.shape[-1] % 128 == 0
    # The kernel indexes the last dim with an implicit unit stride
    assert input.stride(-1) == 1, \
        f"had_r_128_triton: input last dim must be contiguous, got stride {input.stride(-1)}"
    assert output.stride(-1) == 1, \
        f"had_r_128_triton: output last dim must be contiguous, got stride {output.stride(-1)}"
    assert (pre_scale is None) or (post_scale is None)
    rows, cols = input.shape

    # Tiling (swept on RDNA3 via graph-of-64 replay timing): BLOCK_R=4 with a
    # single warp is optimal or tied-for-optimal at every shape from rows==1
    # (decode) through rows==512+ (prefill). The 128-wide tile leaves extra
    # warps idle; wider row tiles only help shapes too small to matter.
    BLOCK_R = 4
    num_warps = 1

    grid = (triton.cdiv(rows, BLOCK_R), cols // 128)
    _had_r_128_kernel[grid](
        input, output,
        pre_scale if pre_scale is not None else post_scale,
        rows,
        input.stride(0), output.stride(0),
        scale * _RSCALE_128,
        IO_FP32=input.dtype == torch.float,
        PRE_SCALED=pre_scale is not None,
        POST_SCALED=post_scale is not None,
        BLOCK_R=BLOCK_R,
        num_warps=num_warps,
    )


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
#
# Each block computes a [BLOCK_M, BLOCK_N] output tile. The weight tile
# [BLOCK_K, BLOCK_N] is decoded on-the-fly from the compressed trellis every
# K-iteration (no full weight matrix is ever materialized).
#
# The dequant is fully vectorized across the whole [BLOCK_K, BLOCK_N] weight
# tile: per-element (word_low_idx, word_high_idx, shift) are computed from the
# inverse-permuted element index, then the uint32 trellis words are gathered
# and funnel-shifted to extract a 16-bit codebook index. The codebook index is
# decoded with inline arithmetic (matching decode_3inst in the C++ ext) rather
# than a LUT, so the whole path is register/ALU bound with no table gathers.
# ---------------------------------------------------------------------------


def _exl3_gemm_configs():
    import os
    cfg_spec = os.environ.get("EXL3_GEMM_CONFIGS")
    if cfg_spec:
        # Format: "BM,BN,BK,GM:nw:ns;..." for quick manual sweeps.
        configs = []
        for part in cfg_spec.split(";"):
            part = part.strip()
            if not part:
                continue
            dims, _, rest = part.partition(":")
            bm, bn, bk, gm = (int(x) for x in dims.split(","))
            nw = int(rest.split(":")[0]) if rest else 4
            ns = int(rest.split(":")[1]) if ":" in rest else 3
            configs.append(triton.Config({"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk, "GROUP_M": gm}, num_warps=nw, num_stages=ns))
        return configs
    # Default autotune set. For the memory-bound decode path (M=1) BLOCK_N=16
    # (one trellis n-tile per block) dominates; larger N triggers costly 2D
    # trellis gathers. BLOCK_K=64 with num_stages=2 is the sweet spot.
    return [
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32, "GROUP_M": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 32, "GROUP_M": 1}, num_warps=2, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=4, num_stages=4),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 128, "GROUP_M": 1}, num_warps=8, num_stages=3),
        triton.Config({"BLOCK_M": 16, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=2, num_stages=3),
        # Larger BLOCK_M for prefill (M>1)
        triton.Config({"BLOCK_M": 32, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=4, num_stages=2),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 16, "BLOCK_K": 64, "GROUP_M": 1}, num_warps=8, num_stages=2),
    ]


@triton.autotune(configs=_exl3_gemm_configs(), key=["M", "N", "K_dim", "K_BITS", "N_PACKED", "CB"])
@triton.jit
def _fused_dequant_gemm_kernel(
    x_ptr, y_ptr,
    trellis_ptr,
    perm_i_ptr,
    M, N, K_dim,
    stride_xm, stride_xk,
    stride_tk, stride_tn,
    stride_ym, stride_yn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
    K_BITS: tl.constexpr,
    N_PACKED: tl.constexpr,
    CB: tl.constexpr,
):
    NK: tl.constexpr = BLOCK_K // 16   # k-sub-tiles per weight tile
    NN: tl.constexpr = BLOCK_N // 16   # n-sub-tiles per weight tile
    N_U32: tl.constexpr = K_BITS * 256 // 32
    # For K_BITS in {1,2,4} the funnel shift never exceeds 31, so the 64-bit
    # funnel (high<<32 | low) >> shift can be computed with 32-bit ops only,
    # avoiding expensive emulated 64-bit arithmetic on RDNA3.
    SHIFT_FITS_32: tl.constexpr = (K_BITS == 1) | (K_BITS == 2) | (K_BITS == 4)

    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_m = offs_m < M
    mask_n = offs_n < N

    # Per-element geometry for the [BLOCK_K, BLOCK_N] weight tile.
    # Within every 16x16 sub-tile the (local_k, local_n) pattern is identical,
    # so the element index / decode indices repeat with period 16 in both axes.
    k_idx = tl.arange(0, BLOCK_K)
    n_idx = tl.arange(0, BLOCK_N)
    local_k = (k_idx % 16)[:, None]            # [BLOCK_K, 1]
    local_n = (n_idx % 16)[None, :]            # [1, BLOCK_N]
    ki = (k_idx // 16)[:, None]                # [BLOCK_K, 1] which k-sub-tile
    nj = (n_idx // 16)[None, :]                # [1, BLOCK_N] which n-sub-tile

    elem_flat = local_k * 16 + local_n         # [BLOCK_K, BLOCK_N]
    elem_idx = tl.load(perm_i_ptr + elem_flat).to(tl.int32)

    # K_BITS-specific decode indices (vectorized over the whole tile).
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
        t_offset = elem_idx // 8 * 8
        r = elem_idx % 8
        b1 = (t_offset + 257) * K_BITS
        b0 = b1 - 16
        b2 = b1 + K_BITS * 7
        i0 = b0 // 32
        i2 = (b2 - 1) // 32
        s2 = (i2 + 1) * 32 - b2
        word_low_idx = i2 % N_U32
        word_high_idx = i0 % N_U32
        shift = s2 + (7 - r) * K_BITS
    else:
        t = (elem_idx // 4) * 4
        j = elem_idx % 4
        b0 = (t + 257) * K_BITS - 16
        b2 = (t + 260) * K_BITS
        i0 = b0 // 32
        i2 = (b2 - 1) // 32
        s2 = (i2 + 1) * 32 - b2
        word_low_idx = i2 % N_U32
        word_high_idx = i0 % N_U32
        shift = s2 + (3 - j) * K_BITS

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    n_k_tiles_total = K_dim // 16
    n_outer = tl.cdiv(n_k_tiles_total, NK)

    for k_outer in range(n_outer):
        k_tile_base = k_outer * NK
        k_offset = k_tile_base * 16 + k_idx     # [BLOCK_K]

        # Load x tile [BLOCK_M, BLOCK_K]
        x_block = tl.load(
            x_ptr + offs_m[:, None] * stride_xm + k_offset[None, :] * stride_xk,
            mask=mask_m[:, None],
            other=0.0,
        )

        # Trellis base address per weight element: depends on (k-sub-tile, n-sub-tile).
        # word index is in uint32 units; trellis is int16 so multiply by 2 for offset.
        tile_base = (k_tile_base + ki) * stride_tk + (pid_n * NN + nj) * stride_tn

        # Gather uint32 words. Cast int16->int64/uint32 and mask to 0xFFFF
        # *before* combining so a high u16 bit never sign-extends.
        low_off = tile_base + word_low_idx * 2
        high_off = tile_base + word_high_idx * 2

        if SHIFT_FITS_32:
            # 32-bit funnel: shift is guaranteed in [0,31] for K_BITS in {1,2,4}.
            low_u16_0 = tl.load(trellis_ptr + low_off).to(tl.uint32) & 0xFFFF
            low_u16_1 = tl.load(trellis_ptr + low_off + 1).to(tl.uint32) & 0xFFFF
            low_u32 = low_u16_0 | (low_u16_1 << 16)

            high_u16_0 = tl.load(trellis_ptr + high_off).to(tl.uint32) & 0xFFFF
            high_u16_1 = tl.load(trellis_ptr + high_off + 1).to(tl.uint32) & 0xFFFF
            high_u32 = high_u16_0 | (high_u16_1 << 16)

            neg_shift = tl.minimum(32 - shift, 31)
            windows = ((low_u32 >> shift) | (high_u32 << neg_shift)) & 0xFFFF
        else:
            low_u16_0 = tl.load(trellis_ptr + low_off).to(tl.int64) & 0xFFFF
            low_u16_1 = tl.load(trellis_ptr + low_off + 1).to(tl.int64) & 0xFFFF
            low_u32 = (low_u16_0 | (low_u16_1 << 16)) & 0xFFFFFFFF

            high_u16_0 = tl.load(trellis_ptr + high_off).to(tl.int64) & 0xFFFF
            high_u16_1 = tl.load(trellis_ptr + high_off + 1).to(tl.int64) & 0xFFFF
            high_u32 = (high_u16_0 | (high_u16_1 << 16)) & 0xFFFFFFFF

            combined = (high_u32 << 32) | low_u32
            windows = (combined >> shift) & 0xFFFF

        # Inline arithmetic decode (matches decode_3inst in the C++ reference).
        # This replaces a 65536-entry LUT gather with ~3 cheap ALU ops, which is
        # the dominant speedup for the memory-bound decode path.
        w_u32 = windows.to(tl.uint32)
        if CB == 0:
            w_u32 = (w_u32 * 89226354 + 64248484) & 0xFFFFFFFF
            w_u32 = 0x3b603b60 ^ (w_u32 & 0x8fff8fff)
        elif CB == 1:
            w_u32 = (w_u32 * 0xCBAC1FED) & 0xFFFFFFFF
            w_u32 = 0x3b603b60 ^ (w_u32 & 0x8fff8fff)
        else:  # CB == 2 (mul1)
            w_u32 = (w_u32 * 0x83DCD12D) & 0xFFFFFFFF
            # byte sum: dp4a(x, 0x01010101, 0x6400) emulated
            db0 = w_u32 & 0xFF
            db1 = (w_u32 >> 8) & 0xFF
            db2 = (w_u32 >> 16) & 0xFF
            db3 = (w_u32 >> 24) & 0xFF
            s = (db0 + db1 + db2 + db3 + 0x6400) & 0xFFFF
            w_u32 = s

        # bitcast low/high 16 bits to fp16 then add (cb 0/1), or fma (cb 2)
        if CB == 0 or CB == 1:
            lo = w_u32 & 0xFFFF
            hi = (w_u32 >> 16) & 0xFFFF
            lo_h = tl.cast(lo.to(tl.int16), tl.float16, bitcast=True)
            hi_h = tl.cast(hi.to(tl.int16), tl.float16, bitcast=True)
            w_block = lo_h + hi_h
        else:
            sum16 = w_u32 & 0xFFFF
            h = tl.cast(sum16.to(tl.int16), tl.float16, bitcast=True)
            k_inv_h = tl.full((1,), 0x1eee, dtype=tl.int16)
            k_inv_h = tl.cast(k_inv_h, tl.float16, bitcast=True)
            k_bias_h = tl.full((1,), 0xc931, dtype=tl.int16)
            k_bias_h = tl.cast(k_bias_h, tl.float16, bitcast=True)
            w_block = h * k_inv_h + k_bias_h

        acc = tl.dot(x_block, w_block, acc)

    tl.store(
        y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :] * stride_yn,
        acc.to(y_ptr.dtype.element_ty),
        mask=mask_m[:, None] & mask_n[None, :],
    )


_wrapped_fused_kernel = torch.library.wrap_triton(_fused_dequant_gemm_kernel)


@torch.library.triton_op("exl3_ops::exl3_gemm_triton", mutates_args=("y",))
def exl3_gemm_triton(
    x: torch.Tensor,
    trellis: torch.Tensor,
    y: torch.Tensor,
    lut: torch.Tensor,
    perm_i: torch.Tensor,
    K_bits: int,
    tiles_n: int,
    cb: int = 0,
) -> None:
    """Fused EXL3 dequant + fp16 matmul. Does NOT materialize the weight matrix."""
    M, K_dim = x.shape
    N = y.shape[1]

    grid = lambda meta: (
        triton.cdiv(M, meta["BLOCK_M"]) * triton.cdiv(N, meta["BLOCK_N"]),
    )

    _wrapped_fused_kernel[grid](
        x, y,
        trellis,
        perm_i,
        M, N, K_dim,
        x.stride(0), x.stride(1),
        trellis.stride(0), trellis.stride(1),
        y.stride(0), y.stride(1),
        K_BITS=K_bits,
        N_PACKED=trellis.shape[-1],
        CB=cb,
    )


# ---------------------------------------------------------------------------
# LinearEXL3 composition op: had_r_128_triton -> gemm -> had_r_128_triton
# ---------------------------------------------------------------------------

_LIB.define(
    "LinearEXL3_triton("
    "Tensor x, Tensor(a!) y, Tensor(b!) xh, "
    "Tensor trellis, Tensor suh, Tensor svh, "
    "int K, bool mcg, bool mul1, Tensor? bias, "
    "int in_features, int out_features"
    ") -> ()"
)


@torch.library.register_fake("exl3_ops::LinearEXL3_triton")
def _linear_exl3_triton_fake(
    x, y, xh, trellis, suh, svh, K, mcg, mul1, bias, in_features, out_features
):
    # y and xh are mutated in-place; no new tensors are returned
    return


@torch.library.impl("exl3_ops::LinearEXL3_triton", "default")
def _linear_exl3_triton_impl(
    x: torch.Tensor,
    y: torch.Tensor,
    xh: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    K: int,
    mcg: bool,
    mul1: bool,
    bias: torch.Tensor | None,
    in_features: int,
    out_features: int,
) -> None:
    """Complete EXL3 linear forward into pre-allocated buffers.

    Writes the Hadamard-transformed input to ``xh`` and the result to ``y``.
    All tensors must be pre-allocated with stable addresses for CUDA graph
    capture; nothing is allocated inside this op.
    """
    # A cast here would allocate and silently break a capturing graph; callers
    # that need dtype conversion must cast before invoking the op.
    assert x.dtype == torch.half, f"LinearEXL3_triton: expected half input, got {x.dtype}"

    # Phase 1: input Hadamard transform -> xh
    torch.ops.exl3_ops.had_r_128_triton(x, xh, suh, None, 1.0)

    # Phase 2 + 3: fused dequant + Triton GEMM -> y
    cb = 1 if mcg else (2 if mul1 else 0)
    torch.ops.exl3_ops.exl3_gemm_triton(
        xh, trellis, y,
        _decode_lut(cb, x.device), _get_perm(x.device),
        K, trellis.shape[1], cb,
    )

    # Phase 4: output Hadamard transform (in place)
    torch.ops.exl3_ops.had_r_128_triton(y, y, None, svh, 1.0)

    if bias is not None:
        y.add_(bias)


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
    bias: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused EXL3 dequant + GEMM, allocating and returning the output.

    Thin convenience wrapper over ``exl3_ops::LinearEXL3_triton``. The BC
    (graph-capture) path calls the op directly to keep tensor addresses
    stable across replays.

    The per-call ``xh`` workspace allocation is deliberate and cheap: the
    caching allocator serves it from its free list (no cudaMalloc, no sync)
    after the first few calls, and this wrapper is off the decode hot path —
    prefill is dominated by the GEMM itself.
    """
    original_shape = x.shape
    x_flat = x.view(-1, in_features)
    rows = x_flat.shape[0]

    x_half = x_flat if x_flat.dtype == torch.half else x_flat.to(torch.half)

    y = torch.empty((rows, out_features), dtype=out_dtype, device=device)
    xh = torch.empty_like(x_half)

    torch.ops.exl3_ops.LinearEXL3_triton(
        x_half, y, xh, trellis, suh, svh, K, mcg, mul1, bias,
        in_features, out_features,
    )

    return y.view(original_shape[:-1] + (out_features,))
