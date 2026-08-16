"""Fused QKV projection kernels for the attention decode path (ROCm only).

Per-layer decode cost of the three separate EXL3 projections (q N=12288,
k N=1024, v N=1024, all K=5120, bits=4) is dominated by two effects on
gfx1100:

  * the narrow N=1024 GEMVs launch only 8-16 CTAs and run latency-bound at
    ~4-8 GB/s, and
  * each projection costs 5 GPU nodes on the ROCm BC path (input copy, input
    Hadamard, GEMV, output Hadamard, output clone), and whole-step graph
    execution pays a per-node frontend cost, so the projections alone were
    ~2.9 ms/token across the 16 full-attention layers.

This module computes all three projections in ONE launch (plus one shared
input-Hadamard launch): the k/v tiles ride along in the same wave as q's 96
CTAs, and the output Hadamard of every 128-column tile is applied in-register
in the GEMV epilogue (BLOCK_N=128 tiles align exactly with the 128-wide
Hadamard blocks).

Arithmetic is replicated element-for-element from the reference kernels in
exl3_gemm_triton.py (_had_r_128_kernel and the K_BITS==4/M1 fast path of
_fused_dequant_gemm_kernel), so every output element is bit-identical to the
unfused path:

  * input Hadamard: half multiply by the sign vector, fp32 radix-2 butterfly,
    * 1/sqrt(128), store half,
  * GEMV: same K-tile loop order, same elementwise fp32 accumulation and the
    same final axis-reduction order (per-element result is independent of
    BLOCK_N),
  * output Hadamard: the GEMV accumulator is rounded to fp16 exactly as the
    intermediate y buffer would be, then butterfly in fp32, scale, round to
    fp16, and multiply by the half sign vector.

Plain stream launches, so the whole-step CUDA graph (block_graph_rocm.py)
captures them like any other op. Gated on torch.version.hip in attn.py with
kill switch EXL3_ATTN_QKV_FUSED=0. CUDA behavior is unchanged.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from .modules.quant.exl3_triton import _had_stage, _decode_u16

_RSCALE_128 = tl.constexpr(0.088388347648)  # 1/sqrt(128), matches the reference kernel


@triton.jit
def _had3_r_128_kernel(
    x_ptr,
    s0_ptr, y0_ptr,
    s1_ptr, y1_ptr,
    s2_ptr, y2_ptr,
    n_rows,
    stride_xr, stride_y0r, stride_y1r, stride_y2r,
    BLOCK_R: tl.constexpr,
):
    """Three pre-scaled 128-point row Hadamard transforms of the same input row
    (three sign vectors), one launch. Bit-identical per element to
    _had_r_128_kernel(PRE_SCALED=True, scale=1.0)."""
    pid_m = tl.program_id(0)
    pid_c = tl.program_id(1)
    rows = pid_m * BLOCK_R + tl.arange(0, BLOCK_R)
    mask_r = rows < n_rows
    col = tl.arange(0, 128)

    x = tl.load(
        x_ptr + rows[:, None] * stride_xr + (pid_c * 128 + col)[None, :],
        mask=mask_r[:, None], other=0.0,
    )
    x0 = x * tl.load(s0_ptr + pid_c * 128 + col)
    x1 = x * tl.load(s1_ptr + pid_c * 128 + col)
    x2 = x * tl.load(s2_ptr + pid_c * 128 + col)

    v0 = x0.to(tl.float32)
    v1 = x1.to(tl.float32)
    v2 = x2.to(tl.float32)
    for span in tl.static_range(7):
        v0 = _had_stage(v0, BLOCK_R, 1 << span)
        v1 = _had_stage(v1, BLOCK_R, 1 << span)
        v2 = _had_stage(v2, BLOCK_R, 1 << span)
    v0 = v0 * _RSCALE_128
    v1 = v1 * _RSCALE_128
    v2 = v2 * _RSCALE_128

    tl.store(y0_ptr + rows[:, None] * stride_y0r + (pid_c * 128 + col)[None, :],
             v0.to(y0_ptr.dtype.element_ty), mask=mask_r[:, None])
    tl.store(y1_ptr + rows[:, None] * stride_y1r + (pid_c * 128 + col)[None, :],
             v1.to(y1_ptr.dtype.element_ty), mask=mask_r[:, None])
    tl.store(y2_ptr + rows[:, None] * stride_y2r + (pid_c * 128 + col)[None, :],
             v2.to(y2_ptr.dtype.element_ty), mask=mask_r[:, None])


@triton.jit
def _gemv3_hadout_segment(
    xh_ptr,                    # [K] half, Hadamard-transformed input
    tu32_ptr,                  # trellis u32 words, [k_tile, n_subtile, word]
    svh_ptr,                   # [N] half output sign vector
    y_ptr,                     # [N] half output (q half when QG_INTERLEAVED)
    pid_n,
    n_total,                   # total columns N of this segment
    stride_tk_u32,
    stride_tn_u32,
    K_dim,
    head_dim,
    g_ptr,                     # [N//2] half gate output when QG_INTERLEAVED
    CB: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QG_INTERLEAVED: tl.constexpr,
):
    """One BLOCK_N-wide M1 GEMV segment (bits=4 fast path) with the output
    Hadamard fused into the epilogue. Per-element arithmetic matches the
    reference kernel; BLOCK_N=128 tiles align with the Hadamard blocks."""
    NK: tl.constexpr = BLOCK_K // 16
    NN: tl.constexpr = BLOCK_N // 16
    tu32_ptr = tu32_ptr.to(tl.pointer_type(tl.uint32))
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask_n = offs_n < n_total

    j8 = tl.arange(0, 8)
    sh = 28 - 4 * j8
    neg_sh = tl.minimum(32 - sh, 31)
    wc = tl.arange(0, NN * 32)
    nj8 = tl.arange(0, NN)
    r16 = tl.arange(0, 16)
    base_n = (pid_n * NN) * stride_tn_u32

    n_k_tiles_total = K_dim // 16
    n_outer = tl.cdiv(n_k_tiles_total, NK)

    acc6 = tl.zeros((2, 2, 2, NN, 8, 4), dtype=tl.float32)
    for k_outer in range(n_outer):
        for ki in tl.static_range(NK):
            ktb = k_outer * NK + ki
            # [k_tile, n_subtile, words]: the NN*32 staged words of one k-tile
            # are contiguous; the m1 neighbor of word 0 belongs to the same
            # subtile (word 31), so the wrap is fixed in registers
            row = ktb * stride_tk_u32 + base_n
            words = tl.load(tu32_ptr + row + wc)
            safe = (ktb > 0) | (base_n > 0)
            m1_lin = tl.load(tu32_ptr + row + wc - 1, mask=safe | (wc > 0), other=0)
            w31 = tl.load(tu32_ptr + row + nj8 * 32 + 31)
            w31_bcast = tl.reshape(
                tl.broadcast_to(w31[:, None], (NN, 32)), (NN * 32,)
            )
            m1 = tl.where((wc % 32) == 0, w31_bcast, m1_lin)
            q = ((words[None, :] >> sh[:, None]) |
                 (m1[None, :] << neg_sh[:, None])) & 0xFFFF
            w_dec = _decode_u16(q.to(tl.uint32), CB).to(tl.float32)
            xk = tl.load(xh_ptr + (ktb * 16 + r16)).to(tl.float32)
            xpat = tl.permute(tl.reshape(xk, (2, 4, 2)), (0, 2, 1))
            xb6 = tl.broadcast_to(
                tl.reshape(xpat, (1, 2, 2, 1, 1, 4)), (2, 2, 2, NN, 8, 4)
            )
            acc6 += tl.reshape(w_dec, (2, 2, 2, NN, 8, 4)) * xb6
    s = tl.sum(acc6, 5)
    s = tl.sum(s, 2)
    s = tl.sum(s, 1)
    acc = tl.reshape(tl.permute(s, (1, 0, 2)), (BLOCK_N,))

    # (BLOCK_N // 128) independent Hadamard blocks; _had_stage's BLOCK_R axis
    # carries them so any BLOCK_N that is a multiple of 128 works.
    HR: tl.constexpr = BLOCK_N // 128
    v = tl.reshape(acc.to(tl.float16).to(tl.float32), (HR, 128))
    for span in tl.static_range(7):
        v = _had_stage(v, HR, 1 << span)
    v = v * _RSCALE_128
    post = tl.load(svh_ptr + offs_n)
    out = tl.reshape(v.to(tl.float16) * post, (BLOCK_N,))

    if QG_INTERLEAVED:
        # q_proj emits (q, g) interleaved in head_dim*2 blocks; with
        # head_dim % BLOCK_N == 0 a tile lies wholly inside one half of one
        # head, so the deinterleave is a per-CTA-uniform index remap
        # (bit-identical to ext.deinterleave_qg)
        n_global = pid_n * BLOCK_N
        half_index = n_global // head_dim
        head = half_index // 2
        dst = head * head_dim + (n_global % head_dim) + tl.arange(0, BLOCK_N)
        if (half_index % 2) == 0:
            tl.store(y_ptr + dst, out, mask=mask_n)
        else:
            tl.store(g_ptr + dst, out, mask=mask_n)
    else:
        tl.store(y_ptr + offs_n, out, mask=mask_n)


@triton.jit
def _gemv3_kernel(
    xq_ptr, tq_ptr, svhq_ptr, yq_ptr,
    xk_ptr, tk_ptr, svhk_ptr, yk_ptr,
    xv_ptr, tv_ptr, svhv_ptr, yv_ptr,
    NQ, NKV,
    stride_tkq_u32, stride_tnq_u32,
    stride_tkk_u32, stride_tnk_u32,
    stride_tkv_u32, stride_tnv_u32,
    K_dim,
    head_dim,
    g_ptr,
    CB: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    QG_INTERLEAVED: tl.constexpr,
):
    """q/k/v M1 GEMV in one launch. All segments share K and BLOCK_N=128
    (every N here is a multiple of 128); the k/v tiles fill the wave alongside
    q's CTAs instead of launching starved 8-CTA kernels."""
    pid = tl.program_id(0)
    tq = NQ // BLOCK_N
    tkv = NKV // BLOCK_N
    if pid < tq:
        _gemv3_hadout_segment(
            xq_ptr, tq_ptr, svhq_ptr, yq_ptr, pid, NQ,
            stride_tkq_u32, stride_tnq_u32, K_dim, head_dim, g_ptr,
            CB, BLOCK_N, BLOCK_K, QG_INTERLEAVED)
    elif pid < tq + tkv:
        _gemv3_hadout_segment(
            xk_ptr, tk_ptr, svhk_ptr, yk_ptr, pid - tq, NKV,
            stride_tkk_u32, stride_tnk_u32, K_dim, head_dim, g_ptr,
            CB, BLOCK_N, BLOCK_K, False)
    else:
        _gemv3_hadout_segment(
            xv_ptr, tv_ptr, svhv_ptr, yv_ptr, pid - tq - tkv, NKV,
            stride_tkv_u32, stride_tnv_u32, K_dim, head_dim, g_ptr,
            CB, BLOCK_N, BLOCK_K, False)


def fused_qkv_had(
    x: torch.Tensor,           # [1, 1, K] half contiguous (decode row)
    suh_q, suh_k, suh_v,       # [K] half sign vectors
    out: list[torch.Tensor],   # three [K] half buffers
):
    K = x.shape[-1]
    _had3_r_128_kernel[(1, K // 128)](
        x, suh_q, out[0], suh_k, out[1], suh_v, out[2],
        1, K, K, K, K,
        BLOCK_R=1,
        num_warps=1,
    )


def fused_qkv_gemv(
    xh_q, xh_k, xh_v,          # [K] half
    t_q, t_k, t_v,             # int16 trellises [K//16, N//16, 64] (or transposed)
    svh_q, svh_k, svh_v,       # [N] half sign vectors
    y_q, y_k, y_v,             # [N] half outputs (q segment holds interleaved q|g
                               # unless qg_interleaved, then y_q/g_out hold the
                               # deinterleaved halves)
    cb: int,
    g_out: torch.Tensor | None = None,   # [N//2] half gate output
    qg_interleaved: bool = False,
    head_dim: int = 256,
    block_n: int = 128,
    block_k: int = 128,
    num_warps: int = 8,
    num_stages: int = 3,
):
    nq = y_q.shape[-1] if not qg_interleaved else 2 * g_out.shape[-1]
    nkv = y_k.shape[-1]
    grid = (nq // block_n + 2 * (nkv // block_n),)
    stride_tk = (t_q.stride(0) // 2, t_k.stride(0) // 2, t_v.stride(0) // 2)
    stride_tn = (t_q.stride(1) // 2, t_k.stride(1) // 2, t_v.stride(1) // 2)
    _gemv3_kernel[grid](
        xh_q, t_q.view(torch.int32), svh_q, y_q,
        xh_k, t_k.view(torch.int32), svh_k, y_k,
        xh_v, t_v.view(torch.int32), svh_v, y_v,
        nq, nkv,
        stride_tk[0], stride_tn[0],
        stride_tk[1], stride_tn[1],
        stride_tk[2], stride_tn[2],
        xh_q.shape[-1],
        head_dim,
        g_out if g_out is not None else y_q,
        CB=cb,
        BLOCK_N=block_n,
        BLOCK_K=block_k,
        QG_INTERLEAVED=qg_interleaved,
        num_warps=num_warps,
        num_stages=num_stages,
    )
