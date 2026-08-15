"""EXL3 prefill via fast dequant-to-dense + FlyDSL RDNA3 WMMA GEMM.

Opt-in alternative to the fused Triton GEMM / reconstruct+hgemm paths for
the M > 1 (prefill) case, enabled with ``EXL3_PREFILL_FLY=1`` (default OFF).

    y = had_r_128(dequant(trellis) @ had_r_128(x))

Structure (three launches + two hadamards):

    1. ``had_r_128_triton(x) -> xh``                       (existing op)
    2. ``_dequant_dense_kernel``: trellis -> W_T[N, K] fp16 (NEW Triton kernel,
       this file — gathers-free algebraic decode identical to the fused kernel
       in exl3_gemm_triton.py, but writing the decoded tile in [n, k] layout
       instead of feeding tl.dot)
    3. FlyDSL WMMA GEMM C[M,N] = xh[M,K] @ W_T[N,K]^T       (vendored
       kernels/_flydsl_kernels/rdna3_f16_gemm.py, Apache-2.0, from
       github.com/ROCm/FlyDSL)
    4. ``had_r_128_triton(y, svh)``                        (existing op)

The [N, K] (B-transposed) dense layout is exactly what the WMMA kernel
consumes, and it is free to produce during decode: the (r, c) -> (j, t)
bijection of the trellis subtile permutation factors into per-axis index
bits, so ordering the decode tile's axes (n-major, k-minor) lands every
weight directly at W_T[n, k] with no transpose pass.

Only the bits=4 and bits=6 gather-free decode paths (128-divisible N and K)
are implemented here; other bit widths / shapes fall back to the caller's
existing path.
"""
from __future__ import annotations

import os
from collections import OrderedDict

import torch
import triton
import triton.language as tl

# ---------------------------------------------------------------------------
# Availability of the FlyDSL GEMM
# ---------------------------------------------------------------------------

try:
    from ._flydsl_kernels.rdna3_f16_gemm import create_wmma_gemm_module as _create_wmma
    _FLY_IMPORTABLE = True
except Exception:  # pragma: no cover
    _create_wmma = None
    _FLY_IMPORTABLE = False

_FLY_FAILED = False
_GEMM_CACHE: dict = {}


def prefill_fly_available() -> bool:
    """True when the FlyDSL prefill path is enabled (EXL3_PREFILL_FLY=1)."""
    global _FLY_FAILED
    if _FLY_FAILED or not _FLY_IMPORTABLE:
        return False
    return os.environ.get("EXL3_PREFILL_FLY", "0") != "0"


# ---------------------------------------------------------------------------
# Dequant-to-dense Triton kernel (W_T[N, K] fp16 out)
# ---------------------------------------------------------------------------

@triton.jit
def _decode_u16(w_u32, CB: tl.constexpr):
    """Inline arithmetic decode of 16-bit codebook indices (decode_3inst)."""
    if CB == 0:
        w_u32 = (w_u32 * 89226354 + 64248484) & 0xFFFFFFFF
        w_u32 = 0x3b603b60 ^ (w_u32 & 0x8fff8fff)
    elif CB == 1:
        w_u32 = (w_u32 * 0xCBAC1FED) & 0xFFFFFFFF
        w_u32 = 0x3b603b60 ^ (w_u32 & 0x8fff8fff)
    else:
        w_u32 = (w_u32 * 0x83DCD12D) & 0xFFFFFFFF
        db0 = w_u32 & 0xFF
        db1 = (w_u32 >> 8) & 0xFF
        db2 = (w_u32 >> 16) & 0xFF
        db3 = (w_u32 >> 24) & 0xFF
        w_u32 = (db0 + db1 + db2 + db3 + 0x6400) & 0xFFFF
    if CB == 0 or CB == 1:
        lo = w_u32 & 0xFFFF
        hi = (w_u32 >> 16) & 0xFFFF
        lo_h = tl.cast(lo.to(tl.int16), tl.float16, bitcast=True)
        hi_h = tl.cast(hi.to(tl.int16), tl.float16, bitcast=True)
        return lo_h + hi_h
    else:
        sum16 = w_u32 & 0xFFFF
        h = tl.cast(sum16.to(tl.int16), tl.float16, bitcast=True)
        k_inv_h = tl.cast(tl.full((1,), 0x1eee, tl.int16), tl.float16, bitcast=True)
        k_bias_h = tl.cast(tl.full((1,), 0xc931, tl.int16), tl.float16, bitcast=True)
        return h * k_inv_h + k_bias_h


@triton.jit
def _funnel6(lo, hi, s):
    """bits=6 funnel: 16-bit code window from a (lo, hi) u32 word pair where
    hi is the word *preceding* lo in the tile's virtual bit stream."""
    sel = s >= 32
    s32 = s & 31
    ns = tl.minimum(32 - s32, 31)
    base = tl.where(sel[:, None, None], hi[None, :, :], lo[None, :, :])
    second = tl.where(sel[:, None, None], lo[None, :, :], hi[None, :, :])
    return ((base >> s32[:, None, None]) | (second << ns[:, None, None])) & 0xFFFF


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 32}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 64}, num_warps=4, num_stages=3),
        triton.Config({"BLOCK_N": 128, "BLOCK_K": 128}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 256, "BLOCK_K": 64}, num_warps=8, num_stages=2),
        triton.Config({"BLOCK_N": 64, "BLOCK_K": 128}, num_warps=4, num_stages=3),
    ],
    key=["N", "K_dim", "K_BITS", "N_PACKED"],
)
@triton.jit
def _dequant_dense_kernel(
    trellis_ptr,
    w_ptr,                  # W_T[N, K] fp16 out
    N, K_dim,
    stride_tk, stride_tn,
    stride_wn,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    K_BITS: tl.constexpr,
    N_PACKED: tl.constexpr,
    CB: tl.constexpr,
):
    """Decode trellis -> W_T[n, k] (n-major, k contiguous).

    Same algebraic decode as the fused GEMM kernel's M>1 branch (verified
    against _dq_indices/_get_perm), but the decoded 16x(BLOCK_N) tile is
    permuted into [BLOCK_N, 16] n-major layout and stored directly:
      bits=4: decode axes (ch, rh, p, nj, cl, q) with
              n = 16*nj + 8*ch + cl, r = 8*rh + 2*q + p
      bits=6: decode axes (j1, j0, a0, nj, cA) x b1 x b0 with
              n = 16*nj + 8*b0 + cA, r = 8*j1 + 4*a0 + 2*b1 + j0
    """
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)

    NN: tl.constexpr = BLOCK_N // 16
    KT: tl.constexpr = BLOCK_K // 16

    tu32_ptr = trellis_ptr.to(tl.pointer_type(tl.uint32))
    stride_tk_u32 = stride_tk // 2
    stride_tn_u32 = stride_tn // 2
    base_n = (pid_n * NN) * stride_tn_u32

    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    if K_BITS == 4:
        j8 = tl.arange(0, 8)
        sh = 28 - 4 * j8
        neg_sh = tl.minimum(32 - sh, 31)
        wc = tl.arange(0, NN * 32)
        nj8 = tl.arange(0, NN)
        for ki in tl.static_range(KT):
            ktb = pid_k * KT + ki
            row = tu32_ptr + ktb * stride_tk_u32 + base_n
            words = tl.load(row + wc)
            safe = (ktb > 0) | (base_n > 0)
            m1_lin = tl.load(row + wc - 1, mask=safe | (wc > 0), other=0)
            w31 = tl.load(row + nj8 * 32 + 31)
            w31_bcast = tl.reshape(tl.broadcast_to(w31[:, None], (NN, 32)), (NN * 32,))
            m1 = tl.where((wc % 32) == 0, w31_bcast, m1_lin)
            q = ((words[None, :] >> sh[:, None]) |
                 (m1[None, :] << neg_sh[:, None])) & 0xFFFF
            w = _decode_u16(q.to(tl.uint32), CB)
            # (ch, rh, p, nj, cl, q) -> (nj, ch, cl, rh, q, p): n-major, k-minor
            w = tl.reshape(w, (2, 2, 2, NN, 8, 4))
            w = tl.permute(w, (3, 0, 4, 1, 5, 2))
            wt = tl.reshape(w, (BLOCK_N, 16))
            k_off = ktb * 16 + tl.arange(0, 16)
            tl.store(
                w_ptr + offs_n[:, None] * stride_wn + k_off[None, :],
                wt,
            )
    elif K_BITS == 6:
        a16 = tl.arange(0, 16)
        nj8 = tl.arange(0, NN)
        j8 = tl.arange(0, 4)
        wbase = tl.reshape(nj8[:, None] * 48 + 3 * a16[None, :], (NN * 16,))
        wone = tl.reshape(nj8[:, None] * 48 + (3 * a16[None, :] + 1) % 48, (NN * 16,))
        wtwo = tl.reshape(nj8[:, None] * 48 + (3 * a16[None, :] + 2) % 48, (NN * 16,))
        wneg = tl.reshape(nj8[:, None] * 48 + (3 * a16[None, :] + 47) % 48, (NN * 16,))
        C0 = tl.full((4,), 26, tl.int32); C1 = tl.full((4,), 34, tl.int32)
        C2 = tl.full((4,), 42, tl.int32); C3 = tl.full((4,), 18, tl.int32)
        sh6 = 6 * j8
        for ki in tl.static_range(KT):
            ktb = pid_k * KT + ki
            row = tu32_ptr + ktb * stride_tk_u32 + base_n
            words2 = tl.reshape(tl.load(row + wbase), (NN, 16))
            wone2 = tl.reshape(tl.load(row + wone), (NN, 16))
            wtwo2 = tl.reshape(tl.load(row + wtwo), (NN, 16))
            wneg2 = tl.reshape(tl.load(row + wneg), (NN, 16))
            d0 = _decode_u16(_funnel6(words2, wneg2, C0 - sh6), CB)
            d1 = _decode_u16(_funnel6(wone2, words2, C1 - sh6), CB)
            d2 = _decode_u16(_funnel6(wtwo2, wone2, C2 - sh6), CB)
            d3 = _decode_u16(_funnel6(wtwo2, wone2, C3 - sh6), CB)
            # (j1, j0, nj, cA, a0) -> (j1, j0, a0, nj, cA) per b, join b1, b0:
            # (j1, j0, a0, nj, cA, b1, b0) -> (nj, b0, cA, j1, a0, b1, j0)
            P0 = tl.permute(tl.reshape(d0, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
            P1 = tl.permute(tl.reshape(d1, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
            P2 = tl.permute(tl.reshape(d2, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
            P3 = tl.permute(tl.reshape(d3, (2, 2, NN, 8, 2)), (0, 1, 4, 2, 3))
            J0 = tl.join(P0, P2)
            J1 = tl.join(P1, P3)
            Wt = tl.join(J0, J1)                      # (j1, j0, a0, nj, cA, b1, b0)
            Wt = tl.permute(Wt, (3, 6, 4, 0, 2, 5, 1))
            wt = tl.reshape(Wt, (BLOCK_N, 16))
            k_off = ktb * 16 + tl.arange(0, 16)
            tl.store(
                w_ptr + offs_n[:, None] * stride_wn + k_off[None, :],
                wt,
            )


def dequant_dense_fly(
    trellis: torch.Tensor,
    bits: int,
    cb: int,
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Dequantize trellis [K/16, N/16, packed] to dense W_T[N, K] fp16 (n-major)."""
    assert bits in (4, 6), "dequant_dense_fly supports bits 4/6 fast paths only"
    N = trellis.shape[1] * 16
    K = trellis.shape[0] * 16
    assert N % 128 == 0 and K % 128 == 0, (N, K)
    if out is None:
        out = torch.empty((N, K), dtype=torch.half, device=trellis.device)
    grid = lambda meta: (
        N // meta["BLOCK_N"],
        K // meta["BLOCK_K"],
    )
    _dequant_dense_kernel[grid](
        trellis,
        out,
        N, K,
        trellis.stride(0), trellis.stride(1),
        out.stride(0),
        K_BITS=bits,
        N_PACKED=trellis.shape[-1],
        CB=cb,
    )
    return out


# ---------------------------------------------------------------------------
# FlyDSL WMMA GEMM wrapper (tile heuristic from the Phase-1 sweep on gfx1100)
# ---------------------------------------------------------------------------

def _pick_tile(M: int, N: int, K: int):
    """(reg_m, reg_n, reg_k, waves_m, waves_n, group_m) for the shape, or None.

    Phase-1 sweep on RX 7900 XTX (gfx1100, f16 in/out):
      M<=128 small grids: 64x128x64 fills the 96 CUs best (46.7 TF/s vs
             rocBLAS 33.1 at 5120x17408; 76.6 vs 52.2 at the lm_head shape)
      M>=512 the production default 128x128x32 wins (96 TF/s, rocBLAS parity)
    N is always BLOCK_N=128 (all EXL3 out_features are 128-divisible); the
    16/32-row tiles exist so arbitrary M chunks can run unpadded.
    """
    if N % 128 or K % 64:
        return None
    if M <= 128:
        if M % 64 == 0:
            return (2, 4, 4, 2, 2, 8)      # 64x128x64, 128 threads
        if M % 32 == 0:
            return (2, 4, 4, 1, 2, 8)      # 32x128x64, 64 threads
        if M % 16 == 0:
            return (1, 4, 4, 1, 2, 8)      # 16x128x64, 64 threads
        return None
    if M % 128 == 0:
        return (4, 4, 2, 2, 2, 8)          # 128x128x32 default
    if M % 64 == 0:
        return (2, 4, 4, 2, 2, 8)          # 64x128x64
    return None


def _get_gemm(M: int, N: int, K: int, out_fp32: bool):
    global _FLY_FAILED
    if not _FLY_IMPORTABLE:
        return None
    key = (M, N, K, out_fp32)
    if key in _GEMM_CACHE:
        return _GEMM_CACHE[key]
    tile = _pick_tile(M, N, K)
    if tile is None:
        _GEMM_CACHE[key] = None
        return None
    rm, rn, rk, wm, wn, gm = tile
    bm, bn, bk = 16 * rm * wm, 16 * rn * wn, 16 * rk
    assert M % bm == 0 and N % bn == 0 and K % bk == 0
    try:
        launch, _, _, _ = _create_wmma(
            M, N, K,
            in_dtype="f16",
            out_dtype="f32" if out_fp32 else "f16",
            reg_m=rm, reg_n=rn, reg_k=rk, waves_m=wm, waves_n=wn, group_m=gm,
        )
    except Exception:
        _FLY_FAILED = True
        _GEMM_CACHE[key] = None
        return None
    _GEMM_CACHE[key] = launch
    return launch


def _pad_rows(x: torch.Tensor, mult: int) -> torch.Tensor:
    rows = x.shape[0]
    padded = torch.zeros((mult * ((rows + mult - 1) // mult), x.shape[1]),
                         dtype=x.dtype, device=x.device)
    padded[:rows] = x
    return padded


def fly_gemm(xh: torch.Tensor, w_t: torch.Tensor, out: torch.Tensor) -> None:
    """C[M, N] = xh[M, K] @ w_t[N, K]^T into ``out`` (pads M if needed)."""
    M, K = xh.shape
    N = w_t.shape[0]
    launch = _get_gemm(M, N, K, out.dtype == torch.float)
    if launch is not None:
        launch(out, xh, w_t, torch.cuda.current_stream())
        return
    # pad M up to the smallest tile multiple with a kernel
    bm = 16
    while bm < M and _pick_tile(bm * 2, N, K) is not None:
        bm *= 2
    if _pick_tile(bm, N, K) is None:
        raise RuntimeError(f"fly_gemm: no tile for M={M} N={N} K={K}")
    xp = _pad_rows(xh, bm)
    out_p = torch.empty((xp.shape[0], N), dtype=out.dtype, device=out.device)
    launch = _get_gemm(xp.shape[0], N, K, out.dtype == torch.float)
    assert launch is not None
    launch(out_p, xp, w_t, torch.cuda.current_stream())
    out.copy_(out_p[:M])


# ---------------------------------------------------------------------------
# Full EXL3 prefill op (had -> dequant -> GEMM -> had)
# ---------------------------------------------------------------------------

_W_CACHE: "OrderedDict[tuple, tuple]" = OrderedDict()
_W_CACHE_MAX_BYTES = 4 << 30


def _get_dense_weight(trellis: torch.Tensor, bits: int, cb: int) -> torch.Tensor:
    """Cache the dense W_T per trellis data_ptr (prefill reuses each layer's
    weights across chunks of the same prompt; LRU-ish, bounded to
    _W_CACHE_MAX_BYTES; a single weight larger than the budget is kept)."""
    key = (trellis.data_ptr(), bits, cb)
    hit = _W_CACHE.get(key)
    if hit is not None:
        _W_CACHE.move_to_end(key)
        return hit[0]
    w = dequant_dense_fly(trellis, bits, cb)
    _W_CACHE[key] = (w, w.numel() * 2)
    while len(_W_CACHE) > 1 and sum(v[1] for v in _W_CACHE.values()) > _W_CACHE_MAX_BYTES:
        _W_CACHE.popitem(last=False)
    return w


def exl3_prefill_fly(
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
    cache_weights: bool = True,
) -> torch.Tensor:
    """Complete EXL3 linear forward for M > 1, dequant-dense + FlyDSL GEMM."""
    global _FLY_FAILED
    assert x.dtype == torch.half, "exl3_prefill_fly expects half input"
    x_flat = x.reshape(-1, in_features)
    rows = x_flat.shape[0]

    cb = 1 if mcg else (2 if mul1 else 0)
    bits = K
    N = out_features

    if bits not in (4, 6) or N % 128 != 0 or in_features % 128 != 0:
        return None  # caller falls back

    y = torch.empty((rows, out_features), dtype=out_dtype, device=device)
    xh = torch.empty_like(x_flat)

    torch.ops.exl3_ops.had_r_128_triton(x_flat, xh, suh, None, 1.0)

    if _FLY_FAILED:
        return None
    try:
        w_t = _get_dense_weight(trellis, bits, cb) if cache_weights \
            else dequant_dense_fly(trellis, bits, cb)
        fly_gemm(xh, w_t, y)
    except Exception:
        _FLY_FAILED = True
        raise
    torch.ops.exl3_ops.had_r_128_triton(y, y, None, svh, 1.0)

    if bias is not None:
        y += bias
    return y.view(x.shape[:-1] + (out_features,))
