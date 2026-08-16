"""FlyDSL EXL3 fused dequant + GEMV (M == 1 decode path).

Opt-in alternative to the Triton implementation (``exl3_gemm_triton.py``),
enabled with ``EXL3_GEMM_FLY=1``. Computes the same function for M == 1:

    y = had_r_128(dequant(trellis) @ had_r_128(x))

with the dequant + GEMV kernel authored in AMD FlyDSL
(https://github.com/ROCm/FlyDSL, pip package ``flydsl``). The row Hadamards
reuse the existing had_r_128_triton function, exactly like the Triton
composition. The module imports cleanly without flydsl/triton installed; the
env gate is checked per call so kill switches behave like the other paths.

Kernel design (single FlyDSL kernel per (bits, cb, out dtype), M == 1 only):

Each thread owns ONE column pair (c, c+8) of one 16x16 trellis subtile and
accumulates two fp32 registers across the whole K loop, so the k-loop contains
no cross-lane traffic whatsoever (no shuffles, no LDS, no barriers) and the
only reduction is the per-thread register sum itself.

bits=4 (verified algebra, see AGENTS.md): element (r, c) of a subtile has
codebook code
    code = funnel(word[t], word[t-1], 28 - 4j) & 0xFFFF
    t = 4*(c%8) + (r%8)//2,  j = 4*(c//8) + 2*(r//8) + r%2,  r = 8rh+2q+p
Columns c and c+8 share (c%8) -> share the 5-word window word[4cl-1 .. 4cl+3]
(word -1 wraps to word 31 of the same subtile).

bits=6: element e = 16a + 4b + jj,
    code = funnel(word[u], word[u-1], C_b - 6jj) & 0xFFFF
    u = 3a + f(b), f = [0,1,2,2], C = [26,34,42,18]
    r = 8*j1 + 4*(a&1) + 2*(b>>1) + j0,  c = 8*(b&1) + (a>>1)
Columns c and c+8 share cA = c%8... (c = 8*b0 + cA) -> share the 7-word window
word[6cA-1 .. 6cA+5] (word -1 wraps to word 47).

The funnel is a single v_alignbit_b32 via inline asm with the shift as a
compile-time immediate for every one of the 32 unrolled codes. The immediate
shift of v_alignbit_b32 is masked to 5 bits (verified on gfx1100), so the
bits=6 shifts >= 32 (C_b - 6jj up to 42) use the exact algebraic fallback
hi >> (s - 32). Word loads for k+1 are prefetched one iteration ahead
(1-deep manual pipeline) so the DRAM stream overlaps the decode/FMA block.
Plain kernel launches on the caller's stream; no host syncs, no allocations:
CUDA/HIP-graph capturable.
"""
from __future__ import annotations

import os
import torch

# ---------------------------------------------------------------------------
# Availability
# ---------------------------------------------------------------------------

try:
    import flydsl.compiler as _flyc
    import flydsl.expr as _fx
    from flydsl._mlir import ir as _ir
    from flydsl._mlir.dialects import llvm as _llvm
    _FLY_IMPORTABLE = True
except Exception:  # pragma: no cover - flydsl is an optional runtime dep
    _FLY_IMPORTABLE = False
    _flyc = None
    _fx = None
    _ir = None
    _llvm = None

# Latch: set after the first in-flight failure so a broken flydsl install
# costs one exception, not one per decode step.
_FLY_FAILED = False


def fly_available() -> bool:
    """True when the FlyDSL path is enabled (EXL3_GEMM_FLY=1) and usable."""
    global _FLY_FAILED
    if _FLY_FAILED or not _FLY_IMPORTABLE:
        return False
    return os.environ.get("EXL3_GEMM_FLY", "0") != "0"


def _i32(u: int) -> int:
    """Reinterpret an unsigned 32-bit literal as a (possibly negative) signed
    python int so it round-trips through fx.Int32 without numpy cast warnings."""
    return u - (1 << 32) if u >= (1 << 31) else u


def _i16(u: int) -> int:
    return u - (1 << 16) if u >= (1 << 15) else u


# ---------------------------------------------------------------------------
# Device-side helpers (host-side traced builders)
# ---------------------------------------------------------------------------

def _funnel_imm(lo, hi, s: int):
    """((hi << 32) | lo) >> s, s a compile-time int, via a single
    v_alignbit_b32.

    Verified on gfx1100 (probe: /tmp/fly_probe_alignbit.py):
    `v_alignbit_b32 d, $1, $2, imm` computes (($1 << 32) | $2) >> (imm & 31),
    i.e. $1 is the HIGH word, $2 the LOW word, and the immediate shift is
    masked to 5 bits. Shifts >= 32 (bits=6: C_b - 6jj up to 42) therefore use
    the exact algebraic fallback hi >> (s - 32): the funnel output for s >= 32
    only contains hi's bits."""
    if s < 32:
        return _llvm.inline_asm(
            _ir.IntegerType.get_signless(32),
            [hi.ir_value(), lo.ir_value()],
            f"v_alignbit_b32 $0, $1, $2, {s}",
            "=v,v,v",
            has_side_effects=False,
        )
    return hi.ir_value().shrui(_fx.Int32(s - 32))


def _decode_code(code, cb: int):
    """decode_3inst from the C++ reference (quant/codebook.cuh): a 16-bit
    codebook index (already masked) -> fp16 weight. All arithmetic on the i32
    bit pattern; shifts are logical (shrui)."""
    fx = _fx
    from flydsl.expr import math as _fmath

    def fxfma(a, b, c):
        return _fmath.fma(a, b, c)

    if cb == 0:
        x = code * 89226354 + 64248484
        x = _i32(0x3B603B60) ^ (x & _i32(0x8FFF8FFF))
    elif cb == 1:
        x = code * _i32(0xCBAC1FED)
        x = _i32(0x3B603B60) ^ (x & _i32(0x8FFF8FFF))
    else:  # cb == 2 (mul1)
        x = code * _i32(0x83DCD12D)
        b0 = x & 0xFF
        b1 = (x.shrui(fx.Int32(8))) & 0xFF
        b2 = (x.shrui(fx.Int32(16))) & 0xFF
        b3 = (x.shrui(fx.Int32(24))) & 0xFF
        x = (b0 + b1 + b2 + b3 + 0x6400) & 0xFFFF

    if cb in (0, 1):
        lo = x & 0xFFFF
        hi = (x.shrui(fx.Int32(16))) & 0xFFFF
        h_lo = fx.Int16(lo).bitcast(fx.Float16)
        h_hi = fx.Int16(hi).bitcast(fx.Float16)
        return h_lo + h_hi
    h = fx.Int16(x).bitcast(fx.Float16)
    k_inv = fx.Int16(0x1EEE).bitcast(fx.Float16)
    k_bias = fx.Int16(_i16(0xC931)).bitcast(fx.Float16)
    # C++ uses __hfma (one rounding). An fp32 FMA over the exact fp16 operands
    # followed by a single fp16 round reproduces those semantics bit-exactly
    # (the fp16 product+sum is exactly representable in fp32).
    return fxfma(h.to(fx.Float32), k_inv.to(fx.Float32), k_bias.to(fx.Float32)).to(fx.Float16)


# Code tables: for every one of the 32 codes a thread decodes per k-tile,
#   (main_w, nb_w, shift, r, bank)
# where the code is funnel(window[main_w], window[nb_w], shift) & 0xFFFF,
# the decoded weight multiplies x[k*16 + r] and accumulates into acc[bank].

def _code_table_bits4():
    tbl = []
    for ch in (0, 1):
        for rh in (0, 1):
            for p in (0, 1):
                j = 4 * ch + 2 * rh + p
                sh = 28 - 4 * j
                for q in (0, 1, 2, 3):
                    # window[0] = word 4cl-1; window[1+q] = word 4cl+q
                    tbl.append((1 + q, q, sh, 8 * rh + 2 * q + p, ch))
    return tbl


def _code_table_bits6():
    F = [0, 1, 2, 2]
    C = [26, 34, 42, 18]
    tbl = []
    for b0 in (0, 1):
        for b1 in (0, 1):
            b = 2 * b1 + b0
            for a0 in (0, 1):
                for jj in (0, 1, 2, 3):
                    j1, j0 = jj // 2, jj % 2
                    sh = C[b] - 6 * jj
                    w = 3 * a0 + F[b]      # word 6cA + w is window[1 + w]
                    r = 8 * j1 + 4 * a0 + 2 * b1 + j0
                    tbl.append((1 + w, w, sh, r, b0))
    return tbl


# ---------------------------------------------------------------------------
# Kernel builders (lazy; require flydsl)
# ---------------------------------------------------------------------------

_GEMV_CACHE = {}


def _build_gemv(bits: int, cb: int, out_fp16: bool, sub: int):
    key = (bits, cb, out_fp16, sub)
    if key in _GEMV_CACHE:
        return _GEMV_CACHE[key]

    fx = _fx
    flyc = _flyc
    gpu = fx.gpu

    NWORDS = 32 if bits == 4 else 48
    WIN = 5 if bits == 4 else 7          # words per thread window
    code_tbl = _code_table_bits4() if bits == 4 else _code_table_bits6()
    if bits == 4:
        off_m1_of = lambda cp: (4 * cp + 31) % 32
        offs_of = lambda cp: [4 * cp + q for q in (0, 1, 2, 3)]
    else:
        off_m1_of = lambda cp: (6 * cp + 47) % 48
        offs_of = lambda cp: [6 * cp + i for i in (0, 1, 2, 3, 4, 5)]

    @flyc.kernel
    def gemv_kernel(
        T: fx.Tensor,        # i32 [NK, NSUB*NWORDS] view of the trellis
        X: fx.Tensor,        # f16 [K]  (hadamard-transformed input row)
        Y: fx.Tensor,        # f16/f32 [N] output row
        NK: fx.Int32,
    ):
        from flydsl.expr import range_constexpr

        tid = gpu.thread_idx.x
        bid = gpu.block_idx.x

        nsub_total = T.shape.unpack()[1] // NWORDS        # static

        s = tid // 8
        cp = tid % 8                        # column-pair index (cl or cA)
        nsub = bid * sub + s
        ok = nsub < nsub_total
        nsub_c = ok.select(nsub, 0)         # clamped: safe loads everywhere

        base = nsub_c * NWORDS
        off_m1 = off_m1_of(cp)               # word -1, wrapped in-subtile
        offs = offs_of(cp)

        # window word offsets within the subtile, prefixed by the wrapped
        # word -1 (window[0] == word (NWORDS-1 wrapped), window[1+i] == word
        # offs[i]). The window is carried as a python LIST through the loop;
        # the AST rewriter packs/unpacks list pytrees into scf.for iter_args.
        win_offs = [off_m1] + offs

        cur = [T[0, base + o] for o in win_offs]

        acc0 = fx.Float32(0.0)
        acc1 = fx.Float32(0.0)

        for k in range(0, NK):
            kk = fx.Int32(k)
            # kk IS the dim0 index of the [NK, NSUB*NWORDS] view — do NOT
            # pre-multiply by the row stride; the 2D indexing applies the
            # dim0 stride itself (a flattened offset here reads OOB).

            # prefetch row k+1 (clamped to k on the last iteration) so the
            # global loads overlap the decode/FMA block of this iteration
            kn = (kk + 1 < NK).select(kk + 1, kk)
            nxt = [T[kn, base + o] for o in win_offs]

            # x values for the CURRENT k-tile (16 halfs -> f32)
            xb = kk * 16
            xr = [fx.Float32(X[xb + i]) for i in range_constexpr(16)]

            # decode + accumulate (fully unrolled; shifts are immediates)
            for (mi, ni, sh, r, bank) in code_tbl:
                code = fx.Int32(_funnel_imm(cur[mi], cur[ni], sh)) & 0xFFFF
                w = fx.Float32(_decode_code(code, cb))
                if bank == 0:
                    acc0 = acc0 + w * xr[r]
                else:
                    acc1 = acc1 + w * xr[r]

            cur = nxt

        # ---- store the two output columns
        n0 = nsub * 16 + cp
        n1 = n0 + 8
        if ok:
            if const_expr(out_fp16):
                Y[n0] = fx.Float16(acc0)
                Y[n1] = fx.Float16(acc1)
            else:
                Y[n0] = acc0
                Y[n1] = acc1

    @flyc.jit
    def gemv(
        T: fx.Tensor,
        X: fx.Tensor,
        Y: fx.Tensor,
        NK: fx.Int32,
        stream: fx.Stream = fx.Stream(None),
    ):
        nsub_total = T.shape.unpack()[1] // NWORDS
        grid_x = (nsub_total + sub - 1) // sub
        gemv_kernel(T, X, Y, NK).launch(
            grid=(grid_x, 1, 1), block=(8 * sub, 1, 1), stream=stream,
        )

    _GEMV_CACHE[key] = gemv
    return gemv


# ---------------------------------------------------------------------------
# Host-side op
# ---------------------------------------------------------------------------

def fly_gemv(xh: torch.Tensor, trellis: torch.Tensor, y: torch.Tensor, cb: int):
    """y[0, :] = dequant(trellis) @ xh[0, :] for M == 1. Launches on the
    current stream; no syncs, no allocations."""
    bits = trellis.shape[-1] * 16 // 256
    K = trellis.shape[-1]
    fn = _build_gemv(
        bits, cb, y.dtype == torch.half,
        int(os.environ.get("EXL3_GEMM_FLY_SUB", "8")),
    )
    t32 = trellis.view(torch.int32).reshape(trellis.shape[0], -1)
    fn(
        t32, xh.reshape(-1), y.reshape(-1),
        _fx.Int32(trellis.shape[0]),
        stream=_fx.Stream(torch.cuda.current_stream()),
    )


def exl3_gemm_fly(
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
    """Complete EXL3 linear forward for a single row, FlyDSL GEMV path.

    Same contract as ``exl3_gemm_triton.exl3_gemm`` (M == 1 only).
    """
    global _FLY_FAILED
    assert x.dtype == torch.half, "exl3_gemm_fly expects half input"
    x_flat = x.reshape(1, in_features)

    cb = 1 if mcg else (2 if mul1 else 0)
    y = torch.empty((1, out_features), dtype=out_dtype, device=device)
    xh = torch.empty_like(x_flat)

    from .modules.quant.exl3_triton import had_r_128_triton
    had_r_128_triton(x_flat, xh, suh, None, 1.0)
    try:
        fly_gemv(xh, trellis, y, cb)
    except Exception:
        _FLY_FAILED = True
        raise
    had_r_128_triton(y, y, None, svh, 1.0)

    if bias is not None:
        y += bias
    return y.view(x.shape[:-1] + (out_features,))
