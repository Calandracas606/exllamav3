"""Fused b/a projection GEMV + beta/g epilogue for GDN layers (ROCm only).

The split-projection GDN layers (Qwen3.5/3.6) have two tiny fp16 projections
(in_proj_b, in_proj_a, both [hidden, num_v_heads]) whose decode-time GEMVs
([1, hidden] @ [hidden, 48]) are pathological through rocBLAS: N=48 can't fill
the CUs and each call costs ~50-90 us, ~9 ms/token over 48 layers. Here the two
weights are merged into one [2*num_v_heads, hidden] buffer (by
GatedDeltaNet's deferred fill, same buffer the C++ BC_GatedDeltaNetSplit path
uses) and a single Triton kernel computes both dot products and applies the
gated_delta_net_fused_op_2 epilogue directly:

    beta[h] = sigmoid(  x . W_b[h] + bias_b[h]) * beta_scale        (bf16 out)
    g[h]    = -softplus(x . W_a[h] + bias_a[h] + dt_bias[h]) * e^{a_log[h]}
                                                                    (fp32 out)

matching gated_delta_net_fused_op_2 numerically (fp32 accumulation,
round-to-nearest bf16, softplus linear above 20).

The kernel is a plain stream launch, so it is captured by the whole-step CUDA
graphs (exllamav3/block_graph_rocm.py) like any other op. CUDA behavior is
unchanged: the path is gated on torch.version.hip in gated_delta_net.py.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

# Fixed config; the kernel is latency/launch-bound at these sizes (all
# BLOCK_K/num_warps combos bench within noise on gfx1100), so no autotuner
# (also avoids mis-ranked configs from ramping GPU clocks during tuning).
_BA_BLOCK_K = 512
_BA_NUM_WARPS = 4

# Cap on bsz * seqlen for the GEMV path; larger shapes are real GEMMs and
# belong to rocBLAS.
_BA_MAX_ROWS = 16


@triton.jit
def _gdn_ba_beta_g_kernel(
    x_ptr,                     # [rows, K] half
    w_ptr,                     # [2*NV, K] half; rows [0, NV) = W_b, [NV, 2*NV) = W_a
    bias_ptr,                  # [2*NV] half or null
    dtb_ptr,                   # [NV] bias, fp16/bf16/fp32
    alog_ptr,                  # [NV] a_log, fp16/bf16/fp32
    beta_ptr,                  # out [rows, NV] bf16
    g_ptr,                     # out [rows, NV] fp32
    K,
    NV,
    beta_scale,
    HAS_BIAS: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(0)     # output column in [0, 2*NV)
    row = tl.program_id(1)
    is_a = pid >= NV
    h = pid - tl.where(is_a, NV, 0)

    x_ptr += row.to(tl.int64) * K
    w_ptr += pid.to(tl.int64) * K

    acc = tl.zeros((BLOCK_K,), dtype = tl.float32)
    for k0 in range(0, K, BLOCK_K):
        offs = k0 + tl.arange(0, BLOCK_K)
        m = offs < K
        xv = tl.load(x_ptr + offs, mask = m, other = 0.0)
        wv = tl.load(w_ptr + offs, mask = m, other = 0.0)
        acc += xv.to(tl.float32) * wv.to(tl.float32)
    s = tl.sum(acc, axis = 0)

    if HAS_BIAS:
        s += tl.load(bias_ptr + pid).to(tl.float32)

    if is_a:
        dtb = tl.load(dtb_ptr + h).to(tl.float32)
        al = tl.load(alog_ptr + h).to(tl.float32)
        av = s + dtb
        sp = tl.where(av > 20.0, av, tl.log(1.0 + tl.exp(av)))
        g_ptr += row.to(tl.int64) * NV + h
        tl.store(g_ptr, -sp * tl.exp(al))
    else:
        beta_ptr += row.to(tl.int64) * NV + h
        tl.store(beta_ptr, (1.0 / (1.0 + tl.exp(-s))) * beta_scale)


def gdn_ba_beta_g(
    x: torch.Tensor,           # [rows, K] half, contiguous
    w_t: torch.Tensor,         # [2*NV, K] half, contiguous
    bias: torch.Tensor | None, # [2*NV] half or None
    dt_bias: torch.Tensor,     # [NV]
    a_log: torch.Tensor,       # [NV]
    beta: torch.Tensor,        # out [rows, NV] bf16
    g: torch.Tensor,           # out [rows, NV] fp32
    beta_scale: float,
):
    rows, K = x.shape
    n = w_t.shape[0]
    NV = n // 2
    _gdn_ba_beta_g_kernel[(n, rows)](
        x, w_t, bias, dt_bias, a_log, beta, g,
        K, NV, beta_scale,
        HAS_BIAS = bias is not None,
        BLOCK_K = _BA_BLOCK_K,
        num_warps = _BA_NUM_WARPS,
    )
