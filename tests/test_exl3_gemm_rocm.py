"""Tests for the shared exl3_gemm stack on ROCm: the templated kernels route
through the split-K streaming inner (exl3_gemm_inner_rocm.cuh) behind the
unchanged upstream wrapper/dispatcher/comp_units, and libtorch/linear.cpp
(BC_LinearEXL3) is upstream's own class.

Reference convention (had_r_128 ->
reconstruct -> hgemm -> had_r_128.
"""

import os
import pytest
import torch

torch.manual_seed(1234)

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(
    not torch.version.hip,
    reason = "ROCm exl3_gemm inner",
)

def device():
    return torch.device(os.environ.get("EXL_TEST_DEVICE", "cuda:0"))


def make_trellis(in_features, out_features, K, dev):
    # Same construction as test_reconstruct_had.py
    return torch.randint(
        0, 65536,
        (in_features // 16, out_features // 16, 256 * K // 16),
        dtype = torch.int32,
        device = dev,
    ).to(torch.short)


def reference(x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev):
    x = x.view(-1, in_features)
    rows = x.shape[0]
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    w = torch.empty((in_features, out_features), dtype = torch.half, device = dev)
    ext.reconstruct(w, trellis, K, mcg, mul1)
    y = torch.empty((rows, out_features), dtype = torch.half, device = dev)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y


def run_gemm(x, trellis, suh, svh, K, mcg, mul1, in_features, out_features,
             out_fp32, dev, force_num_sms = 0):
    m = x.shape[0]
    y = torch.empty(
        (m, out_features),
        dtype = torch.float if out_fp32 else torch.half,
        device = dev,
    )
    xh = torch.empty_like(x)
    ext.exl3_gemm(x, trellis, y, suh, xh, svh, -1, mcg, mul1, force_num_sms)
    return y


def rel_err(y, ref):
    return (y.float() - ref.float()).abs().max().item() / max(ref.float().abs().max().item(), 1e-6)


# All bit widths and codebook modes (tier C fragment path), m in {1, 4, 16},
# both output dtypes
@pytest.mark.parametrize("bits", [1, 2, 3, 4, 5, 6, 7, 8])
@pytest.mark.parametrize("mode", ["mul1", "mcg", "plain"])
@pytest.mark.parametrize("m", [1, 4, 16])
@pytest.mark.parametrize("out_fp32", [False, True])
def test_exl3_gemm_matrix(bits, mode, m, out_fp32):
    dev = device()
    in_f, out_f = 1024, 768
    mcg, mul1 = {"mul1": (False, True), "mcg": (True, False), "plain": (False, False)}[mode]
    trellis = make_trellis(in_f, out_f, bits, dev)
    suh = torch.randn(1, in_f, dtype = torch.half, device = dev)
    svh = torch.randn(1, out_f, dtype = torch.half, device = dev)
    x = torch.randn(m, in_f, dtype = torch.half, device = dev) * 0.1
    y = run_gemm(x, trellis, suh, svh, bits, mcg, mul1, in_f, out_f, out_fp32, dev)
    ref = reference(x, trellis, suh, svh, bits, mcg, mul1, in_f, out_f, dev)
    assert rel_err(y, ref) < 0.03, f"bits={bits} {mode} m={m} fp32={out_fp32}"


# Real model shapes, including the m=15 slab case (lm_head wider than the
# partials buffer) and the multi-chunk m-loop
@pytest.mark.parametrize(
    "in_f,out_f,m,bits",
    [
        (5120, 12288, 1, 4),    # qkv-class
        (5120, 17408, 1, 4),    # gate/up
        (17408, 5120, 1, 4),    # down_proj
        (5120, 151936, 1, 4),   # lm_head b4
        (5120, 151936, 1, 6),   # lm_head b6
        (5120, 17408, 15, 4),   # prefill chunk
        (5120, 151936, 15, 4),  # slab case
        (5120, 151936, 33, 4),  # multi m-chunk
    ],
)
def test_exl3_gemm_model_shapes(in_f, out_f, m, bits):
    dev = device()
    trellis = make_trellis(in_f, out_f, bits, dev)
    suh = torch.randn(1, in_f, dtype = torch.half, device = dev)
    svh = torch.randn(1, out_f, dtype = torch.half, device = dev)
    x = torch.randn(m, in_f, dtype = torch.half, device = dev) * 0.05
    y = run_gemm(x, trellis, suh, svh, bits, False, True, in_f, out_f, False, dev)
    ref = reference(x, trellis, suh, svh, bits, False, True, in_f, out_f, dev)
    assert rel_err(y, ref) < 0.03


# Repeated calls are deterministic (graphed decode relies on this)
@pytest.mark.parametrize("bits", [4, 6])
def test_exl3_gemm_deterministic(bits):
    dev = device()
    in_f, out_f = 2048, 1024
    trellis = make_trellis(in_f, out_f, bits, dev)
    suh = torch.randn(1, in_f, dtype = torch.half, device = dev)
    svh = torch.randn(1, out_f, dtype = torch.half, device = dev)
    x = torch.randn(1, in_f, dtype = torch.half, device = dev) * 0.1
    y1 = run_gemm(x, trellis, suh, svh, bits, False, True, in_f, out_f, False, dev).clone()
    for _ in range(5):
        y2 = run_gemm(x, trellis, suh, svh, bits, False, True, in_f, out_f, False, dev)
        assert torch.equal(y1, y2)


# Upstream's BC_LinearEXL3 (libtorch/linear.cpp, now built on ROCm) run_alloc
def test_bc_linear_exl3_run_alloc():
    dev = device()
    in_f, out_f, bits = 2048, 1024, 4
    trellis = make_trellis(in_f, out_f, bits, dev)
    suh = torch.randn(1, in_f, dtype = torch.half, device = dev)
    svh = torch.randn(1, out_f, dtype = torch.half, device = dev)
    xh = torch.empty(1, in_f, dtype = torch.half, device = dev)
    bc = ext.BC_LinearEXL3(trellis, suh, svh, bits, None, False, True, xh)
    x = torch.randn(1, in_f, dtype = torch.half, device = dev) * 0.1
    y = bc.run_alloc(x, out_f, False)
    ref = reference(x, trellis, suh, svh, bits, False, True, in_f, out_f, dev)
    assert rel_err(y, ref) < 0.03
    # m > 1 through run_alloc (exercises the eager exl3_gemm entry)
    x2 = torch.randn(7, in_f, dtype = torch.half, device = dev) * 0.1
    y2 = bc.run_alloc(x2, out_f, False)
    ref2 = reference(x2, trellis, suh, svh, bits, False, True, in_f, out_f, dev)
    assert rel_err(y2, ref2) < 0.03
