"""Tests for the Triton exl3_gemm operator.

Verifies correctness against the reference reconstruct_hgemm path, runs
torch.library.opcheck, and checks torch.compile with fullgraph=True.

Uses torch.testing for assertions.
"""
import os

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.exl3_gemm_triton import exl3_gemm as exl3_gemm_triton


# ---------------------------------------------------------------------------
# Helpers (shared logic with test_exl3_gemm_op.py)
# ---------------------------------------------------------------------------

def device():
    return os.environ.get("EXL_TEST_DEVICE", "cuda:0")


def make_random_trellis(in_features, out_features, K, dev):
    rows = in_features // 16
    cols = out_features // 16
    packed_size = 256 * K // 16
    encoded = torch.randint(-32768, 32767, (rows, cols, 256), dtype=torch.int16, device=dev)
    packed = torch.zeros((rows, cols, packed_size), dtype=torch.int16, device=dev)
    ext.pack_trellis(packed, encoded.contiguous(), K)
    return packed


def make_random_suh_svh(in_features, out_features, dev):
    suh = (torch.randint(0, 2, (in_features,), device=dev).to(torch.half) * 2 - 1)
    svh = (torch.randint(0, 2, (out_features,), device=dev).to(torch.half) * 2 - 1)
    return suh, svh


def reference_reconstruct_hgemm(x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev):
    original_shape = x.shape
    x = x.view(-1, in_features)
    rows = x.shape[0]
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    w = torch.empty((in_features, out_features), dtype=torch.half, device=dev)
    ext.reconstruct(w, trellis, K, mcg, mul1)
    y = torch.empty((rows, out_features), dtype=torch.half, device=dev)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y.view(original_shape[:-1] + (out_features,))


# ---------------------------------------------------------------------------
# Test shapes
# ---------------------------------------------------------------------------

SHAPES = [
    (128, 128, 4, True, False),
    (128, 128, 4, False, True),
    (256, 256, 4, True, False),
    (256, 256, 4, False, True),
    (256, 512, 4, True, False),
    (512, 256, 4, False, True),
    (512, 512, 4, True, False),
    (512, 512, 4, False, True),
    (512, 1024, 4, True, False),
    (1024, 512, 4, False, True),
    (1024, 1024, 4, True, False),
    (1024, 1024, 6, True, False),
    (1024, 1024, 6, False, True),
]


# ---------------------------------------------------------------------------
# Correctness vs reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_features, out_features, K, mcg, mul1", SHAPES)
def test_triton_correctness_2d(in_features, out_features, K, mcg, mul1):
    dev = device()
    torch.manual_seed(42)
    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    x = torch.randn(2, in_features, dtype=torch.half, device=dev)

    y_ref = reference_reconstruct_hgemm(x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev)
    y_op = exl3_gemm_triton(x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev, torch.half)

    torch.testing.assert_close(y_op, y_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("in_features, out_features, K, mcg, mul1", SHAPES)
def test_triton_correctness_bsz1(in_features, out_features, K, mcg, mul1):
    dev = device()
    torch.manual_seed(123)
    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    x = torch.randn(1, in_features, dtype=torch.half, device=dev)

    y_ref = reference_reconstruct_hgemm(x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev)
    y_op = exl3_gemm_triton(x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev, torch.half)

    torch.testing.assert_close(y_op, y_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("in_features, out_features, K, mcg, mul1", SHAPES)
def test_triton_correctness_3d(in_features, out_features, K, mcg, mul1):
    dev = device()
    torch.manual_seed(999)
    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    x = torch.randn(2, 4, in_features, dtype=torch.half, device=dev)

    y_ref = reference_reconstruct_hgemm(x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev)
    y_op = exl3_gemm_triton(x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev, torch.half)

    torch.testing.assert_close(y_op, y_ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# opcheck
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_features, out_features, K, mcg, mul1", SHAPES[:4])
def test_triton_opcheck(in_features, out_features, K, mcg, mul1):
    """torch.library.opcheck for the pure matmul triton_op."""
    dev = device()
    torch.manual_seed(42)
    x = torch.randn(2, in_features, dtype=torch.half, device=dev)
    w = torch.randn(in_features, out_features, dtype=torch.half, device=dev)
    y = torch.empty(2, out_features, dtype=torch.half, device=dev)

    args = (x, w, y)
    test_utils = ["test_schema"]
    if torch.version.hip is None:
        test_utils.append("test_faketensor")
    results = torch.library.opcheck(torch.ops.exl3.exl3_gemm_triton.default, args, test_utils=test_utils)
    for test_name, result in results.items():
        assert result == "SUCCESS", f"opcheck failed for {test_name}: {result}"


# ---------------------------------------------------------------------------
# torch.compile fullgraph
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_features, out_features, K, mcg, mul1", SHAPES[:4])
def test_triton_compile_fullgraph(in_features, out_features, K, mcg, mul1):
    """torch.compile fullgraph with the pure triton matmul op."""
    dev = device()
    torch.manual_seed(42)
    x = torch.randn(2, in_features, dtype=torch.half, device=dev)
    w = torch.randn(in_features, out_features, dtype=torch.half, device=dev)

    def fn(x, w):
        y = torch.empty(2, out_features, dtype=torch.half, device=dev)
        torch.ops.exl3.exl3_gemm_triton(x, w, y)
        return y

    y_ref = fn(x, w)
    fn_compiled = torch.compile(fn, fullgraph=True)
    y_compiled = fn_compiled(x, w)

    torch.testing.assert_close(y_compiled, y_ref, rtol=1e-2, atol=1e-2)
