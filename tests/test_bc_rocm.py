"""Tests for the ROCm BC_LinearEXL3 graph-capture wrapper (bc_rocm.py).

The replay result must match the uncaptured compute; the graph must be
captured after the warmup threshold; bsz > 1, non-half inputs and fp32 output
must each take their intended path. ROCm-only.
"""
import os

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.bc_rocm import _WARMUP_ITERS, BC_LinearEXL3
from exllamav3.modules.quant.exl3_triton import linear_exl3_triton as exl3_gemm_triton

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (or ROCm) device required"),
    pytest.mark.skipif(not torch.version.hip, reason="bc_rocm is ROCm-only"),
]


def device():
    return torch.device(os.environ.get("EXL_TEST_DEVICE", "cuda:0"))


# ---------------------------------------------------------------------------
# Helpers (mirrored from tests/test_exl3_triton.py)
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# BC_LinearEXL3 graph capture (bc_rocm.py)
#
# The replay result must match the uncaptured compute; the graph must be
# captured after the warmup threshold; bsz > 1, non-half inputs and fp32 output
# must each take their intended path.
# ---------------------------------------------------------------------------

BC_SHAPES = [
    # (in_features, out_features, K, mcg, mul1, bias)
    (128, 256, 4, False, False, True),
    (256, 256, 4, True, False, False),
    (256, 512, 4, False, True, True),
]


def make_bc(in_features, out_features, K, mcg, mul1, bias, dev, xh_dtype=torch.half):
    """Build a BC_LinearEXL3 over valid EXL3-packed tensors."""
    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    b = (torch.randn(out_features, device=dev) * 0.1).to(torch.half) if bias else None
    xh = torch.empty((1, in_features), dtype=xh_dtype, device=dev)
    return BC_LinearEXL3(trellis, suh, svh, K, b, mcg, mul1, xh)


def test_bc_capture_state_machine():
    """Graph must be captured on the warmup-threshold call and replayed after."""
    dev = device()
    torch.manual_seed(42)
    bc = make_bc(128, 256, 4, False, False, False, dev)
    x = torch.randn(1, 128, dtype=torch.half, device=dev)

    for _ in range(_WARMUP_ITERS):
        assert bc._graphs.get(torch.half) is None, "graph captured too early"
        bc.run_alloc(x, 256, False)
    assert bc._graphs.get(torch.half) is not None, "graph not captured at threshold"

    # Subsequent calls must replay, not re-capture
    y0 = bc.run_alloc(x, 256, False).clone()
    g = bc._graphs[torch.half]
    for _ in range(5):
        y = bc.run_alloc(x, 256, False)
    assert bc._graphs[torch.half] is g, "graph was re-captured on replay path"
    torch.testing.assert_close(y, y0, rtol=0, atol=0)


@pytest.mark.parametrize("in_features,out_features,K,mcg,mul1,bias", BC_SHAPES)
def test_bc_replay_matches_uncaptured(in_features, out_features, K, mcg, mul1, bias):
    """Replayed output must match the uncaptured compute for varying inputs."""
    dev = device()
    torch.manual_seed(7)
    bc = make_bc(in_features, out_features, K, mcg, mul1, bias, dev)
    x = torch.randn(1, in_features, dtype=torch.half, device=dev)

    ref = bc.run_alloc(x, out_features, False).clone()  # first uncaptured call
    for _ in range(_WARMUP_ITERS - 1):
        bc.run_alloc(x, out_features, False)
    assert bc._graphs.get(torch.half) is not None

    # Replaying the same input reproduces the uncaptured result bit-exactly
    y = bc.run_alloc(x, out_features, False)
    torch.testing.assert_close(y, ref, rtol=0, atol=0)
    assert y.dtype == torch.half
    assert y.shape == (1, out_features)
    assert torch.isfinite(y.float()).all()

    # A different input must produce its own (correct) output
    x2 = torch.randn(1, in_features, dtype=torch.half, device=dev)
    y2 = bc.run_alloc(x2, out_features, False)
    assert not torch.equal(y2, y), "replay ignored input (stale static buffer)"

    r2 = exl3_gemm_triton(x2, bc.trellis, bc.suh, bc.svh, K, mcg, mul1,
                           in_features, out_features, torch.device(dev), torch.half, bc.bias)
    torch.testing.assert_close(y2, r2, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("in_features,out_features,K,mcg,mul1,bias", BC_SHAPES)
def test_bc_replay_determinism(in_features, out_features, K, mcg, mul1, bias):
    """Alternating inputs on the replay path must give stable, per-input results."""
    dev = device()
    torch.manual_seed(99)
    bc = make_bc(in_features, out_features, K, mcg, mul1, bias, dev)

    x1 = torch.randn(1, in_features, dtype=torch.half, device=dev)
    x2 = torch.randn(1, in_features, dtype=torch.half, device=dev)
    for _ in range(_WARMUP_ITERS):
        bc.run_alloc(x1, out_features, False)

    y1a = bc.run_alloc(x1, out_features, False).clone()
    y2a = bc.run_alloc(x2, out_features, False).clone()
    y1b = bc.run_alloc(x1, out_features, False).clone()
    y2b = bc.run_alloc(x2, out_features, False).clone()

    assert not torch.equal(y1a, y2a), "distinct inputs collapsed to one output"
    torch.testing.assert_close(y1a, y1b, rtol=0, atol=0)
    torch.testing.assert_close(y2a, y2b, rtol=0, atol=0)


def test_bc_fp32_output_dtype():
    """The fp32-output variant must capture its own graph and return float."""
    dev = device()
    torch.manual_seed(11)
    bc = make_bc(256, 512, 4, False, False, False, dev)
    x = torch.randn(1, 256, dtype=torch.half, device=dev)

    for _ in range(_WARMUP_ITERS):
        bc.run_alloc(x, 512, True)
    assert torch.float in bc._graphs, "fp32 graph not captured"

    y = bc.run_alloc(x, 512, True)
    assert y.dtype == torch.float
    assert y.shape == (1, 512)

    r = exl3_gemm_triton(x, bc.trellis, bc.suh, bc.svh, 4, False, False,
                          256, 512, torch.device(dev), torch.float)
    torch.testing.assert_close(y, r, rtol=1e-2, atol=1e-2)


def test_bc_bsz_gt_1_falls_through():
    """bsz > 1 must bypass the graph and still produce correct output."""
    dev = device()
    torch.manual_seed(13)
    bc = make_bc(128, 256, 4, False, False, False, dev)

    x1 = torch.randn(1, 128, dtype=torch.half, device=dev)
    for _ in range(_WARMUP_ITERS):
        bc.run_alloc(x1, 256, False)
    assert torch.half in bc._graphs, "bsz1 graph should be captured"

    xb = torch.randn(4, 128, dtype=torch.half, device=dev)
    yb = bc.run_alloc(xb, 256, False)

    assert yb.shape == (4, 256)
    assert yb.dtype == torch.half
    r = exl3_gemm_triton(xb, bc.trellis, bc.suh, bc.svh, 4, False, False,
                          128, 256, torch.device(dev), torch.half)
    torch.testing.assert_close(yb, r, rtol=1e-2, atol=1e-2)


def test_bc_non_half_input_falls_through():
    """Non-half bsz1 input must bypass the graph (wrapper casts, no capture)."""
    dev = device()
    torch.manual_seed(17)
    bc = make_bc(128, 256, 4, False, False, False, dev)

    x32 = torch.randn(1, 128, dtype=torch.float32, device=dev)
    y = bc.run_alloc(x32, 256, False)

    assert bc._graphs == {}, "no graph should be captured for non-half input"
    assert y.dtype == torch.half
    r = exl3_gemm_triton(x32, bc.trellis, bc.suh, bc.svh, 4, False, False,
                          128, 256, torch.device(dev), torch.half)
    torch.testing.assert_close(y, r, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("xh_dtype", [torch.half, torch.float32])
def test_bc_xh_buffer_reuse(xh_dtype):
    """Half constructor buffer is reused as the graph staging buffer; float32 is not."""
    dev = device()
    torch.manual_seed(19)
    bc = make_bc(128, 256, 4, False, False, False, dev, xh_dtype=xh_dtype)
    x = torch.randn(1, 128, dtype=torch.half, device=dev)
    for _ in range(_WARMUP_ITERS):
        bc.run_alloc(x, 256, False)

    assert torch.half in bc._graphs
    if xh_dtype == torch.half:
        assert bc._static_xh is bc.xh, "half xh should be reused as the staging buffer"
    else:
        assert bc._static_xh is not bc.xh
        assert bc._static_xh.dtype == torch.half


def test_bc_out_features_mismatch_rejected():
    """A mismatched out_features must fail the capture guard, not capture wrong."""
    dev = device()
    torch.manual_seed(23)
    bc = make_bc(128, 256, 4, False, False, False, dev)
    x = torch.randn(1, 128, dtype=torch.half, device=dev)

    # _capture asserts before touching any buffer, so the guard is testable directly
    with pytest.raises(AssertionError):
        bc._capture(x, 512, torch.half)  # trellis is sized for 256
