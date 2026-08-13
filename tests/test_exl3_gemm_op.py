"""Tests for the exl3_gemm custom operator.

Verifies that:
1. The custom op produces the same output as the reference reconstruct_hgemm path
2. torch.library.opcheck passes (schema, fake kernel, mutation contract)
3. The op works under torch.compile with fullgraph=True
"""
import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.exl3_gemm_op import exl3_gemm


# ---------------------------------------------------------------------------
# Helpers: create synthetic EXL3-packed tensors
# ---------------------------------------------------------------------------

def make_random_trellis(in_features: int, out_features: int, K: int, device: str) -> torch.Tensor:
    """Create a random trellis tensor with valid EXL3 packing."""
    rows = in_features // 16
    cols = out_features // 16
    packed_size = 256 * K // 16
    encoded = torch.randint(
        -32768, 32767,
        (rows, cols, 256),
        dtype=torch.int16,
        device=device,
    )
    packed = torch.zeros(
        (rows, cols, packed_size),
        dtype=torch.int16,
        device=device,
    )
    ext.pack_trellis(packed, encoded.contiguous(), K)
    return packed


def make_random_suh_svh(in_features: int, out_features: int, device: str):
    """Create random sign vectors (±1 as half)."""
    suh = (torch.randint(0, 2, (in_features,), device=device).to(torch.half) * 2 - 1)
    svh = (torch.randint(0, 2, (out_features,), device=device).to(torch.half) * 2 - 1)
    return suh, svh


def reference_reconstruct_hgemm(
    x: torch.Tensor,
    trellis: torch.Tensor,
    suh: torch.Tensor,
    svh: torch.Tensor,
    K: int,
    mcg: bool,
    mul1: bool,
    in_features: int,
    out_features: int,
    device: str,
) -> torch.Tensor:
    """Reference implementation: reconstruct weights + hgemm + Hadamard transforms."""
    original_shape = x.shape
    x = x.view(-1, in_features)
    rows = x.shape[0]
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)

    w = torch.empty((in_features, out_features), dtype=torch.half, device=device)
    ext.reconstruct(w, trellis, K, mcg, mul1)

    y = torch.empty((rows, out_features), dtype=torch.half, device=device)
    ext.hgemm(xh, w, y)

    ext.had_r_128(y, y, None, svh, 1.0)
    return y.view(original_shape[:-1] + (out_features,))


# ---------------------------------------------------------------------------
# Parametrized test shapes
# ---------------------------------------------------------------------------

SHAPES = [
    # (in_features, out_features, K, mcg, mul1)
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
    (128, 128, 4, True, False),
    (128, 128, 4, False, True),
]


def device():
    import os
    return os.environ.get("EXL_TEST_DEVICE", "cuda:0")


# ---------------------------------------------------------------------------
# Correctness tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_features, out_features, K, mcg, mul1", SHAPES)
def test_exl3_gemm_correctness(in_features, out_features, K, mcg, mul1):
    """Custom op output must match reference reconstruct_hgemm."""
    dev = device()
    torch.manual_seed(42)

    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    x = torch.randn(2, in_features, dtype=torch.half, device=dev)

    # Reference
    y_ref = reference_reconstruct_hgemm(
        x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev
    )

    # Custom op
    y_op = exl3_gemm(
        x, trellis, suh, svh, K, mcg, mul1,
        in_features, out_features, dev, torch.half,
    )

    torch.testing.assert_close(y_op, y_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("in_features, out_features, K, mcg, mul1", SHAPES)
def test_exl3_gemm_correctness_bsz1(in_features, out_features, K, mcg, mul1):
    """Single-row (decode) path."""
    dev = device()
    torch.manual_seed(123)

    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    x = torch.randn(1, in_features, dtype=torch.half, device=dev)

    y_ref = reference_reconstruct_hgemm(
        x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev
    )
    y_op = exl3_gemm(
        x, trellis, suh, svh, K, mcg, mul1,
        in_features, out_features, dev, torch.half,
    )

    torch.testing.assert_close(y_op, y_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("in_features, out_features, K, mcg, mul1", SHAPES)
def test_exl3_gemm_correctness_3d(in_features, out_features, K, mcg, mul1):
    """Multi-dimensional input (batch, seq, features)."""
    dev = device()
    torch.manual_seed(999)

    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    x = torch.randn(2, 4, in_features, dtype=torch.half, device=dev)

    y_ref = reference_reconstruct_hgemm(
        x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev
    )
    y_op = exl3_gemm(
        x, trellis, suh, svh, K, mcg, mul1,
        in_features, out_features, dev, torch.half,
    )

    torch.testing.assert_close(y_op, y_ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# opcheck tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_features, out_features, K, mcg, mul1", SHAPES[:4])
def test_exl3_gemm_opcheck(in_features, out_features, K, mcg, mul1):
    """Run torch's opcheck to validate registration, schema, fake kernel, etc."""
    dev = device()
    torch.manual_seed(42)

    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    x = torch.randn(2, in_features, dtype=torch.half, device=dev)
    y = torch.empty(2, out_features, dtype=torch.half, device=dev)

    args = (x, y, trellis, suh, svh, K, mcg, mul1, in_features, out_features)

    results = torch.library.opcheck(torch.ops.exl3.exl3_gemm.default, args)
    # opcheck returns a dict of test_name -> result string
    for test_name, result in results.items():
        assert result == "SUCCESS", f"opcheck failed for {test_name}: {result}"


# ---------------------------------------------------------------------------
# torch.compile fullgraph tests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_features, out_features, K, mcg, mul1", SHAPES[:4])
def test_exl3_gemm_compile_fullgraph(in_features, out_features, K, mcg, mul1):
    """Custom op must work under torch.compile with fullgraph=True."""
    dev = device()
    torch.manual_seed(42)

    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)

    def fn(x):
        return exl3_gemm(
            x, trellis, suh, svh, K, mcg, mul1,
            in_features, out_features, dev, torch.half,
        )

    x = torch.randn(2, in_features, dtype=torch.half, device=dev)

    # Uncompiled output
    y_ref = fn(x)

    # Compiled output — must not graph-break
    fn_compiled = torch.compile(fn, fullgraph=True)
    y_compiled = fn_compiled(x)

    torch.testing.assert_close(y_compiled, y_ref, rtol=1e-2, atol=1e-2)


# ---------------------------------------------------------------------------
# had_r_128 custom op tests
# ---------------------------------------------------------------------------

HAD_SHAPES = [128, 256, 512, 1024, 2048]

@pytest.mark.parametrize("dim", HAD_SHAPES)
def test_had_r_128_matches_pybind(dim):
    """TORCH_LIBRARY op must produce identical output to pybind11 version."""
    dev = device()
    torch.manual_seed(42)

    x = torch.randn(2, dim, dtype=torch.half, device=dev)
    suh = (torch.randint(0, 2, (dim,), device=dev).to(torch.half) * 2 - 1)

    # TORCH_LIBRARY version
    y_op = torch.empty_like(x)
    torch.ops.exl3_ops.had_r_128(x, y_op, suh, None, 1.0)

    # pybind11 version
    y_pb = torch.empty_like(x)
    ext.had_r_128(x, y_pb, suh, None, 1.0)

    torch.testing.assert_close(y_op, y_pb, rtol=0, atol=0)


@pytest.mark.parametrize("dim", HAD_SHAPES)
def test_had_r_128_post_scale(dim):
    """had_r_128 with post_scale (svh) must match pybind."""
    dev = device()
    torch.manual_seed(42)

    x = torch.randn(2, dim, dtype=torch.half, device=dev)
    svh = (torch.randint(0, 2, (dim,), device=dev).to(torch.half) * 2 - 1)

    y_op = torch.empty_like(x)
    torch.ops.exl3_ops.had_r_128(x, y_op, None, svh, 1.0)

    y_pb = torch.empty_like(x)
    ext.had_r_128(x, y_pb, None, svh, 1.0)

    torch.testing.assert_close(y_op, y_pb, rtol=0, atol=0)


@pytest.mark.parametrize("dim", HAD_SHAPES[:3])
def test_had_r_128_opcheck(dim):
    """torch.library.opcheck for had_r_128."""
    dev = device()
    torch.manual_seed(42)

    x = torch.randn(2, dim, dtype=torch.half, device=dev)
    y = torch.empty_like(x)
    suh = torch.ones(dim, dtype=torch.half, device=dev)

    args = (x, y, suh, None, 1.0)
    results = torch.library.opcheck(torch.ops.exl3_ops.had_r_128.default, args)
    for test_name, result in results.items():
        assert result == "SUCCESS", f"opcheck failed for {test_name}: {result}"


@pytest.mark.parametrize("dim", HAD_SHAPES[:3])
def test_had_r_128_compile_fullgraph(dim):
    """had_r_128 custom op must work under torch.compile with fullgraph=True."""
    dev = device()
    torch.manual_seed(42)

    x = torch.randn(2, dim, dtype=torch.half, device=dev)
    suh = torch.ones(dim, dtype=torch.half, device=dev)

    def fn(inp):
        out = torch.empty_like(inp)
        torch.ops.exl3_ops.had_r_128(inp, out, suh, None, 1.0)
        return out

    y_ref = fn(x)

    fn_compiled = torch.compile(fn, fullgraph=True)
    y_compiled = fn_compiled(x)

    torch.testing.assert_close(y_compiled, y_ref, rtol=0, atol=0)
