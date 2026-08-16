"""Tests for the Triton EXL3 linear implementation (modules/quant/exl3_triton.py).

The C++ reconstruct+hgemm path (LinearEXL3.reconstruct_hgemm) is treated as
the reference and assumed correct. Covers:

- the fused dequant+GEMM op (exl3_ops::exl3_gemm_triton)
- the full linear composition op (exl3_ops::LinearEXL3_triton)
- the Triton Hadamard transform (exl3_ops::had_r_128_triton) vs the C++
  ext.had_r_128, bit-identical
- torch.compile(fullgraph=True) composability
- the LinearEXL3.forward dispatch with EXL3_PREFER_TRITON_LINEAR=1 vs the reference

Bits 4 and 6 have dedicated fast paths; other widths use the generic path.
"""
import os

import pytest
import torch

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_triton import (
    had_r_128_triton as had_r_128_triton_fn,
    has_triton,
    linear_exl3_triton,
)

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA (or ROCm) device required"),
    pytest.mark.skipif(not has_triton, reason="Triton not available"),
]


def device():
    return torch.device(os.environ.get("EXL_TEST_DEVICE", "cuda:0"))


# ---------------------------------------------------------------------------
# Helpers
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


def reference_reconstruct_hgemm(x, trellis, suh, svh, K, mcg, mul1,
                                in_features, out_features, dev):
    """The C++ reference: hadamard -> reconstruct -> hgemm -> hadamard."""
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
# Shapes: (in_features, out_features, K, mcg, mul1)
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
    (1024, 1024, 8, True, False),   # generic (non-fast-path) width
]


# ---------------------------------------------------------------------------
# Full linear (the public entry)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_features,out_features,K,mcg,mul1", SHAPES)
@pytest.mark.parametrize("rows", [1, 2, 17])
def test_linear_vs_reference(in_features, out_features, K, mcg, mul1, rows):
    dev = device()
    torch.manual_seed(42)
    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    x = torch.randn(rows, in_features, dtype=torch.half, device=dev)

    y_ref = reference_reconstruct_hgemm(
        x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev
    )
    y = linear_exl3_triton(
        x, trellis, suh, svh, K, mcg, mul1,
        in_features, out_features, dev, torch.half,
    )

    torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=0.5)


@pytest.mark.parametrize("in_features,out_features,K,mcg,mul1", SHAPES[:6])
def test_linear_with_bias(in_features, out_features, K, mcg, mul1):
    dev = device()
    torch.manual_seed(7)
    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    x = torch.randn(2, in_features, dtype=torch.half, device=dev)
    bias = (torch.randn(out_features, device=dev) * 0.1).to(torch.half)

    y_ref = reference_reconstruct_hgemm(
        x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev
    ) + bias

    y = linear_exl3_triton(
        x, trellis, suh, svh, K, mcg, mul1,
        in_features, out_features, dev, torch.half, bias,
    )

    torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=0.5)


@pytest.mark.parametrize("in_features,out_features,K,mcg,mul1", SHAPES[:6])
def test_linear_fp32_output(in_features, out_features, K, mcg, mul1):
    dev = device()
    torch.manual_seed(11)
    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    x = torch.randn(1, in_features, dtype=torch.half, device=dev)

    y = linear_exl3_triton(
        x, trellis, suh, svh, K, mcg, mul1,
        in_features, out_features, dev, torch.float,
    )
    assert y.dtype == torch.float

    y_ref = reference_reconstruct_hgemm(
        x, trellis, suh, svh, K, mcg, mul1, in_features, out_features, dev
    )
    torch.testing.assert_close(y, y_ref.float(), rtol=2e-2, atol=0.5)


# ---------------------------------------------------------------------------
# The composition op directly
# ---------------------------------------------------------------------------



@pytest.mark.parametrize("in_features,out_features,K,mcg,mul1", SHAPES[:4])
def test_linear_op_compile_fullgraph(in_features, out_features, K, mcg, mul1):
    dev = device()
    torch.manual_seed(42)
    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)
    x = torch.randn(2, in_features, dtype=torch.half, device=dev)

    def fn(inp):
        y = torch.empty(inp.shape[0], out_features, dtype=torch.half, device=dev)
        xh = torch.empty_like(inp)
        from exllamav3.modules.quant.exl3_triton import _linear_exl3_triton
        _linear_exl3_triton(
            inp, y, xh, trellis, suh, svh, K, mcg, mul1, None, in_features, out_features,
        )
        return y

    y_ref = fn(x)
    y_compiled = torch.compile(fn, fullgraph=True)(x)
    torch.testing.assert_close(y_compiled, y_ref, rtol=2e-2, atol=0.5)


# ---------------------------------------------------------------------------
# Triton Hadamard vs the C++ kernel (bit-identical)
# ---------------------------------------------------------------------------

HAD_SHAPES = [128, 256, 512, 1024, 2048]


@pytest.mark.parametrize("dim", HAD_SHAPES)
@pytest.mark.parametrize("dtype", [torch.half, torch.float])
def test_had_r_128_triton_vs_cpp(dim, dtype):
    dev = device()
    torch.manual_seed(42)
    x = torch.randn(2, dim, dtype=dtype, device=dev)
    s = (torch.randint(0, 2, (dim,), device=dev).to(torch.half) * 2 - 1)

    for which in ("pre", "post"):
        y_tri = torch.empty_like(x)
        y_cpp = torch.empty_like(x)
        if which == "pre":
            had_r_128_triton_fn(x, y_tri, s, None, 1.0)
            ext.had_r_128(x, y_cpp, s, None, 1.0)
        else:
            had_r_128_triton_fn(x, y_tri, None, s, 1.0)
            ext.had_r_128(x, y_cpp, None, s, 1.0)
        torch.testing.assert_close(y_tri, y_cpp, rtol=0, atol=0)




# ---------------------------------------------------------------------------
# LinearEXL3.forward dispatch: EXL3_PREFER_TRITON_LINEAR=1 vs the C++ reference
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("in_features,out_features,K,mcg,mul1", SHAPES[:6])
def test_forward_dispatch_matches_reference(in_features, out_features, K, mcg, mul1, monkeypatch):
    """With EXL3_PREFER_TRITON_LINEAR=1 the module forward must match reconstruct_hgemm."""
    from exllamav3.modules.quant import exl3 as exl3_mod

    dev = device()
    torch.manual_seed(123)
    trellis = make_random_trellis(in_features, out_features, K, dev)
    suh, svh = make_random_suh_svh(in_features, out_features, dev)

    mcg_tensor = torch.ones(1, dtype=torch.half, device=dev) if mcg else None
    mul1_tensor = torch.ones(1, dtype=torch.half, device=dev) if mul1 else None

    lin = exl3_mod.LinearEXL3(
        None, in_features, out_features,
        suh=suh, svh=svh, trellis=trellis,
        mcg=mcg_tensor, mul1=mul1_tensor,
    )

    x = torch.randn(3, in_features, dtype=torch.half, device=dev)

    # Reference: reconstruct_hgemm through the normal dispatch
    y_ref = lin.reconstruct_hgemm(x, None)

    # Triton: force the env flag through the module-level gate
    monkeypatch.setattr(exl3_mod, "use_triton", True)
    y = lin.forward(x, {})

    torch.testing.assert_close(y, y_ref, rtol=2e-2, atol=0.5)
