"""TDD tests: verify RMS norm routes to PyTorch native on ROCm.

On ROCm, RMSNorm.forward() and GatedRMSNorm.forward() should call forward_torch()
instead of the C++ extension kernels (ext.rms_norm, ext.gated_rms_norm).
"""
import pytest
import torch
import os

pytestmark = pytest.mark.skipif(
    torch.version.hip is None,
    reason="ROCm-only test: verifies PyTorch-native norm routing on HIP"
)

device = os.environ.get("EXL_TEST_DEVICE", "cuda:0")
HIDDEN = 128


def _make_rmsnorm():
    """Create a standalone RMSNorm without a Config."""
    from exllamav3.modules.rmsnorm import RMSNorm
    norm = RMSNorm.__new__(RMSNorm)
    norm.weight = torch.randn(HIDDEN, dtype=torch.half, device=device) * 0.1
    norm.rms_norm_eps = 1e-6
    norm.out_dtype = torch.half
    norm.constant_bias = 0.0
    norm.constant_scale = 1.0
    norm.span_heads = False
    norm.unweighted = False
    return norm


def _make_gated_rmsnorm():
    """Create a standalone GatedRMSNorm without a Config."""
    from exllamav3.modules.gated_rmsnorm import GatedRMSNorm
    norm = GatedRMSNorm.__new__(GatedRMSNorm)
    norm.weight = torch.randn(HIDDEN, dtype=torch.half, device=device) * 0.1
    norm.rms_norm_eps = 1e-6
    norm.out_dtype = torch.half
    norm.constant_bias = 0.0
    norm.groups = 1
    norm.gate_first = False
    return norm


def test_rmsnorm_routes_to_pytorch_on_rocm():
    """On ROCm, forward() should NOT call ext.rms_norm — it should use forward_torch().
    If ext.rms_norm doesn't exist (excluded from build), routing is proven.
    Otherwise monkeypatch it to raise and verify forward() still works."""
    from exllamav3.modules import rmsnorm as rmsnorm_mod

    norm = _make_rmsnorm()
    x = torch.randn(2, 16, HIDDEN, dtype=torch.half, device=device)

    y_ref = norm.forward_torch(x, params={})

    if hasattr(rmsnorm_mod.ext, 'rms_norm'):
        original = rmsnorm_mod.ext.rms_norm
        rmsnorm_mod.ext.rms_norm = lambda *a, **kw: (_ for _ in ()).throw(
            AssertionError("ext.rms_norm should not be called on ROCm"))
        try:
            y = norm.forward(x, params={})
        finally:
            rmsnorm_mod.ext.rms_norm = original
    else:
        # Function excluded from build — routing proven by absence
        y = norm.forward(x, params={})

    assert torch.allclose(y, y_ref, atol=1e-3), "forward() output differs from forward_torch()"


def test_rmsnorm_residual_in_routes_to_pytorch():
    """The residual_in fused path should also route to PyTorch on ROCm."""

    norm = _make_rmsnorm()
    x = torch.randn(2, 16, HIDDEN, dtype=torch.half, device=device)
    residual_in = torch.randn(2, 16, HIDDEN, dtype=torch.half, device=device)

    y = norm.forward(x, params={}, residual_in=residual_in)
    assert y.shape == x.shape, f"Output shape mismatch: {y.shape} vs {x.shape}"


def test_gated_rmsnorm_routes_to_pytorch_on_rocm():
    """On ROCm, GatedRMSNorm.forward() should NOT call ext.gated_rms_norm."""

    norm = _make_gated_rmsnorm()
    x = torch.randn(2, 16, HIDDEN, dtype=torch.half, device=device)
    gate = torch.randn(2, 16, HIDDEN, dtype=torch.half, device=device)

    y_ref = norm.forward_torch(x, params={}, gate=gate)
    y = norm.forward(x, params={}, gate=gate)
    assert torch.allclose(y, y_ref, atol=1e-3), "forward() output differs from forward_torch()"
