"""AITER (AI Tensor Engine for ROCm) Triton kernel bridge.

When AITER is installed (``pip install -e .`` with ``AITER_TRITON_ONLY=1``),
its Triton kernels replace the pure-PyTorch fallbacks for operations that are
excluded from the exllamav3 C++ extension on ROCm (norm.cu, attention.cu, etc.).

Each kernel is registered as a PyTorch custom operator via ``torch.library.custom_op``
with a ``register_fake`` meta kernel, making it traceable by ``torch.compile`` and
exportable via ``torch.export`` without graph breaks. The public functions match
the signatures expected by exllamav3's dispatch layer (in-place output writes)
and are monkey-patched onto ``exllamav3_ext`` transparently.

On gfx1100 (RDNA3), only AITER's Triton layer is usable — CK and ASM kernels are
CDNA-only. This module is therefore gated on ``AITER_TRITON_ONLY`` being active.
"""
from __future__ import annotations

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)

_aiter_available = False
_aiter_checked = False


def is_aiter_available() -> bool:
    """Return True if AITER Triton kernels can be imported and used."""
    global _aiter_available, _aiter_checked
    if _aiter_checked:
        return _aiter_available
    _aiter_checked = True
    if not torch.version.hip:
        return False
    if os.environ.get("EXL_AITER_DISABLE", "0") == "1":
        return False
    try:
        import aiter  # noqa: F401
        from aiter.ops.triton.normalization.rmsnorm import rmsnorm_forward_inference  # noqa: F401
        _aiter_available = True
        logger.info("AITER Triton kernels available — using AITER for ROCm fallback ops")
    except Exception:
        _aiter_available = False
    return _aiter_available


# ---------------------------------------------------------------------------
# Custom operator registration
#
# We register *functional* custom ops (no mutation, return fresh tensors).
# These are the ops that torch.compile can trace through. The in-place wrappers
# below call these functional ops and copy the result into the output tensor.
# ---------------------------------------------------------------------------

# Thread the AITER import through a lazy loader so the custom_op body doesn't
# import aiter at module-load time (which would fail on non-ROCm platforms).

def _aiter_rmsnorm(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    from aiter.ops.triton.normalization.rmsnorm import rmsnorm_forward_inference
    return rmsnorm_forward_inference(x, w, eps)


_ops_registered = False

def _register_custom_ops():
    global _ops_registered
    if _ops_registered:
        return
    _ops_registered = True

    @torch.library.custom_op(
        "exl3::rms_norm", mutates_args=(), device_types="cuda"
    )
    def rms_norm_op(
        x: torch.Tensor,
        w: Optional[torch.Tensor],
        eps: float,
        constant_bias: float,
        constant_scale: float,
    ) -> torch.Tensor:
        if w is None:
            w = torch.ones(x.shape[-1], dtype=x.dtype, device=x.device)
        if constant_bias != 0.0:
            w = w + constant_bias
        result = _aiter_rmsnorm(x, w, eps)
        if constant_scale != 1.0:
            result = result * constant_scale
        return result

    @rms_norm_op.register_fake
    def _(x, w, eps, constant_bias, constant_scale):
        return torch.empty_like(x)


# ---------------------------------------------------------------------------
# Public API — in-place wrappers matching ext.rms_norm / ext.rms_norm_res_in
#
# These are what get monkey-patched onto exllamav3_ext. They call the
# functional custom op internally so torch.compile sees a clean op boundary.
# ---------------------------------------------------------------------------

def rms_norm(
    x: torch.Tensor,
    w: torch.Tensor | None,
    y: torch.Tensor,
    eps: float,
    constant_bias: float = 0.0,
    constant_scale: float = 1.0,
    span_heads: bool = False,
    add_residual: bool = False,
) -> None:
    """AITER-accelerated RMSNorm matching ext.rms_norm signature.

    exllamav3 always flattens to 2D [N, C] before calling (except span_heads),
    which is exactly the layout AITER's rmsnorm_forward_inference expects.
    """
    result = torch.ops.exl3.rms_norm(x, w, eps, constant_bias, constant_scale)
    y.copy_(result)


def rms_norm_res_in(
    x: torch.Tensor,
    w: torch.Tensor | None,
    y: torch.Tensor,
    r: torch.Tensor,
    eps: float,
    constant_bias: float = 0.0,
    constant_scale: float = 1.0,
) -> None:
    """AITER-accelerated RMSNorm with residual input."""
    # Functional: compute residual + norm in one op call
    res = r + x
    result = torch.ops.exl3.rms_norm(res, w, eps, constant_bias, constant_scale)
    r.copy_(res)
    y.copy_(result)
