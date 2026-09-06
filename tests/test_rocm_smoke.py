"""
Minimal ROCm smoke test: the extension imports, and a representative set of the kernels
the loader and forward paths depend on produces correct results on device 0. Loads no
model. Run on CUDA this is a plain kernel correctness check.
"""

import os
import pytest
import torch

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason = "requires a GPU",
)

device = os.environ.get("EXL_TEST_DEVICE", "cuda:0")
torch.cuda.set_device(device)

from exllamav3.ext import exllamav3_ext as ext


@torch.inference_mode()
def test_rms_norm():
    torch.manual_seed(1234)
    x = torch.randn(8, 256, dtype = torch.half, device = device)
    w = torch.randn(256, dtype = torch.half, device = device)
    y = torch.empty_like(x)
    ext.rms_norm(x, w, y, 1e-6, 0.0, 1.0, False, False)
    ref = x.float() * torch.rsqrt(x.float().pow(2).mean(-1, keepdim = True) + 1e-6)
    ref = (ref * w.float()).half()
    torch.testing.assert_close(y, ref, atol = 2e-2, rtol = 2e-2)


@torch.inference_mode()
def test_silu_mul():
    torch.manual_seed(1234)
    x = torch.randn(4, 128, dtype = torch.half, device = device)
    y = torch.randn(4, 128, dtype = torch.half, device = device)
    z = torch.empty_like(x)
    ext.silu_mul(x, y, z, 0.0)
    ref = (torch.nn.functional.silu(x.float()) * y.float()).half()
    torch.testing.assert_close(z, ref, atol = 2e-2, rtol = 2e-2)


@torch.inference_mode()
def test_softcap_inplace():
    x = torch.randn(2, 512, dtype = torch.half, device = device) * 30
    y = torch.empty_like(x)
    ext.softcap(x, y, 10.0)
    ref = (torch.tanh(x.float() / 10.0) * 10.0).half()
    torch.testing.assert_close(y, ref, atol = 2e-2, rtol = 2e-2)
    # in-place form used by linear.py
    ext.softcap(x, x, 10.0)
    torch.testing.assert_close(x, ref, atol = 2e-2, rtol = 2e-2)


@torch.inference_mode()
def test_routing_std_bsz1():
    """MoE routing was a crash-at-call hole on ROCm; check it selects the top-k experts."""
    torch.manual_seed(1234)
    E, K, H = 32, 4, 64
    y = torch.randn(1, H, dtype = torch.half, device = device)
    gate = torch.randn(H, E, dtype = torch.half, device = device)  # [in, out]
    gate_t = gate.T.contiguous()
    scores = torch.empty(1, E, dtype = torch.half, device = device)
    sel = torch.empty(1, K, dtype = torch.long, device = device)
    w = torch.empty(1, K, dtype = torch.half, device = device)
    ext.routing_std(y, gate, scores, sel, w, None, gate_t, None)
    logits = (y.float() @ gate.float())
    top = torch.topk(logits, K).indices.sort().values
    assert torch.equal(sel.sort().values.cpu(), top.cpu())
    ref_w = torch.softmax(logits, -1).topk(K).values
    torch.testing.assert_close(w.float(), ref_w, atol = 2e-2, rtol = 2e-2)


@torch.inference_mode()
def test_dsa_topk_smoke():
    """dsa_topk is bound on both platforms; a tiny deterministic case must run clean."""
    R, T, k = 2, 128, 8
    scores = torch.rand(R, T, dtype = torch.half, device = device)
    out = torch.empty(R, k, dtype = torch.int, device = device)
    ext.dsa_topk(scores, out, k, None, 0)
    vals = scores.gather(-1, out)
    # every selected set is the true top-k of its row
    top = torch.topk(scores.float(), k, dim = -1).values.sort(-1).values
    torch.testing.assert_close(vals.float().sort(-1).values, top, atol = 1e-3, rtol = 1e-3)


@pytest.mark.skipif(not torch.version.hip, reason = "ROCm-only capability set")
def test_rocm_capability_surface():
    """Symbols the rewritten ROCm build must provide natively (no fallbacks)."""
    for name in ["rms_norm", "silu_mul", "softcap", "routing_std", "routing_ds3_nogroup",
                 "routing_sel_norm", "count_inf_nan", "histogram", "dsa_topk",
                 "batched_conv_rewind", "batched_state_rewind", "dspark_write_rows",
                 "hc_mix", "hc_apply"]:
        assert hasattr(ext, name), name
