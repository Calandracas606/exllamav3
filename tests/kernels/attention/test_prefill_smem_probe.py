"""
The prefill tile ladder must be decided by what the device can load, not by what a compile-only
probe reports.

pick_config() takes the first tile whose probe reports a footprint within the device's budget, and
the probe used only CompiledKernel.metadata.shared from a warmup compile. On some Triton builds the
warmup metadata and the loaded module disagree for the same specialization, so the ladder accepts a
tile the launch then refuses.

Observed on a 3x7900XTX (TP=3) server, Qwen3-27B, fp16 cache, head_dim 256:

    config (128, 32, 8, 2): 65536 B (fits 65536 B)
    out of resource: shared memory, Required: 131072, Hardware limit: 65536

65536 is BLOCK_M * HD_PAD * 2 and 131072 is BLOCK_M * HD_PAD * 4 (128 x 256): the probe saw an
fp16-sized accumulator where the load needed fp32. A head_dim 128 config probed 32768 =
128 x 128 x 2, the same halving. The same device launched (64, 32, 4, 2) at 65536 B.

The fix keeps the compiled footprint as a fast filter and adds an empty-grid launch, which resolves
the specialization and runs the shared-memory budget check without executing any program instance.
"""
import pytest
import torch

from exllamav3.modules.attention_fn import smem, triton_paged
from exllamav3.modules.attention_fn.triton_paged import _paged_attn_prefill_kernel
from exllamav3.constants import PAGE_SIZE
from testlib.attention import rand_paged_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _inputs(device):
    """head_dim 256 / GQA over a long paged cache: the geometry from the report, where the stock
    tile does not fit and the ladder has to step down."""
    torch.manual_seed(0)
    bsz, past, q_len, hd = 1, 2048, 132, 256
    n_q_heads, n_kv_heads = 8, 2
    kc, vc, bt, sl = rand_paged_cache(bsz, past + q_len + 8, n_kv_heads, hd, past, device)
    k = torch.randn((bsz, q_len, n_kv_heads, hd), dtype = torch.half, device = device)
    v = torch.randn_like(k)
    q = torch.randn((bsz, q_len, n_q_heads, hd), dtype = torch.half, device = device)
    triton_paged._paged_kv_update(
        k, v, kc, vc, bt, sl, bt.shape[1], bsz, q_len, n_kv_heads, PAGE_SIZE, hd)
    return q, kc, vc, bt, sl


def _run(q, kc, vc, bt, sl):
    return triton_paged.paged_attn_triton_prefill(q, None, None, kc, vc, bt, sl, causal = True)


def test_ladder_survives_an_under_reporting_compile(monkeypatch):
    """Emulate a build whose warmup metadata under-reports the footprint -- what produced the
    crash -- and check the ladder still resolves a tile the launch can load."""
    device = torch.device("cuda:0")
    q, kc, vc, bt, sl = _inputs(device)
    real_run = _paged_attn_prefill_kernel.run

    def under_reporting_run(*args, grid = None, warmup = False, **kwargs):
        compiled = real_run(*args, grid = grid, warmup = warmup, **kwargs)
        if warmup:
            class _Halved:
                shared = compiled.metadata.shared // 2
            compiled.metadata = _Halved()
        return compiled

    monkeypatch.setattr(_paged_attn_prefill_kernel, "run", under_reporting_run)
    smem._picks.clear()

    out = _run(q, kc, vc, bt, sl)
    torch.cuda.synchronize()
    assert torch.isfinite(out).all()
    assert any(c is not None for c in smem._picks.values()), "the ladder resolved no tile"


def test_picked_tile_is_loadable(monkeypatch):
    """Independently confirm the picked tile passes the load check, using the arguments of the
    probe compile."""
    device = torch.device("cuda:0")
    q, kc, vc, bt, sl = _inputs(device)
    captured = {}
    real_run = _paged_attn_prefill_kernel.run

    def capture(*args, grid = None, warmup = False, **kwargs):
        if warmup:
            captured["args"] = args
            captured["kwargs"] = kwargs
        return real_run(*args, grid = grid, warmup = warmup, **kwargs)

    monkeypatch.setattr(_paged_attn_prefill_kernel, "run", capture)
    smem._picks.clear()

    _run(q, kc, vc, bt, sl)
    torch.cuda.synchronize()

    assert captured, "no probe compile captured"
    assert smem.launch_compatible(
        _paged_attn_prefill_kernel, (0, 0, 0), captured["args"], **captured["kwargs"]), \
        "the ladder picked a tile the device cannot load"


def test_probe_never_launches_a_live_grid(monkeypatch):
    """The probe loads the module with an empty grid so the budget check runs, but no program
    instance may execute: the ladder passes placeholder buffers. Only the probe's own calls are
    inspected -- the real computation that follows is expected to use a live grid."""
    device = torch.device("cuda:0")
    q, kc, vc, bt, sl = _inputs(device)
    seen = []
    real_run = _paged_attn_prefill_kernel.run
    in_probe = [False]
    real_pick = smem.pick_config

    def capture(*args, grid = None, warmup = False, **kwargs):
        if in_probe[0]:
            seen.append((warmup, grid))
        return real_run(*args, grid = grid, warmup = warmup, **kwargs)

    def spy_pick(dev, name, key, candidates, probe):
        in_probe[0] = True
        try:
            return real_pick(dev, name, key, candidates, probe)
        finally:
            in_probe[0] = False

    monkeypatch.setattr(_paged_attn_prefill_kernel, "run", capture)
    monkeypatch.setattr(triton_paged, "pick_config", spy_pick)
    smem._picks.clear()

    _run(q, kc, vc, bt, sl)
    torch.cuda.synchronize()

    assert any(warmup and grid == (1,) for warmup, grid in seen), "no compile-only probe"
    probe_grids = [grid for warmup, grid in seen if grid != (1,)]
    assert probe_grids and all(grid == (0, 0, 0) for grid in probe_grids), \
        f"probe used a live grid: {probe_grids}"
