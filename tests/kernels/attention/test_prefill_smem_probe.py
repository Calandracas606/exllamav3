"""
The shared-memory ladder must judge a tile by the footprint the device will actually load, not by
what a compile-only warmup reports.

pick_config() walks a ladder of tile configs and takes the first whose probe reports a footprint
within the device's budget. Probes went through smem.shared_bytes(), which used only a warmup
compile. The limit that fails, however, is enforced in CompiledKernel._init_handles() -- the check
a launch performs -- and a warmup never runs it. A build where those two disagree therefore lets a
tile that needs more shared memory than the device has measure as fitting; the ladder stops there
and the launch dies with OutOfResources.

That is not hypothetical: a 3x7900XTX (TP=3) server running Qwen3-27B at head_dim 256 probed the
(128, 32, 8, 2) prefill tile at 65536 B and then failed to launch it needing 131072 B, exactly the
fp32 (BLOCK_M, HD_PAD) accumulator. The same build launched (64, 32, 4, 2) at 65536 B, so the
first tile was simply not loadable and the ladder should have stepped past it.
"""
import pytest
import torch

from exllamav3.modules.attention_fn import smem, triton_paged
from testlib.attention import quant_roundtrip, rand_paged_cache

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")


def _inputs(device):
    """A short q chunk against a long paged cache at head_dim 256 / GQA, the shape from the
    report. Enough kv for the split path to engage, few enough q rows to stay cheap."""
    from exllamav3.constants import PAGE_SIZE
    torch.manual_seed(0)
    bsz, past, q_len, hd, bits = 1, 24000, 128, 256, 4
    n_q_heads, n_kv_heads = 8, 2
    kc, vc, bt, sl = rand_paged_cache(bsz, past + q_len + 8, n_kv_heads, hd, past, device)
    k = torch.randn((bsz, q_len, n_kv_heads, hd), dtype = torch.half, device = device)
    v = torch.randn_like(k)
    q = torch.randn((bsz, q_len, n_q_heads, hd), dtype = torch.half, device = device)
    triton_paged._paged_kv_update(
        k, v, kc, vc, bt, sl, bt.shape[1], bsz, q_len, n_kv_heads, PAGE_SIZE, hd)
    pages = kc.shape[0]
    qk, sk, _ = quant_roundtrip(kc.view(pages, PAGE_SIZE, n_kv_heads * hd), bits)
    qv, sv, _ = quant_roundtrip(vc.view(pages, PAGE_SIZE, n_kv_heads * hd), bits)
    return dict(q = q, qk = qk, sk = sk, qv = qv, sv = sv, bt = bt, sl = sl,
                bits = bits, kvh = n_kv_heads, q_len = q_len)


def _run(inp):
    return triton_paged.paged_attn_triton_prefill(
        inp["q"], None, None, inp["qk"], inp["qv"], inp["bt"], inp["sl"], causal = True,
        qc = (inp["sk"], inp["sv"], inp["bits"], inp["bits"]),
        pre_appended_len = inp["q_len"], n_kv_heads_override = inp["kvh"],
    )


def test_picked_prefill_tile_actually_loads(monkeypatch):
    """Drive the real prefill entry point, capture the tile the ladder chose, then assert that
    tile's module loads. Before the fix the probe could accept a tile whose load raises, so this
    is the property that was broken: accepted implies loadable."""
    device = torch.device("cuda:0")
    picked = {}
    real_pick = smem.pick_config

    def spy(dev, name, key, candidates, probe):
        cfg = real_pick(dev, name, key, candidates, probe)
        if name == "paged_attn_prefill":
            picked["cfg"] = cfg
            picked["probe"] = probe
        return cfg

    monkeypatch.setattr(smem, "pick_config", spy)
    monkeypatch.setattr(triton_paged, "pick_config", spy)
    smem._picks.clear()

    _run(_inputs(device))

    assert "cfg" in picked, "prefill did not go through the ladder"
    cfg = picked["cfg"]
    limit = smem.smem_limit(device)
    need = picked["probe"](cfg)
    assert need <= limit, f"ladder accepted {cfg} needing {need} B over {limit} B"

    # Ask the device to load that same tile the way a launch does. This is the check a warmup skips
    grid, args, num_warps, num_stages = _prepare_like_a_launch(device, cfg)
    triton_paged._paged_attn_prefill_kernel.run(
        *args, grid = grid, warmup = True,
        num_warps = num_warps, num_stages = num_stages,
    )._init_handles()


def _prepare_like_a_launch(device, cfg):
    """Recover (grid, args, num_warps, num_stages) for `cfg` by intercepting the kernel call the
    real launch makes, so the load check uses byte-identical arguments."""
    captured = {}
    real_run = triton_paged._paged_attn_prefill_kernel.run

    def capture(*args, grid = None, warmup = False, **kwargs):
        captured["args"] = args
        captured["grid"] = grid
        captured["kwargs"] = kwargs
        if warmup:
            return real_run(*args, grid = grid, warmup = True, **kwargs)
        return real_run(*args, grid = grid, warmup = False, **kwargs)

    triton_paged._paged_attn_prefill_kernel.run = capture
    try:
        level = smem._debug
        smem._picks.clear()
        _run(_inputs(device))
    finally:
        triton_paged._paged_attn_prefill_kernel.run = real_run

    assert captured, "no probe call captured"
    kw = captured["kwargs"]
    return captured["grid"], captured["args"], kw.get("num_warps"), kw.get("num_stages")


def test_shared_bytes_loads_and_does_not_launch():
    """shared_bytes must compile with a warmup (never execute the body) and must run the load
    check, so its verdict is the device's, not the compiler's."""

    class _Meta:
        shared = 4096

    class _Kernel:
        def __init__(self):
            self.metadata = _Meta()
            self._run = None
            self.module = None
            self.loaded = 0

        def _init_handles(self):
            self.loaded += 1
            self.module = object()
            self._run = lambda *a, **k: None

    class _JIT:
        def __init__(self, kernel):
            self.kernel = kernel
            self.calls = []

        def run(self, *args, grid = None, warmup = False, **kwargs):
            self.calls.append(warmup)
            assert warmup, "shared_bytes must not launch the kernel"
            return self.kernel

    kernel = _Kernel()
    jit = _JIT(kernel)
    need = smem.shared_bytes(jit, (torch.zeros(1, device = "cuda:0"),), num_warps = 1)
    assert need == 4096
    assert kernel.loaded == 1, "shared_bytes skipped the load check"
    assert jit.calls == [True], "shared_bytes must compile with a warmup only"


def test_shared_bytes_reports_an_unloadable_tile_as_over_budget():
    """When the device refuses the module, the probe must report no fit rather than a number the
    ladder would compare against its budget."""
    from triton.runtime.errors import OutOfResources

    class _Kernel:
        class metadata:
            shared = 131072

        def _init_handles(self):
            raise OutOfResources(131072, 65536, "shared memory")

    class _JIT:
        def run(self, *args, grid = None, warmup = False, **kwargs):
            return _Kernel()

    need = smem.shared_bytes(_JIT(), (torch.zeros(1, device = "cuda:0"),), num_warps = 1)
    assert need == float("inf"), "an unloadable tile must not report a finite footprint"
    assert need > smem.smem_limit(torch.device("cuda:0"))
