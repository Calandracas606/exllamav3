"""
A launch-time OutOfResources must say how its config was chosen, not only what the loader wanted.

The loader's message names the footprint it refused. It cannot say whether the tile walk measured
that config as fitting, never measured it, or measured something else -- and those three cases have
different causes. The note attached at the launch site separates them, which is what makes a crash
report from hardware we do not have actionable.

The exception itself is left alone: type, args and message stay exactly as Triton raised them, so
existing handling (tabbyAPI forwards OutOfResources to the request) keeps working.
"""
import pytest
import torch

from exllamav3.modules.attention_fn import smem
from triton.runtime.errors import OutOfResources

DEVICE = torch.device("cuda:0")


@pytest.fixture(autouse = True)
def clean_log():
    smem._probe_log.clear()
    yield
    smem._probe_log.clear()


def _err():
    return OutOfResources(131072, 65536, "shared memory")


def test_exception_type_and_message_are_preserved():
    """The note must not disturb what callers already match on."""
    e = _err()
    before = (type(e), e.args)
    smem._note_probe_miss(e, "paged_attn_prefill", ("K",), DEVICE, (128, 32, 8, 2))
    assert (type(e), e.args) == before
    assert isinstance(e, OutOfResources)


def test_notes_a_config_the_walk_measured_as_fitting():
    """The picked config was measured to fit, so probe and load disagree for it."""
    smem._probe_log[("paged_attn_prefill", ("K",))] = [((128, 32, 8, 2), 65536)]
    e = _err()
    assert smem._note_probe_miss(e, "paged_attn_prefill", ("K",), DEVICE, (128, 32, 8, 2))
    note = "\n".join(e.__notes__)
    assert "smem-probe-miss" in note
    assert "(128, 32, 8, 2): 65536 B" in note
    assert "disagree" in note


def test_notes_a_config_the_walk_never_probed():
    """The config was not among the walk's candidates, so the pick did not come from this walk."""
    smem._probe_log[("paged_attn_prefill", ("K",))] = [((64, 32, 4, 2), 65536)]
    e = _err()
    smem._note_probe_miss(e, "paged_attn_prefill", ("K",), DEVICE, (128, 32, 8, 2))
    note = "\n".join(e.__notes__)
    assert "was never probed" in note
    assert "(64, 32, 4, 2)" in note


def test_notes_a_missing_record():
    """No record for the key means the config came from the cache, or the key changed."""
    e = _err()
    smem._note_probe_miss(e, "paged_attn_prefill", ("K",), DEVICE, (128, 32, 8, 2))
    note = "\n".join(e.__notes__)
    assert "recorded nothing" in note


def test_records_the_walk_when_a_config_is_picked(monkeypatch):
    """pick_config must record every candidate it probed, so a later launch can be explained."""
    limit = smem.smem_limit(DEVICE)
    monkeypatch.setattr(smem, "_limit", {torch.cuda.current_device(): limit})
    smem._picks.clear()

    def probe(cfg):
        return limit * 2 if cfg == "big" else limit // 2

    picked = smem.pick_config(DEVICE, "test_note_walk", ("rec",), ["big", "small"], probe)
    assert picked == "small"
    assert smem._get_probe_log("test_note_walk", ("rec",)) == [("big", limit * 2), ("small", limit // 2)]


def test_prefill_launch_notes_an_out_of_resources(monkeypatch):
    """The wrapped launch must annotate a real OutOfResources raised by the kernel and re-raise the
    same object, with the walk attached. Drives the except branch in paged_attn_triton_prefill."""
    from exllamav3.modules.attention_fn import triton_paged

    if not torch.cuda.is_available():
        pytest.skip("needs a GPU")

    from exllamav3.constants import PAGE_SIZE
    from testlib.attention import rand_paged_cache

    limit = smem.smem_limit(DEVICE)
    torch.manual_seed(0)
    kc, vc, bt, sl = rand_paged_cache(1, 320, 2, 64, 256, DEVICE)
    q = torch.randn((1, 64, 8, 64), dtype = torch.half, device = DEVICE)

    raised = OutOfResources(limit * 2, limit, "shared memory")

    class _Meta:
        shared = 1024

    class _Compiled:
        metadata = _Meta()

    class _Kernel:
        """Enough of a JIT kernel for the walk to pick a config, then fail at the launch itself."""
        def run(self, *a, **kw):
            return _Compiled()

        def __getitem__(self, grid):
            def _launch(*a, **kw):
                raise raised
            return _launch

    monkeypatch.setattr(triton_paged, "_paged_attn_prefill_kernel", _Kernel())
    smem._picks.clear()
    smem._probe_log.clear()

    with pytest.raises(OutOfResources) as caught:
        triton_paged.paged_attn_triton_prefill(
            q, None, None, kc, vc, bt, sl, causal = True,
            block_m = 64, block_n = 32, num_warps = 4, num_stages = 2,
        )
    # Same exception object, so callers matching on type/args still see Triton's own error
    assert caught.value is raised
    assert any("smem-probe-miss" in n for n in caught.value.__notes__)
