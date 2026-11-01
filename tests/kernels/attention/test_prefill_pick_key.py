"""
The prefill pick key must cover every input the shared-memory footprint depends on, and no more.

The footprint is set by the tile and by `is_split`. is_split is not a tile property: prepare() derives
it from `splits`, which follows `programs` (q_blocks * bsz * n_q_heads) and bound_kv
(num_pages_per_seq * page_size + kv_append_len, clamped by max_kv_len). Without a split the kernel is
handed the fp16 q tensor as a placeholder accumulator; with one it gets a real fp32 buffer, and that
tile is twice the size. So one tile has two valid footprints:

    (128, 32, 8, 2), head_dim 256, no split  ->  65536 B
    (128, 32, 8, 2), head_dim 256, split     -> 131072 B   (over a 65536 B device limit)

exllamav3 chunks a long prefill, so one request calls the kernel at several q_len under one module.
If the key ignores q_len, the pick cached for the first chunk is reused for the rest: a 4096 token
chunk with splits=1 caches 65536 B, and the 132 token remainder with splits=4 then launches that tile
at 131072 B, failing in the loader after the prefill has started:

    out of resource: shared memory, Required: 131072, Hardware limit: 65536

The key is deliberately coarse: q_len is quantised the way Triton specialises integers, and the two
integer groups are collapsed to the product and the sum that actually feed the decision. Keying on raw
q_len would still be correct but would probe once per distinct chunk length instead of once per kind.
"""
import pytest
import torch

from exllamav3.modules.attention_fn import smem, triton_paged

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

DEVICE = torch.device("cuda:0")
PAST_TOKENS = 22528
# The tensor-parallel shard shape from the report: 18 q heads / 3, 3 kv heads / 3, head_dim 256
N_Q_HEADS = 6
N_KV_HEADS = 1
HEAD_DIM = 256


_CACHE = {}

def _cache():
    """One paged cache for the whole module, so every call sees the same block-table width and the
    same kv bound -- which is what a served model looks like, and what isolates q_len in the key."""
    if not _CACHE:
        from testlib.attention import rand_paged_cache
        _CACHE["v"] = rand_paged_cache(
            1, PAST_TOKENS + 8192 + 8, N_KV_HEADS, HEAD_DIM, PAST_TOKENS, DEVICE)
    return _CACHE["v"]


def _prefill(q_len, max_kv_len = None):
    k_cache, v_cache, block_table, seqlens = _cache()
    q = torch.randn((1, q_len, N_Q_HEADS, HEAD_DIM), dtype = torch.half, device = DEVICE)
    return triton_paged.paged_attn_triton_prefill(
        q, None, None, k_cache, v_cache, block_table, seqlens, causal = True, max_kv_len = max_kv_len)


def _key_spy(monkeypatch):
    """Capture every pick key the walk is asked about, by reference to the live code path."""
    keys = []
    original = triton_paged.pick_config

    def spy(device, name, key, candidates, probe):
        if name == "paged_attn_prefill":
            keys.append(key)
        return original(device, name, key, candidates, probe)

    monkeypatch.setattr(triton_paged, "pick_config", spy)
    smem._picks.clear()
    return keys


def test_prefill_pick_key_separates_split_from_non_split_chunks(monkeypatch):
    """The failing case: a chunk that measures without a split must not share a pick with one that
    launches with splits=4. These two lengths bucket differently, so they get separate picks."""
    keys = _key_spy(monkeypatch)

    _prefill(4096)   # divisible by 16, the long chunk from the report
    _prefill(132)    # the remainder, not divisible by 16

    assert len(keys) == 2
    assert keys[0] != keys[1], (
        "q_len is missing from the prefill pick key, so a pick measured for one prefill length is "
        f"reused for another: {keys[0]}")


def test_prefill_pick_key_is_no_finer_than_triton_specialisation(monkeypatch):
    """Several lengths in one specialisation bucket share a pick, so a chunked prefill probes a
    couple of times rather than once per chunk."""
    keys = _key_spy(monkeypatch)

    for q_len in (1024, 2048, 4096):
        _prefill(q_len)

    assert len(keys) == 3
    assert keys[0] == keys[1] == keys[2], (
        "lengths that specialise identically should share one pick, else each chunk re-probes: "
        f"{keys}")


def test_prefill_pick_key_covers_the_kv_bound(monkeypatch):
    """max_kv_len clamps bound_kv in prepare(), so it has to reach the key: without it two calls
    with the same q_len can land on different is_split values."""
    keys = _key_spy(monkeypatch)

    _prefill(4096, max_kv_len = 1024)
    _prefill(4096, max_kv_len = None)

    assert len(keys) == 2
    assert keys[0] != keys[1], (
        "max_kv_len changes bound_kv and therefore is_split, but is missing from the key: "
        f"{keys[0]}")
