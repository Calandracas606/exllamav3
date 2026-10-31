"""
The prefill pick key must cover every launch input the shared-memory footprint depends on.

The footprint is set by the tile and by `is_split`. is_split is not a tile property: prepare() derives
it from `splits`, which is chosen from `programs` (q_len, bsz, head count) and the kv bound. Without
a split the kernel is handed the fp16 q tensor as a placeholder accumulator; with one it gets a real
fp32 buffer. The fp32 accumulator tile is twice the size, so the same tile measures 65536 B in one
case and 131072 B in the other:

    (128, 32, 8, 2), head_dim 256, no split  ->  65536 B
    (128, 32, 8, 2), head_dim 256, split     -> 131072 B   (over a 65536 B device limit)

exllamav3 prefill chunks a long prompt, so one request produces several q_len values under one
transformer module. If q_len is not in the pick key, the pick cached for the first chunk is reused
for the rest. A 4096 token chunk with splits=1 measures 65536 B, gets cached, and the 132 token
remainder with splits=4 then launches that tile at 131072 B -- which fails in the loader, after the
prefill has started, taking the request with it:

    out of resource: shared memory, Required: 131072, Hardware limit: 65536
"""
import pytest
import torch

from exllamav3.modules.attention_fn import smem, triton_paged

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason = "needs a GPU")

DEVICE = torch.device("cuda:0")
PAST = 22528
# The tensor-parallel shard shape from the report: 18 q heads / 3, 3 kv heads / 3, head_dim 256
N_Q_HEADS = 6
N_KV_HEADS = 1
HEAD_DIM = 256


def _prefill(q_len):
    from testlib.attention import rand_paged_cache

    torch.manual_seed(0)
    k_cache, v_cache, block_table, seqlens = rand_paged_cache(
        1, PAST + q_len + 8, N_KV_HEADS, HEAD_DIM, PAST, DEVICE)
    q = torch.randn((1, q_len, N_Q_HEADS, HEAD_DIM), dtype = torch.half, device = DEVICE)
    return triton_paged.paged_attn_triton_prefill(
        q, None, None, k_cache, v_cache, block_table, seqlens, causal = True)


def _pick_keys():
    """Every key the walk was asked about during one call."""
    keys = []
    original = triton_paged.pick_config

    def spy(device, name, key, candidates, probe):
        if name == "paged_attn_prefill":
            keys.append(key)
        return original(device, name, key, candidates, probe)

    return keys, spy


def test_q_len_is_part_of_the_pick_key(monkeypatch):
    """A 4096 token chunk and a 132 token chunk must not share one pick: their footprints differ."""
    keys, spy = _pick_keys()
    monkeypatch.setattr(triton_paged, "pick_config", spy)
    smem._picks.clear()
    smem._probe_log.clear()

    _prefill(4096)
    _prefill(132)

    assert len(keys) == 2
    assert keys[0] != keys[1], (
        "q_len is missing from the pick key, so a pick measured for one prefill length is reused "
        f"for another: {keys[0]}")
