"""Argmax generation identity: fused q/k/v + direct o_proj attention path vs fallback.

Toggles the ROCm attention fast path off by clearing the module flag (the env
var gates it at import; flipping exllamav3.modules.attn._rocm_fused_qkv plus
the cached eligibility dicts reproduces the fallback exactly). Graphs stay ON
both runs (the whole-step graph re-captures when the executed path changes
because the capture key includes nothing about it -- so we reset the graph
state by deleting the module attr the same way the env would).
"""
import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL = sys.argv[1] if len(sys.argv) > 1 else "27b"
N = int(os.environ.get("N", "300"))
PROMPTS = [
    "The capital of France is",
    "My grandfather used to tell me stories about",
    "In the year 2525, humanity finally solved",
]

MODELS = {
    "9b": "/tmp/hf_cache/models--turboderp--Qwen3.5-9B-exl3/snapshots/01192d8c6d9cbd94f9cf99c0ddb85e9e217ccc01",
    "27b": "/tmp/hf_cache2/models--turboderp--Qwen3.6-27B-exl3/snapshots/04075440a2269126a40e164dbb718700784c39dd",
}

import exllamav3.modules.attn as attn_mod
import exllamav3.modules.attention_fn.triton_paged as tp_mod
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
from exllamav3.modules.attn import Attention

config = Config.from_directory(MODELS[MODEL])
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=4096)
tokenizer = Tokenizer(config)
model.load(progressbar=False, device="cuda:0")

attn_layers = [b.attn for b in model.modules
               if type(b).__name__ == "TransformerBlock" and isinstance(getattr(b, "attn", None), Attention)]
assert attn_layers
print(f"{len(attn_layers)} full-attention layers, fused path armed = {attn_mod._rocm_fused_qkv}")

# whole-step graph state must be reset between runs so the second run re-captures
import exllamav3.block_graph_rocm as bgr


def reset_graphs():
    mgr = bgr._managers.get(model)
    if mgr is not None:
        mgr.graphs.clear()
        mgr.graph_order.clear()
        mgr.warmups.clear()
        mgr.disabled = False


def run(prompt):
    reset_graphs()
    gen = Generator(model, cache, tokenizer)
    ids = tokenizer.encode(prompt, add_bos=True)
    job = Job(input_ids=ids, max_new_tokens=N, sampler=ArgmaxSampler())
    gen.enqueue(job)
    while gen.num_remaining_jobs():
        gen.iterate()
    torch.cuda.synchronize()
    return job.sequences[0].sequence_ids.torch()[0].tolist()


all_ok = True
for prompt in PROMPTS:
    seq_new = run(prompt)
    # force fallback (both attention fast paths off)
    attn_mod._rocm_fused_qkv = False
    tp_mod._paged_fused = False
    for m in attn_layers:
        m._rocm_fused = None
        m._rocm_o = None
    seq_old = run(prompt)
    tp_mod._paged_fused = bool(getattr(torch.version, "hip", None))
    attn_mod._rocm_fused_qkv = torch.version.hip and os.environ.get("EXL3_ATTN_QKV_FUSED", "1") != "0"
    for m in attn_layers:
        m._rocm_fused = None
        m._rocm_o = None

    identical = seq_new == seq_old
    n_match = 0
    for a, b in zip(seq_new, seq_old):
        if a != b:
            break
        n_match += 1
    print(f"[{prompt[:30]!r:34s}] tokens {len(seq_new)}/{len(seq_old)}  identical prefix {n_match}  "
          f"{'IDENTICAL' if identical else 'DIVERGED'}")
    if not identical:
        all_ok = False
        for i, (a, b) in enumerate(zip(seq_new, seq_old)):
            if a != b:
                print(f"   first diff at {i}: {a} vs {b}")
                print(f"   new: {tokenizer.decode(torch.tensor([seq_new[max(0, i - 10):i + 10]]))}")
                print(f"   old: {tokenizer.decode(torch.tensor([seq_old[max(0, i - 10):i + 10]]))}")
                break

print("ALL IDENTICAL" if all_ok else "DIVERGENCE DETECTED")
