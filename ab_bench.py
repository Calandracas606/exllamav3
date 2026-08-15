"""In-process A/B benchmark: alternates the attention fast paths on/off within
one model load, canceling clock/thermal drift between configurations.

Measures windowed steady-state decode tok/s for each config across several
alternations.
"""
import os, sys, time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import exllamav3.modules.attn as attn_mod
import exllamav3.modules.attention_fn.triton_paged as tp_mod
from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler

MP = "/tmp/hf_cache2/models--turboderp--Qwen3.6-27B-exl3/snapshots/04075440a2269126a40e164dbb718700784c39dd"
N = int(os.environ.get("NTOK", "384"))
ROUNDS = int(os.environ.get("ROUNDS", "3"))

config = Config.from_directory(MP)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=4096)
tokenizer = Tokenizer(config)
DEV = os.environ.get("DEVICE", "cuda:0")
model.load(progressbar=False, device=DEV)

import exllamav3.block_graph_rocm as bgr

SKIP_ATTN = {"on": False}
_orig_attn_fwd = None

def set_skip(on: bool):
    global _orig_attn_fwd
    from exllamav3.modules.attn import Attention
    if on:
        if _orig_attn_fwd is None:
            _orig_attn_fwd = Attention.forward
        def skip_fwd(self, x, params, out_dtype=None):
            return torch.zeros_like(x, dtype=self.out_dtype)
        Attention.forward = skip_fwd
    else:
        if _orig_attn_fwd is not None:
            Attention.forward = _orig_attn_fwd


def set_paths(on: bool, trellis_t: bool = False):
    attn_mod._rocm_fused_qkv = on
    tp_mod._paged_fused = on
    attn_mod._rocm_trellis_t = on and trellis_t
    for b in model.modules:
        if type(b).__name__ == "TransformerBlock" and type(getattr(b, "attn", None)).__name__ == "Attention":
            b.attn._rocm_fused = None
            b.attn._rocm_o = None
    mgr = bgr._managers.get(model)
    if mgr is not None:
        mgr.graphs.clear()
        mgr.graph_order.clear()
        mgr.warmups.clear()
        mgr.disabled = False

def bench(on: bool, trellis_t: bool = False) -> float:
    set_paths(on, trellis_t)
    gen = Generator(model, cache, tokenizer)
    ids = tokenizer.encode("Write a detailed essay about the history of computing,", add_bos=True)
    gen.enqueue(Job(input_ids=ids, max_new_tokens=64, sampler=ArgmaxSampler()))
    while gen.num_remaining_jobs():
        gen.iterate()
    torch.cuda.synchronize()
    gen2 = Generator(model, cache, tokenizer)
    gen2.enqueue(Job(input_ids=ids, max_new_tokens=N, sampler=ArgmaxSampler()))
    gen2.iterate()
    torch.cuda.synchronize()
    n = 0
    windows = []
    last = time.perf_counter()
    t_all = time.perf_counter()
    while gen2.num_remaining_jobs() and n < N:
        gen2.iterate()
        n += 1
        if n % 64 == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            windows.append((now - last) / 64)
            last = now
    torch.cuda.synchronize()
    med = sorted(windows)[len(windows) // 2]
    return med * 1000  # ms/token

import statistics as st
if os.environ.get("ABLATION"):
    both = os.environ.get("ABLATION") == "both"
    fast = os.environ.get("ABLATION") != "base"
    set_paths(fast)
    for sk in (False, True):
        set_skip(sk); bench(fast)
    if both:
        set_paths(not fast)
        for sk in (False, True):
            set_skip(sk); bench(not fast)
    res = {}
    combos = [("FAST" if fast else "BASE", False), ("FAST" if fast else "BASE", True)]
    if both:
        combos += [("BASE" if fast else "FAST", False), ("BASE" if fast else "FAST", True)]
    for r in range(ROUNDS):
        for tag, sk in combos:
            set_paths(tag == "FAST"); set_skip(sk)
            res.setdefault((tag, sk), []).append(bench(tag == "FAST"))
    for (tag, sk), v in res.items():
        print(f"{tag:4s} {'no-attn' if sk else 'full   '}: median {st.median(v):.2f}  runs {[f'{x:.2f}' for x in v]}")
    for tag in ("BASE", "FAST"):
        if (tag, False) in res and (tag, True) in res:
            d = st.median(res[(tag, False)]) - st.median(res[(tag, True)])
            print(f"{tag} attention-block cost: {d:.2f} ms/token")
    if both and ("BASE", False) in res and ("FAST", False) in res:
        print(f"FAST-vs-BASE full-model gain: {st.median(res[('BASE', False)]) - st.median(res[('FAST', False)]):.2f} ms/token")
    raise SystemExit(0)

# one throwaway round for clocks (also warms the transposed-trellis variant)
combos = [("FAST", True, False), ("BASE", False, False), ("FASTAT", True, True)]
for tag, on, tt in combos:
    bench(on, tt)

results = {}
for r in range(ROUNDS):
    for tag, on, tt in combos:
        ms = bench(on, tt)
        results.setdefault(tag, []).append(ms)
        print(f"round {r} {tag}: {ms:.2f} ms/token ({1000 / ms:.2f} tok/s)", flush=True)

for tag, _, _ in combos:
    v = results[tag]
    print(f"{tag:6s}: median {st.median(v):.2f} ms/token  (runs: {[f'{x:.2f}' for x in v]})  -> {1000 / st.median(v):.2f} tok/s", flush=True)
