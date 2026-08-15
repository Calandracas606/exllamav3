"""In-situ per-stage GPU timing of one full-attention layer during real decode (27B).

Same methodology as gdn_breakdown.py: replace layer attention forward with an instrumented
copy wrapping each stage with CUDA events on the live stream, whole-step graphs OFF, run a
real generation, accumulate over N decode tokens. The three Triton paged-attn launches
(kv update, split, combine) are timed individually through launcher proxies.
"""
import os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ["EXL3_BLOCK_GRAPHS"] = "0"

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler
from exllamav3.modules.attn import Attention
from exllamav3.modules.attention_fn import attn_dispatch
from exllamav3.util.tensor import get_for_device
from exllamav3.ext import exllamav3_ext as ext
import exllamav3.bc_rocm as bc_rocm
if os.environ.get("SUPPRESS_BC") == "1":
    bc_rocm.capture_suppress = True

MP = "/tmp/hf_cache2/models--turboderp--Qwen3.6-27B-exl3/snapshots/04075440a2269126a40e164dbb718700784c39dd"
NUM_TOKENS = int(os.environ.get("NB", "192"))

config = Config.from_directory(MP)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=int(os.environ.get("CACHE", "4096")))
tokenizer = Tokenizer(config)
model.load(progressbar=False, device="cuda:0")

attn_layers = [b.attn for b in model.modules
               if type(b).__name__ == "TransformerBlock" and isinstance(getattr(b, "attn", None), Attention)]
print(f"full-attn layers: {len(attn_layers)}, layer_idx of first: {attn_layers[0].layer_idx}")
m = attn_layers[0]

N_STAGES = 8
names = ["q_proj (EXL3 GEMV N=12288)", "deinterleave_qg (ext)",
         "k_proj + v_proj (EXL3 GEMV N=1024 x2)",
         "rope (incl. q/k norm, C++)", "attn_dispatch total",
         "gate mul_sigmoid_ + reshape", "o_proj (EXL3 GEMV N=5120)", "(unused)"]
KNAMES = ["kv_update kernel (Triton)", "split kernel (Triton)", "combine kernel (Triton)",
          "had_r_128 kernel (x2 per linear)", "fused dequant-GEMV kernel"]
timings = torch.zeros(len(names))
ktimings = torch.zeros(len(KNAMES))
kcounts = torch.zeros(len(KNAMES))
pending = []
kpending = []
bpending = []
count = 0
armed = {"on": False}

# -- launcher proxies for the three Triton launches -------------------------
import exllamav3.modules.attention_fn.triton_paged as tp
import exllamav3.exl3_gemm_triton as egt

kbuckets = {}   # (M, N) -> [total_ms, count] for the GEMV kernel

class KernelProxy:
    def __init__(self, kern, slot, shape_args=None):
        self.kern = kern
        self.slot = slot
        self.shape_args = shape_args
    def __getitem__(self, grid):
        real = self.kern[grid]
        def launcher(*args, **kwargs):
            if not armed["on"]:
                return real(*args, **kwargs)
            if self.shape_args is not None:
                mi, ni = self.shape_args
                key = (int(args[mi]), int(args[ni]))
                b = kbuckets.setdefault(key, [0.0, 0])
                b[1] += 1
                pre2 = torch.cuda.Event(enable_timing=True)
                post2 = torch.cuda.Event(enable_timing=True)
                pre2.record()
                r = real(*args, **kwargs)
                post2.record()
                bpending.append((pre2, post2, key))
                if len(bpending) >= 64:
                    torch.cuda.synchronize()
                    for pr, po, k_ in bpending:
                        kbuckets[k_][0] += pr.elapsed_time(po)
                    bpending.clear()
                return r
            pre = torch.cuda.Event(enable_timing=True)
            post = torch.cuda.Event(enable_timing=True)
            pre.record()
            real(*args, **kwargs)
            post.record()
            global kpending
            kpending.append((pre, post, self.slot))
            kcounts[self.slot] += 1
            if len(kpending) >= 32:
                torch.cuda.synchronize()
                for pr, po, sl in kpending:
                    ktimings[sl] += pr.elapsed_time(po)
                kpending = []
        return launcher

kern_proxies = {
    "_paged_kv_update_kernel": KernelProxy(tp._paged_kv_update_kernel, 0),
    "_paged_attn_decode_split_kernel": KernelProxy(tp._paged_attn_decode_split_kernel, 1),
    "_paged_attn_decode_combine_kernel": KernelProxy(tp._paged_attn_decode_combine_kernel, 2),
}
# GEMV-path proxies (module-global lookups in egt)
egt_proxies = {
    "_had_r_128_kernel": KernelProxy(egt._had_r_128_kernel, 3),
    "_wrapped_fused_kernel": KernelProxy(egt._wrapped_fused_kernel, 4, shape_args=(4, 5)),  # x, y, trellis, perm_i, M, N, K_dim, ...
}
for name, p_ in egt_proxies.items():
    setattr(egt, name, p_)
for name, p in kern_proxies.items():
    setattr(tp, name, p)

orig_forward = Attention.decode_flash_attn

def instrumented(self, x, bsz, seqlen, params):
    global count
    if self is not m or not armed["on"] or x.shape[:2] != (1, 1):
        return orig_forward(self, x, bsz, seqlen, params)

    evs = [torch.cuda.Event(enable_timing=True) for _ in range(N_STAGES + 1)]

    cache = params.get("cache")
    block_table = get_for_device(params, "block_table", self.device)
    cache_seqlens = get_for_device(params, "cache_seqlens", self.device)
    position = params.get("position", 0)
    positions = get_for_device(params, "positions", self.device, None)
    position_ids = get_for_device(params, "position_ids", self.device, None)
    inv_freq = get_for_device(params, "inv_freq", self.device, None)
    causal = params.get("causal", True)

    i = 0
    evs[0].record()

    # ---- project_qkv: separate Triton EXL3 GEMVs (multi_qg/multi_kv are None on ROCm) ----
    qgh = self.q_proj.forward(x, params)
    evs[1].record()                                              # 0: q proj
    if self.head_dim % 8 == 0 and qgh.dtype == torch.half:
        q = torch.empty((1, 1, self.num_q_heads, self.head_dim), dtype=torch.half, device=qgh.device)
        g = torch.empty((1, 1, self.num_q_heads * self.head_dim), dtype=torch.half, device=qgh.device)
        ext.deinterleave_qg(qgh.view(1, 1, -1, self.head_dim * 2), q, g, self.head_dim)
    else:
        q, g = torch.chunk(qgh.view(1, 1, -1, self.head_dim * 2), 2, dim=-1)
        g = g.reshape(1, 1, -1)
    evs[2].record()                                              # 1: deinterleave
    k = self.k_proj.forward(x, params)
    v = self.v_proj.forward(x, params)
    evs[3].record()                                              # 2: k+v proj

    q = q.view(1, 1, self.num_q_heads, self.head_dim)
    k = k.view(1, 1, self.num_kv_heads, self.head_dim)
    v = v.view(1, 1, self.num_kv_heads, self.head_dim)

    # rope (with fused q/k norm)
    q, k = self.rope.apply(
        q, k, position, positions, position_ids, True,
        self.q_norm_tensor, self.k_norm_tensor, self.norm_eps, self.norm_constant_bias,
        inv_freq, self.post_rope_norm)
    evs[4].record()                                              # 3: rope

    o = attn_dispatch(
        q=q, k=k, v=v, cache=cache, cache_idx=self.layer_idx,
        cache_instance=params.get("layer_instance"), block_table=block_table,
        cache_seqlens=cache_seqlens, causal=causal, sm_scale=self.sm_scale,
        window_size=self.sliding_window, softcap=self.logit_softcapping,
        sinks=self.sinks, dispatch_cache=self.dispatch_cache)
    evs[5].record()                                              # 4: attn total

    o = o.reshape((bsz, seqlen, self.num_q_heads * self.head_dim))
    ext.mul_sigmoid_(o, g)
    evs[6].record()                                              # 5: gate

    x = self.o_proj.forward(o, params)
    evs[7].record()                                              # 6: o_proj

    global pending
    pending.append((evs, None, None))
    count += 1
    if count % 16 == 0:
        evs_list = [it for it in pending if it[1] is None]
        klist = [it for it in pending if it[1] is not None]
        torch.cuda.synchronize()
        for e, _, _ in evs_list:
            for st in range(N_STAGES - 1):
                timings[st] += e[st].elapsed_time(e[st + 1])
        pending = []
    return x

Attention.decode_flash_attn = instrumented

gen = Generator(model, cache, tokenizer)
ids = tokenizer.encode("Write a detailed essay about the history of computing,", add_bos=True)

job = Job(input_ids=ids, max_new_tokens=48, sampler=ArgmaxSampler())
gen.enqueue(job)
while gen.num_remaining_jobs(): gen.iterate()
torch.cuda.synchronize()

armed["on"] = True
job = Job(input_ids=ids, max_new_tokens=NUM_TOKENS, sampler=ArgmaxSampler())
gen.enqueue(job)
while gen.num_remaining_jobs(): gen.iterate()
torch.cuda.synchronize()
armed["on"] = False

Attention.decode_flash_attn = orig_forward
torch.cuda.synchronize()
for item in pending:
    evs = item[0]
    for st in range(N_STAGES - 1):
        timings[st] += evs[st].elapsed_time(evs[st + 1])

for pr, po, sl in kpending:
    ktimings[sl] += pr.elapsed_time(po)
kpending = []
for pr, po, k_ in bpending:
    kbuckets[k_][0] += pr.elapsed_time(po)
bpending = []
print("fused dequant-GEMV by (M, N) [pure kernel time]:")
for k_ in sorted(kbuckets, key=lambda k: -kbuckets[k][0]):
    tot_ms, cnt = kbuckets[k_]
    print(f"  M={k_[0]:5d} N={k_[1]:6d}  calls={cnt:7d}  us/call={tot_ms / max(cnt,1) * 1000:8.1f}")

print(f"\ninstrumented decode steps: {count}")
print(f"cache size: {int(os.environ.get('CACHE', '4096'))} tokens")
print(f"{'stage':36s} {'us/call':>9s} {'x16 ms/token':>12s}")
print("-" * 60)
tot = 0.0
for s, n in enumerate(names):
    us = timings[s].item() / max(count, 1) * 1000
    if s not in (5, 6, 7):
        tot += us
    print(f"{n:36s} {us:9.1f} {us * 16 / 1000:12.2f}")
n_k = int(kcounts.min().item())
for s_i, n in enumerate(KNAMES):
    us = ktimings[s_i].item() / max(n_k, 1) * 1000
    print(f"  . {n:33s} {us:9.1f} {us * 16 / 1000:12.2f}   [{int(kcounts[s_i])} calls]")
print("-" * 60)
print(f"{'TOTAL (per attn layer, stages)':36s} {tot:9.1f} {tot * 16 / 1000:12.2f}")
