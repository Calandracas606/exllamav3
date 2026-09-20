# Causal Profiling Report — exllamav3 ROCm decode (token generation)

**Scope:** decode-path causal attribution on AMD RX 7900 XTX (gfx1100, ROCm 7.14, torch 2.12+rocm, triton 3.7.1), fork @ `95df418` (integration tip, all ROCm work merged). READ-ONLY investigation; all experiments done via untracked scratch scripts under `profiling/` that monkeypatch at runtime.
**Target models:** `/tmp/Qwen3.5-9B-exl3-4.00bpw` (primary), `/tmp/Qwen3.8-27B-exl3-4.00bpw` (validation). End goal framing: beat llama.cpp 4-bit at 33 tok/s on 27B (currently ~17.5).

---

## 1. Executive summary — ranked optimization targets

Numbers are fresh measurements (this session, ctx=512, bsz=1, steady state, in-process cuda-event timing, whole-step graphs ON unless stated). "Causal share" = whole-step wall-time delta when the block is ablated to a no-op (upper bound on the block's serial contribution).

### 27B (57.19 ms/token = 17.5 tok/s baseline)

| # | Target | Causal share (measured) | Why it pays | Estimated ceiling |
|---|--------|------------------------|-------------|-------------------|
| 1 | **MLP GEMV streams** (up/gate/down, b4 trellis) | **29.81 ms/token (52.1%)** | 8.70 GB/token at only **292 GB/s** in-situ; standalone-hot kernel rates are 500-630 GB/s (AGENTS prior), lm_head demonstrates 411-551 GB/s in-situ in this very model | At 500 GB/s: 17.4 ms → saves **~12.4 ms** → ~23.6 tok/s by itself |
| 2 | **Non-GEMV residual** (Python-fallback norms, GDN glue kernels, sampler/argmax, embed staging, gaps) | **8.92 ms/token (15.6%)** (57.19 − 29.81 − 12.54 − 3.60 − 2.32) | 64 block-norm + 48 gated-RMSNorm calls run through `ext_fallbacks.py` pure-PyTorch (6-8 tiny kernels each); 9B stage table measures block norms alone at **3.40 ms/token (19%)** | Fusing norms into neighbors + trimming glue: −5 to −7 ms → ~20 tok/s by itself |
| 3 | **GDN non-projection stages** (recurrent kernel, conv1d, gated norm, transposes) | 12.54 − 4.90(qkv) − ~5.1(z/o) ≈ **2.5 ms**, plus z/o projections running at ~200-260 GB/s | z (756 MB) + o (790 MB) at 500 GB/s would take 3.1 ms vs ~5.1 today | Combined GDN block −4 ms → ~19 tok/s |
| 4 | **Full-attention block** (16 layers: q/k/v/o + paged attn kernels + rope) | **3.60 ms/token (6.3%)** | Matches AGENTS prior (4.36 ms at longer ctx); attention weights+KV ~1.0 GB at 450+ GB/s → ~2 ms | −1.6 ms → ~18.5 tok/s |
| 5 | **lm_head (6-bit GEMV)** | **2.32 ms/token (4.1%)** | Already the *best* in-situ rate (411 GB/s 27B / 551 GB/s 9B); FlyDSL standalone hit 770 GB/s (AGENTS prior) | −0.5..−1 ms at most |

**What 33 tok/s requires (§5):** 13.45 GB/token / 30.3 ms = **444 GB/s sustained whole-model DRAM** vs 235 measured. No single lever reaches it. Two combined scenarios computed: (a) all GEMV streams at 500 GB/s in-situ + residual overhead halved to 4.5 ms → **29.5 tok/s**; (b) 550 GB/s + 3 ms overhead → **34.0 tok/s**. I.e. the goal needs *both* a ~1.7-1.9× GEMV bandwidth improvement *and* a ~2× cut of non-GEMV overhead. The GEMV rate is the primary lever; the overhead reduction is the secondary, cheaper-looking one (the kernels exist in C++ upstream; they are only excluded from the ROCm build).

### 9B (17.62 ms/token = 56.8 tok/s baseline) — same ordering

MLP 7.22 ms (41.0%) > GDN 4.24 ms (24.1%) > block-norms+misc residual ≈ 3.93 ms (22%) > lm_head 1.39 ms (7.9%) > full-attn 0.84 ms (4.7%).

---

## 2. Methodology

- **Worktree:** detached HEAD at `95df418`; scratch scripts in `profiling/` (untracked). Python via the main tree venv (`$MAIN/.venv`), JIT ext cache hit (~23 s build).
- **Protocol** (per AGENTS.md §GPU REALITY): 4-5 s dummy-matmul clock ramp before every timed window; 24-48 warmup decode steps (settles Triton autotuners, per-linear BC graphs, and the whole-step graph capture); timing via bracketed `torch.cuda.Event`s over 40-60 steps, 3 rounds per config, median reported. Cross-run base drift measured at ±0.4 ms on 9B (~2%, within the ±1.5% prior for in-session comparisons).
- **Validity guards learned the hard way** (documented for reproducibility):
  - Monkeypatching module forwards *after* a whole-step graph is captured does nothing (the replay bypasses Python). Correct protocol: install the patch **before first capture**, then warm up (graph captures the patched pipeline). Implemented in `profiling/ablate_child2.py` (subprocess per config, timeout-guarded).
  - Decoding at `past_len>0` against KV pages that were never allocated/written → `hipErrorIllegalAddress` in the paged kernels. Always prefill first.
  - **Mid-run graph recapture (clearing `block_graph_rocm` manager state while a graph exists) livelocks the GPU** (GFX 100%, zero memory traffic, unkillable until process death). Do graph-state resets only in fresh subprocesses.
  - Raw `model.forward` decode loops work on 9B but faulted on 27B inside whole-step-graph replay; the Generator-driven flow (as in `bench_decode.py`) is the proven 27B harness. All 27B numbers here use it.
- **Interleaving:** the subprocess driver (`profiling/driver_ablate.py`) re-ran the unpatched baseline between every ablation config; deltas are quoted against the median of all baselines (17.615 ms 9B / 57.19 ms 27B).
- **Ablation = upper bound:** replacing a stage with zeros/identity removes its GPU work *and* changes downstream values (garbage outputs); timing structure is what's measured. Deltas for independent stages sum to ≈ the parent block within ~5-10% (checked: qkv+z+o+conv+recurrent+norm ≈ 4.6 vs no_GDN 4.24 on 9B — slight overlap expected since removing a projection also removes its input-wait).
- **Profiler tools:** `rocprofiler-systems` is **not on PyPI**; installed from the AMD ROCm index as `rocm[profiler]` extra (package `rocm-profiler==7.14.0`, provides `rocprof-sys-run` CLI). Not ultimately needed: kernel-level attribution was obtained via stage timing + ablations + toggles, which are the trustworthy methods on this box (AGENTS: torch.profiler under-counts device time on ROCm; private-graph microbenches unsound). `/opt/rocm/bin/rocprofv3` available as fallback; not exercised.
- **Weights-per-token accounting** from safetensors headers (`profiling/weight_bytes.py`) — exact, no GPU.

Scripts → tables mapping: `baseline_9b.py` → §3; `driver_ablate.py`+`ablate_child2.py` → §4 (9B/27B ablations); `causal_v2.py --mode stages` → §4 stage table (graphs off); `weight_bytes.py` → §5; `probe_ext_surface.py`, `trace_probe2.py` → §6.

---

## 3. Baselines (ctx=512, bsz=1, default env unless noted)

| Model | Config | ms/token (2 runs) | tok/s | Notes |
|---|---|---|---|---|
| 9B-4bpw | default (graphs ON) | 17.56 / 17.58 | **56.9 / 56.9** | all toggles at default: QKV_FUSED=1, PAGED_FUSED=1, BA_GEMV=1, BC_ATTN auto(=0 on ROCm) |
| 9B-4bpw | `EXL3_PREFER_TRITON_LINEAR=1` | 17.46 / 17.50 | 57.3 / 57.1 | **parity** (−0.6%, within noise) |
| 9B-4bpw | `EXL3_BLOCK_GRAPHS=0` | 21.79 / 21.57 | 45.9 / 46.4 | +4.1 ms (+23%); per-linear BC graphs still active |
| 27B-4bpw | default (graphs ON) | 56.50 / 56.44 | **17.7 / 17.7** | subprocess-child reruns: 57.19/56.42 medians — session consistent |
| 27B-4bpw | `EXL3_PREFER_TRITON_LINEAR=1` | 56.41 / 56.71 | 17.7 / 17.6 | **parity** |
| 27B-4bpw | `EXL3_BLOCK_GRAPHS=0` | 61.07 / 60.02 | 16.4 / 16.7 | +4.0 ms (+7%) |

**Which paths were active (verified by code + BC-attn trace):** decode uses per-linear `bc_rocm.BC_LinearEXL3` graphs inside the whole-step `block_graph_rocm` graph; fused qkv GEMV (`attn_rocm_kernels`), fused paged decode kernel, fused GDN b/a Triton GEMV (`gdn_ba_gemm`), native C++ `cuda_recurrent_gated_delta_rule` / `cuda_causal_conv1d_update`; `gated_rms_norm`/`rms_norm`/`silu_mul` etc. via Python fallbacks. `EXL3_BC_ATTN` path: **DECLINED on all 8 full-attn layers even when forced =1** (`EXL3_BC_ATTN_TRACE=1` printed `-- BC-attn: DECLINED layer {3,7,11,...}`) — see §6.1.

**Surprise vs priors:** `EXL3_PREFER_TRITON_LINEAR=1` is a **no-op at 4 bpw with graphs on** — the default path already dispatches to the same Triton kernels via `BC_LinearEXL3`; the env only changes the dispatch entry point. (AGENTS' 38-47 tok/s 9B per-bitwidth numbers were a different comparison — across bitrates, not this A/B.)

---

## 4. Causal attribution

### 4.1 Block-level ablations (graphs ON, patch-before-capture; delta vs interleaved baseline)

**9B** (base 17.615 ms; scripts: `driver_ablate.py`, `ablate_child2.py`):

| Block | ms/token | share | implied effective BW (weights/token) |
|---|---|---|---|
| MLP (32 layers) | **7.22** | **41.0%** | 2416 MB → 335 GB/s (down 233, up 371, gate ~500) |
| GDN layers (24) | **4.24** | **24.1%** | qkv 403 MB→317, z 202→205, o 202→197 GB/s; + conv/recurrent/norm |
| ‖ block norms + misc residual | ≈3.93 | ≈22% | (§4.2: block norms alone 3.40 ms graphs-off) |
| lm_head (b6) | **1.39** | **7.9%** | 763 MB → **551 GB/s** (best in model) |
| full-attn layers (8) | **0.84** | **4.7%** | q 134 MB→514; k/v latency-bound |

GDN sub-stages (9B): qkv 1.27 · z 0.98 · o 1.02 · conv1d 0.33 · recurrent kernel 0.44 (17.615−17.172) · gated norm ≈0.6 (17.615−17.00) · b/a GEMV ≈0.0 (17.694 vs 17.64 local base — the fused Triton kernel is essentially free; it replaced two ~50-90 µs rocBLAS GEMVs, the AGENTS ~9 ms/token-on-48-layers win is already banked).

MLP sub-stages (9B): down_proj **3.45** · up_proj 2.17 (gate_proj implied ≈1.6 with silu_mul). Full-attn sub-stages: q 0.26 · k 0.17 · v ~0.02 · o ~0.02 (k/v/o GEMVs are latency-bound tiny-N; their cost sits in the fused-QKV/paged kernels and gaps).

**27B validation** (base 57.19 ms; `ablate_child2.py`, Generator flow):

| Block | ms/token | share | effective BW |
|---|---|---|---|
| MLP (64 layers) | **29.81** | **52.1%** | 8699 MB → **292 GB/s** |
| GDN layers (48) | **12.54** | **21.9%** | qkv 1260 MB → 257 GB/s; z+o ≈5.1 ms |
| full-attn (16) | **3.60** | **6.3%** | (AGENTS prior 4.36 ms at larger ctx — consistent) |
| lm_head (b6) | **2.32** | **4.1%** | 954 MB → 411 GB/s |
| residual (norms+glue+ sampler+embed+gaps) | **8.92** | **15.6%** | see §4.2 |

The 9B ranking transfers to 27B unchanged; MLP share *grows* with model size (52%).

### 4.2 Stage timing, graphs OFF (9B, cuda-event pairs around module forwards; `causal_v2.py`)

Wall 26.68 ms/token (vs 17.62 graphs-on ⇒ **whole-step graphs recover 9.1 ms of dispatch** at 9B scale):

| Stage | ms/token |
|---|---|
| MLP_LAYER | 10.94 (down 3.96 · up 2.49 · gate+silu ≈4.5 incl. python `silu_mul`) |
| GDN_LAYER | 8.42 (o 2.24 · z 1.75 · qkv 1.59 · norm 1.29 · recurrent+conv ≈1.5) |
| BLOCK_NORM_MLP + BLOCK_NORM_ATTN | **3.40** (64 × Python-fallback `rms_norm`) |
| FATTN_LAYER | 1.71 |
| LM_HEAD | 1.11 |
| EMBED + FINAL_NORM | 0.10 |

Sum of stages ≈ 24.7 of 26.7 wall → ~2 ms CPU-side gaps remain even graphs-off (launch-bound tails). Note stage times here include per-stage dispatch bubbles; they over-count pure GPU time but expose the Python-fallback norm cost (3.4 ms) that the ablation table attributes to the "residual".

### 4.3 Path toggles (realized value of existing optimizations; in-situ A/B)

| Toggle | 9B Δms | 27B Δms | Reading |
|---|---|---|---|
| `EXL3_BLOCK_GRAPHS=0` | +4.14 (+23%) | +4.03 (+7%) | whole-step graphs worth ~4 ms/token over per-linear-graphs-only; AGENTS prior (28→0.2 ms dispatch) mostly banked by per-linear BC + whole-step graphs together |
| `EXL3_PREFER_TRITON_LINEAR=1` | −0.10 (noise) | −0.03 (noise) | parity on decode at 4 bpw (same kernels) |
| `EXL3_BC_ATTN=1` (forced) | n/a — DECLINED | n/a | §6.1: C++ BC_Attention unavailable on ROCm; Python whole-step graph covers it |
| `EXL3_GDN_BA_GEMV` | ablation ≈ 0.0 | — | kernel is one tiny launch; its win (vs rocBLAS b/a GEMVs) is already in the baseline |
| `EXL3_ATTN_QKV_FUSED`, `EXL3_PAGED_FUSED` | not re-toggled this session | — | AGENTS priors stand (attention 5.28→4.36 ms/token; interleaved A/B) |

### 4.4 Gain curves (27B; g = T/(T − t(1−1/f)), T = 57.19 ms; validated against the measured ablation point at f→∞)

| Block (t) | 1.5× | 2× | 3× | ∞ (measured ablation) |
|---|---|---|---|---|
| MLP (29.81) | 21.2 tok/s | 23.6 | 26.8 | 39.5 |
| GDN (12.54) | 18.9 | 19.6 | 20.5 | 22.4 |
| residual (8.92) | 18.4 | 19.0 | 19.5 | 20.6 |
| FATTN (3.60) | 17.9 | 18.0 | 18.2 | 18.6 |
| lm_head (2.32) | 17.7 | 17.8 | 18.0 | 18.2 |

MLP-2× + residual-2× + GDN-2× combined (serial, additive): 57.19 − 14.9 − 4.5 − 6.3 = 31.5 ms ≈ **31.7 tok/s** — i.e. 33 needs slightly more than "2× everything non-attention".

---

## 5. Roofline to 33 tok/s

Per-token DRAM stream (27B, from safetensors; embed/vision on CPU excluded): MLP 8.70 GB · GDN projections 2.81 GB · attn projections 0.84 GB · lm_head 0.95 GB · KV ~0.04 GB · small tensors ~0.01 GB = **13.45 GB/token**.

- Current: 13.45 GB / 57.19 ms = **235 GB/s** whole-model (matches AGENTS prior 235).
- 33 tok/s (30.3 ms) ⇒ **444 GB/s sustained** — every stream, every token, including norms/glue.
- Reference rates: b4 GEMV standalone-hot 500-630 GB/s (AGENTS); b6 standalone 738 (Triton) / 770 (FlyDSL); **best in-situ measured here: lm_head 411 (27B) / 551 (9B) GB/s**; in-situ b4 GEMVs currently 200-370 GB/s. AGENTS notes in-situ roughly halves standalone rates under the 13.6 GB/token stream pressure — the whole gap to 33 tok/s is exactly this in-situ factor.

Required per-bucket rates if non-GEMV residual is cut to R ms: GEMV streams (12.4 GB) must sustain 12.4·1000/(30.3−R) GB/s → R=9: 627 · R=6: 549 · R=4: 508 · R=3: 489 GB/s. Combined scenarios (§1): 500 GB/s + 4.5 ms residual ⇒ 29.5 tok/s; **550 GB/s + 3 ms ⇒ 34.0 tok/s**.

**Conclusion:** 33 tok/s is *plausible but tight*: it requires lifting in-situ b4 GEMV bandwidth ~1.7-1.9× (toward what the 6-bit lm_head stream already achieves, and toward FlyDSL's demonstrated 770 GB/s standalone) *and* compressing the ~9 ms non-GEMV residual to ~3 ms. Levers ranked by leverage-per-effort: (1) MLP GEMV rate, (2) Python-fallback norm fusion (C++ kernels exist upstream — they're just excluded from the ROCm build), (3) GDN z/o GEMV rate + glue fusion, (4) attention block, (5) lm_head (near ceiling).

---

## 6. CUDA-path techniques not used on ROCm

### 6.1 Triton-kernel extraction for C++ launch/capture (`bc_attn.py`)

- `exllamav3/modules/attention_fn/bc_attn.py:59-76`: `_compile_kernel()` does `triton.compile(ASTSource(...))` → `ck.asm["cubin"]` → `ext.TritonKernel(cubin, name, num_warps, shared)` so attention kernels launch **from C++ inside captured graphs**.
- On ROCm the compiled artifact is an **hsaco** (`ck.asm["hsaco"]`), not a cubin — `bc_attn.py` would need `asm["hsaco"]`.
- `ext.TritonKernel` **does not exist on the ROCm build**: the pybind registration lives in `exllamav3/exllamav3_ext/libtorch/attention_bc.h:1`, which is included only in the `#if !defined(USE_ROCM)` branch of `bindings.cpp:239`. *However*, a fully hipified implementation exists and compiles: `triton_kernel_hip.cpp` loads via `hipModuleLoadData`/`CudaDrv` (format-agnostic `module_load_data`) — binding it on ROCm is a small build-config change, not a port.
- Does bc_attn's path activate on ROCm today? **No.** `ext.py:229-234` sets `EXL3_BC_ATTN=0` (build would crash in `_compile_kernel` — no `TritonKernel`). Forcing `EXL3_BC_ATTN=1` + `EXL3_BC_ATTN_TRACE=1` still prints `BC-attn: DECLINED` for every full-attn layer (empirically verified, 8/8 layers on 9B; decline from `_module_eligible`, plausibly the NoPE/q_norm clause at `bc_attn.py:372` for this arch's mRoPE setup).
- What covers it instead: `exllamav3/block_graph_rocm.py` captures the **entire decode step** (embedding → LM head) as one `torch.cuda.CUDAGraph` from Python, with dynamic values staged through persistent device buffers (`refresh_inputs`, `block_graph_rocm.py:133-167`). For steady-state bsz=1 decode this is equivalent coverage to BC_Attention + per-module BC graphs: the kernels in the replay are the same Triton kernels the dispatch path JITs.

### 6.2 BC_* C++ whole-block API surface on ROCm

Registration sites: `libtorch/*_bc.h` includes at `bindings.cpp:239-248` — all inside the CUDA-only branch. On ROCm, `ext.py:182-191` replaces every missing class with the `_BCNone` stub (callable → `None`), then installs real Python replacements for the two linear classes from `bc_rocm.py`.

| ext symbol | Used in Qwen3.5 decode? | CUDA path | ROCm status | Measured relevance |
|---|---|---|---|---|
| `BC_LinearEXL3` | **yes — every EXL3 linear** | C++ graph-captured GEMV | **Python graph-capture** (`bc_rocm.BC_LinearEXL3`) | covers all GEMV launches inside whole-step graph; §4 |
| `BC_LinearFP16` | rarely (fp16 linears; b/a proj pre-merge) | C++ | Python passthrough (`bc_rocm.BC_LinearFP16` = plain matmul) | b/a merged into GEMV by `gdn_ba_gemm` instead |
| `BC_GatedDeltaNet` | no (fused-proj variant; Qwen3-Next style) | C++ whole-layer graph | **missing** (stub→None) → torch path | n/a for Qwen3.5 |
| `BC_GatedDeltaNetSplit` | **yes — split-proj GDN layers** (would be) | C++ whole-layer graph incl. conv+recurrent+norm | **missing** → Python stage-by-stage (`gated_delta_net.py:878-965`) | this is the GDN 12.5 ms block; stages run as separate kernels + Python norms |
| `BC_Mamba2` | no | C++ | missing (stub) | n/a |
| `BC_MLP` / `BC_GatedMLP` | **yes — MLP layers** (would be) | C++ graph incl. fused silu·gate epilogue | **missing** → Python: up/gate/down linears + Python `silu_mul` | MLP 29.8 ms block; gate+silu ≈1.6 ms/token of it (9B) |
| `BC_BlockSparseMLP` | no (MoE archs) | C++ | missing | n/a |
| `BC_Attention` | **yes — full-attn layers** (would be) | C++ graph + TritonKernel hsaco→cubin | **missing** (declined; §6.1) | FATTN 3.6 ms block |
| `BC_GatedRMSNorm` | **yes — GDN norm** (24/48×) | C++ fused | **missing** → `ext_fallbacks.gated_rms_norm` (pure PyTorch, 6-8 kernels) | ~0.6 ms/token (9B) |
| `BC_DSV4*`, `BC_MLAttention`, `BC_SAM` | no (DSV4/MLA archs) | C++ | missing | n/a |
| `gated_delta_net_fused_op_3` | (upstream CUDA helper) | C++ | **no Python binding at all** | fork replaced with Triton `gdn_ba_gemm.py` (AGENTS prior; ablation shows ≈0 cost now) |

### 6.3 ext.py fallback dispatch during decode (per token)

Probe (`profiling/probe_ext_surface.py`) + call-site analysis:

| Fallback | Calls/token (9B) | Est. cost/token | Notes |
|---|---|---|---|
| `rms_norm` (+`rms_norm_res_in`) | 64 | **≈3.4 ms** (§4.2 stage table) | each = float copy + pow + mean + rsqrt + 2 muls + copy ≈ 7 kernels; pure launch/latency bound (8 KB tensors) |
| `gated_rms_norm` | 24 | ≈0.6 ms | wider (gate + silu path) |
| `silu_mul` (via `GatedMLP`) | 32 | ≈1-1.6 ms (inside MLP gate figure) | `F.silu(x)*y` + `z.copy_` |
| `mul_sigmoid_broadcast_` / `mul_sigmoid_` | 8 | <0.1 ms | attention output gates |
| `deinterleave_qg` | 0 (fused into ROCm qkv kernel) | 0 | eliminated by `attn_rocm_kernels` |
| `softcap`, `gelu_mul`, … | 0 | 0 | arch-dependent |
| `FUSED_SAMPLER` | forced off (`ext.py:227`) | — | argmax on GPU + `.cpu()` sync per token (~0.1-0.3 ms incl. sync; inside "residual") |
| `exl3_mgemm`/`exl3_gemm`/`exl3_gemv` | 0 in decode (mgemm path needs `multi_qg`; missing on ROCm — prefill/prefill-chunk paths would need Triton instead) | — | `bindings.cpp` ROCm branch omits them; decode unaffected |

**Total Python-fallback GPU work ≈ 4.5-5 ms/token on 9B (≈27%), ~5-6 ms on 27B** — second-largest optimization class after the GEMV rate itself. All of it is *inside* the captured graph (so no dispatch cost), but the kernels themselves are unfused torch ops.

### 6.4 Marginal value of porting BC_*/cubin-extraction to ROCm today

Honest answer: **little for steady-state decode; the whole-step graph already covers it.** Measured: graphs-off costs only +4 ms/token over graphs-on (9B and 27B alike) — and that is with per-linear BC graphs still active; the full pre-graph dispatch penalty was ~28 ms (AGENTS prior), i.e. ~24 ms is already recovered by `bc_rocm`+`block_graph_rocm` and the last ~4 ms is mostly per-module Python forward overhead that per-module C++ BC classes would also have to go through Python to be invoked... unless the *whole* block chain moves to C++ (BC_GatedDeltaNetSplit + BC_MLP + BC_Attention composited), which is what block_graph_rocm's single graph replay already achieves with one launch.

Where BC_*-style C++ *would* still buy something on ROCm:
1. **Kernel fusion inside blocks** (the real prize, not launch overhead): `BC_GatedRMSNorm` and `BC_MLP`'s fused silu·gate epilogue are *compute* fusions, not just capture helpers. The 4.5-5 ms/token of Python-fallback norm/activation kernels is exactly what those classes eliminate on CUDA. A Triton `rms_norm`/`gated_rms_norm`/`silu_mul_gate` set (or binding the existing C++ `norm.cu`/`activation.cu` after hipify — they're excluded, not unportable) attacks the same cost without any BC_* machinery.
2. **Uncaptured shapes**: bsz>1 (MAX_BSZ=8), q_len>1 ≤16 (spec-decode drafts), and any `EXL3_BLOCK_GRAPHS=0` fallback → these run full Python dispatch per module today.
3. **Prefill chunks** (PAGE_SIZE multiples): whole-step graphs never capture prefill; per-module BC graphs on CUDA cover the bsz==1 linear replays there. On ROCm prefill pays Python dispatch per module per chunk (out of decode scope, but it's where BC_* would still matter; AGENTS' fly-prefill report reached the same conclusion).
4. **Memory overhead of capture**: whole-step graph + statics ≈ small (one graph per key, MAX_GRAPHS=4); BC_* would not reduce this materially.
5. **First-token latency**: 8 uncaptured warmup steps + capture (~1-2 s) — BC per-module capture has the same order of warmup.

**Where the remaining 57 ms actually goes (27B, measured):** 29.8 ms MLP GEMV kernels, 12.5 ms GDN (GEMVs + recurrent/conv/norm kernels), 3.6 ms attention block, 2.3 ms lm_head, ~9 ms norms/glue/sampler/embed/gaps. It is *GPU-kernel-bound*, not dispatch-bound — the CUDA-only launch-side machinery is not the bottleneck on ROCm anymore.

### 6.5 Other CUDA-only techniques observed

- **C++-side fused epilogues**: `BC_MLP`'s silu·mul, `BC_GatedRMSNorm`, `gated_delta_net_fused_op` variants (these ARE compiled on ROCm — `gdn.cu` is in the ROCm source list — and used; only the *whole-block graph wrappers* are missing). `routing.cu`, `hc_mix.cu`, `histogram.cu`, `softcap.cu` excluded but unused in Qwen3.5 decode.
- **Kernel packing**: `exl3_mgemm` (multi-matrix single launch, `bindings.cpp` CUDA branch) — would pack q/k/v (and gate) GEMVs of one layer into a single launch on CUDA; on ROCm the fork implemented the same idea as a Triton fused-GEMV (`attn_rocm_kernels.fused_qkv_gemv`) — equivalent for decode, missing for prefill shapes.
- **Fused sampler** (`sampling_fused.cu`, `EXL3_FUSED_SAMPLER`): disabled on ROCm (`ext.py:223-227`); decode pays a GPU argmax + blocking `.cpu()` per token. Cost included in the ~9 ms residual (est. 0.1-0.3 ms/token + sync stalls; a pinned-memory/`to_non_blocking` + delayed-read pattern or a Triton argmax would recover most).
- **Pinned-memory staging**: `block_graph_rocm.refresh_inputs` already copies `block_table`/`cache_seqlens`/`positions` through persistent device buffers with `non_blocking=pinned`; the CPU embedding stages via `static_hidden.copy_` (H2D ~8 KB/token — negligible). Nothing left on the table here.

---

## 7. Suggestions (ranked, measurement-traceable; NO implementation done)

1. **Raise in-situ b4 GEMV bandwidth for MLP shapes** (29.8 ms → 17-19 ms possible). Evidence: §4.1 rates (292-335 GB/s vs lm_head's 411-551 in the same step); AGENTS per-K standalone rates (K8: 757 GB/s). Directions worth investigating *in ranked order*: per-K fast-path coverage for the exact MLP shapes (K=4096/12288), FlyDSL for the biggest streams (beats Triton 1.13× on the b6 stream; AGENTS), L2-friendly issue order inside the whole-step graph, and a persistent-style GEMV that keeps more reads in flight (the trellis-layout experiment already ruled out coalescing as the limit — AGENTS).
2. **Fuse the Python-fallback norms/activations** (~4.5 ms/token 9B, ~5-6 ms 27B → ~1 ms). Evidence: §4.2 stage table (block norms 3.40 ms), §6.3 call census. Cheapest big win on the list: a Triton `rms_norm`/`rms_norm_res_in`/`gated_rms_norm`/`silu_mul` quartet (or hipify+bind the existing `norm.cu`/`activation.cu`, which the ROCm build currently excludes).
3. **GDN z/o GEMV rate + glue** (block 12.5 ms → ~8 ms). Evidence: z/o at ~200-260 GB/s vs qkv 257-317 and lm_head 411+; conv/recurrent/norm ≈2.5 ms of separate small kernels. Directions: same GEMV work as (1) applied to [5120→3840]/[6144→5120]-class shapes; fold the gated-norm into the recurrent kernel epilogue (mirrors upstream `BC_GatedDeltaNetSplit`).
4. **Attention block polish** (3.6 → ~2 ms ceiling). Evidence: §4.1; AGENTS floor analysis (500 GB/s in-situ ⇒ ~2.2 ms). Diminishing returns — do last.
5. **Sampler/embed micro-path** (0.3-0.5 ms): non-blocking argmax readback or Triton argmax; keep the CPU embedding (H2D is negligible).
6. **Do NOT spend effort** on: BC_*/TritonKernel C++ porting for steady-state decode (§6.4 — graphs already cover it; whole-app is GPU-bound), `EXL3_PREFER_TRITON_LINEAR` toggling (parity at 4 bpw decode), further dispatch-side work (residual dispatch ≈ 0.2 ms/token with graphs on, AGENTS prior, consistent with our graphs-off delta).

**Combined ceiling check:** (1)+(2)+(3) at conservative targets (500 GB/s GEMV, 1 ms norms, halved glue) lands ≈ 29.5 tok/s; at stretch targets (550 GB/s, 3 ms residual) ≈ 34 tok/s. 33 tok/s is reachable only if the in-situ GEMV rate problem is solved for the MLP-class streams — that is the single make-or-break lever.

---

## Appendix: reproduction scripts (untracked)

| Script | Produces |
|---|---|
| `profiling/baseline_9b.py` | §3 baselines (per-env subprocess) |
| `profiling/driver_ablate.py` + `profiling/ablate_child.py` | §4.1 9B ablation table (subprocess-per-patch, interleaved bases) |
| `profiling/ablate_child2.py` | 27B ablations + fixed 9B conv/recurrent patches (Generator-based, patch-before-capture) |
| `profiling/causal_v2.py --mode stages` | §4.2 graphs-off stage table |
| `profiling/weight_bytes.py` | §5 per-class weight bytes |
| `profiling/probe_ext_surface.py` | §6.2/6.3 ext symbol census |
| `profiling/trace_probe2.py` | §6.1 BC-attn DECLINED verification |
| `profiling/inspect_modules.py` | module-tree ground truth |
| Raw artifacts | `/tmp/ablate_subprocs.json`, `/tmp/base_*.json`, `/tmp/stages_9b.json`, `/tmp/overnight*.log` |
