# MLP Deep Dive — exllamav3 ROCm decode (research only, no code changes)

**Parent report:** `bench/REPORT_decode_causal_profiling.md` (27B MLP = 29.81 ms/token, 52.1% causal share; 8.70 GB/token at 292 GB/s in-situ).
**This report:** where those 29.81 ms actually go, with verified mechanisms. All measurements fresh this session (worktree @ `89c8379`), ctx=512, bsz=1, whole-step graphs ON for in-situ numbers, hot clocks (4-5 s dummy-matmul ramp), best-of-N event timing. Scratch scripts under `profiling_mlp/` (untracked); raw JSON/logs in `/tmp/mlp_*.json`, `/tmp/mlp*.log`.

**Framing under test (user hypothesis):** decode is bandwidth-bound ⇒ MLP should stream at ~80% of peak DRAM (960 GB/s spec → ~768 GB/s). Measured: 292 GB/s in-situ aggregate. This report sorts the 2.6× gap into (a) physics, (b) kernel inefficiency, (c) interference — verifiably.

---

## 1. Executive summary — ranked targets WITHIN the MLP

| # | Finding (27B unless noted) | Causal number | Verified mechanism | What it implies |
|---|---|---|---|---|
| 1 | **GEMV kernels are latency/occupancy-limited, NOT DRAM-fed** — the master result | kernel-time rate 239-330 GB/s vs **899 GB/s** measured DRAM anchor on the same GPU (§A); L2-resident runs are only +1-4% faster than DRAM-cold (§A) | the fix class is **kernel-side memory-level parallelism** (configs today, kernel restructuring tomorrow), not access-pattern/traffic reduction | ~2.6-3.7× headroom exists in physics; kernel must be rebuilt to consume it |
| 2 | **down_proj (K=17408, N=5120) is the worst shape: 239-244 GB/s vs gate/up 329-333** — the intra-block 1.4× (27B) / 1.7× (9B: 242 vs 409 composite, 260 vs 484 pure-kernel) spread | 11.9 of 29.8 ms on 27B (9B: 3.35 of 7.22 ms) | starved-N parallelism deficit: default BN128 → 40 CTAs (<1 wave on 48 CUs); every alternative tile trades CTA count against staged-window size (§B rocprof table) | per-shape configs recover part; full recovery needs split-K-class kernel work |
| 3 | **Autotune config-set deficiency, not mis-tuning**: the default M=1 config list lacks the winning down-shape tiles entirely | in-situ forced-config A/B: 9B **17.40 → 16.63 ms/token (+4.5%, 57.5→60.1 tok/s)**; 27B **56.43 → 55.21 (+2.2%, 17.7→18.1 tok/s)**, zero code changes (§B) | forced BN32/BK256 + BN32-64/BK128 picks beat the default BN128/BK128/nw8 for down; g/u prefer BN128/BK128 or BN64/BK256 | cheapest verified win in the whole fork right now; extend `_exl3_gemm_configs()` |
| 4 | **In-situ interference ≈ 0**: full-MLP standalone sweep = in-situ block time (9B: 7.22 vs 7.22 ms; 27B: 29.27 vs 29.81, +1.8%) (§C) | whole-model L2/scheduling penalty on the MLP block: ≤2% | the in-model environment does not degrade the GEMVs; standalone rates are the truth | microbench-driven kernel work WILL transfer to the model |
| 5 | **silu_mul glue is small**: 0.23 ms/token (9B, 32 layers) by ablation (§F) | fusing into GEMV epilogue (upstream BC_GatedMLP behavior, missing on ROCm) recovers ≤0.23 ms (9B) / ≤0.4 ms (27B est.) | python-fallback `F.silu(x)*y; z.copy_(r)` = 2 kernels/layer | do after GEMV work; worth ~1-1.5% |
| 6 | **27B gate/up penalty vs 9B unexplained by CTA count**: 136 CTAs (2.8 waves) yet 330 vs 9B's 406-409 GB/s at 96 CTAs (§B) | 8.66 vs (scaled) ~6.9 ms/pass | correlates with K=5120 vs 4096 (25% longer K-loop per CTA) and trellis size 44.6 vs 25.2 MB; not CTA count (enough in both); open mechanism — suspect DRAM page/bank behavior of 128 B chunks strided by tiles_n, or issue-rate limit at higher per-CTA work | candidate for split-K or N-major trellis re-layout research |

**Honest 80%-peak ceiling (§E):** if every MLP GEMV ran at the best in-situ rate already demonstrated for its shape class (550 GB/s, the 9B lm_head b6 stream), the 27B MLP floor = 8.70 GB / 0.55 TB/s ≈ 15.8 ms → whole model ≈ 43 ms ≈ **23.2 tok/s**. The full 80% framing (768 GB/s) gives MLP = 11.3 ms → 36 ms ≈ **27.6 tok/s** (other blocks at baseline). Neither reaches 33 tok/s alone — consistent with the parent report's conclusion that MLP-rate + overhead-reduction must combine.

---

## 2. Methodology and validity

Same discipline as the parent report (patch-before-capture subprocess protocol, clock ramps, best-of-N, order-permutation). New pitfalls documented:
- **Single-pass standalone timings are 5-15% low** (clock ramp): first cfg_sweep pass on g was 2.24 ms vs 2.00 steady. All standalone numbers here are best-of-4.
- **Counter tooling**: `/opt/rocm/bin/rocprofv3` (system, v7.14 ROCm tools) crashes with `Option 'spirv-expand-step' registered more than once` (double-LLVM load with the venv python) or hangs outright. **The venv-installed `rocprofv3` (`$MAIN/.venv/bin/rocprofv3`, rocm-profiler 7.14.0 wrapper → ROCProfilerV3 SDK 1.3.2) works** — it sets its own `LD_LIBRARY_PATH` (rocm_sysdeps) before exec. All counter data below uses the venv profiler. (See §A for invocation.)
- PMC multiplexing: counters collected in one pass with `-o <prefix>`; per-dispatch rows keyed by `Grid_Size` disambiguate shapes (gate/up N=12288→grid 96@BN128; down N=4096→grid 32; 27B: 136/40).

Reproduction: `profiling_mlp/*.py` + `/tmp/mlp_pipeline*.sh` (driver logs preserved).

---

## 3. Section A — actual vs theoretical DRAM traffic

**Tool (the working invocation):** `$MAIN/.venv/bin/rocprofv3 --pmc <ONE_COUNTER> -o /tmp/sc9b_<NAME> -f csv -- python profiling_mlp/counter_target2.py` — the **venv** profiler (system `/opt/rocm/bin/rocprofv3` aborts on double-LLVM in this container); **one counter per invocation** (multi-counter PMC aborts at exit with a correlation-id check failure); per-dispatch rows carry Kernel_Id/Grid_Size/Workgroup_Size/timestamps.

**Anchor (physics ceiling):** single 2.4 GB buffer, `buf.sum()`: **899 GB/s** (2.67 ms) — the DRAM subsystem demonstrably feeds ~900 GB/s to a trivially parallel reader; the 80%-of-peak figure (768) is comfortably achievable by simple kernels.

**Counter status (rocprofv3, venv profiler — invocation below):** per-dispatch **durations, grids, workgroups and Kernel_Ids are collected correctly**; raw SQ-block counters work (verified `SQ_WAVES_sum` on a matmul: 205 501 waves); **all GL2C_* (L2) counters and derived FETCH_SIZE/WRITE_SIZE return 0.000 on this gfx1100 stack** (tested single- and multi-counter, on torch kernels and Triton kernels alike; `pmc-check` claims support but values stay zero — GL2C block counters are effectively dead here). Actual-DRAM-bytes therefore cannot be read directly; the traffic question is settled by the L2-resident differential below plus kernel-source analysis, and the kernel-time rates come from rocprof hardware timestamps (better than event timing: pure kernel time, no gaps).

**Pure GEMV kernel rates from rocprofv3 durations (9B; group = (Kernel_Id, grid, workgroup) = compiled config; 7 654 GEMV dispatches timed):**

| config (decoded) | shape | CTAs | n | med µs | **GB/s (pure kernel)** | % of 899 anchor |
|---|---|---|---|---|---|---|
| BN128/BK128/w8 (default pick) | gate/up (N=12288) | 96 | 1499 | 52.0 | **484** | 54% |
| BN128/BK128/w8 (default pick) | down (N=4096) | 32 | 1074 | 96.8 | **260** | 29% |
| BN64/BK128/w4 | gate/up | 192 | 1028 | 65.7 | 383 | 43% |
| BN64/BK128/w4 | down | 64 | 1252 | 100.2 | 251 | 28% |
| BN128/BK64/w8 | gate/up | 96 | 733 | 67.8 | 371 | 41% |
| BN128/BK64/w8 | down | 32 | 657 | 129.5 | 194 | 22% |
| BN16/generic (autotune trial) | gate/up | 768 | 725 | 74.7 | 337 | 37% |
| BN16/generic (autotune trial) | down | 256 | 686 | 106.7 | 236 | 26% |

(The composite event-timed sweep rates quoted elsewhere — g/u 405-409, down 242 — include the two hadamard kernels (1.8 µs each, measured from the same CSVs) plus ~4 µs inter-kernel gap per linear; pure-kernel GEMV rates are ~15-19% higher. Both views tell the same story.)

**L2-resident vs cold (the traffic-independence proof):**

| Model | shape | isolated (one trellis, ≤45 MB ≈ L2) | sweep cold | Δ |
|---|---|---|---|---|
| 9B | gate/up 4096→12288 | 423/420 GB/s | 405/409 | +4% |
| 9B | down 12288→4096 | 250 | 242 | +3% |
| 27B | gate/up 5120→17408 | 333/332 | 330/330 | +1% |
| 27B | down 17408→5120 | 249 | 239 | +4% |

If DRAM underfeed or traffic amplification (re-reads, codebook re-fetch, hadamard temps) were the limiter, serving the entire working set from L2 would be several× faster. It is 1-4% faster. Kernel-source check agrees: the b4 fast path stages each trellis k-tile with two linear u32 row loads (row + row shifted one word; the m1 wrap is in-register — `exl3_triton.py:526-545`), so extra global traffic per weight is ~0 by construction; codebook/LUT and x/y are KB-scale.

**Verdict (A):** *actual ≈ theoretical traffic, and it does not matter* — the kernels leave DRAM idle (29-54% of the demonstrated 899 GB/s anchor at pure-kernel level). The gap is **kernel inefficiency (memory-level parallelism), not physics and not traffic**. The fix class is occupancy/latency/config + kernel restructuring.

## 4. Section B — why down_proj is 1.4-1.7× slower per byte

**Shapes:** 9B down K=12288→N=4096 (25.2 MB); 27B down K=17408→N=5120 (44.6 MB). Grid = N/BLOCK_N.

**Hypotheses tested (forced single config per process via `EXL3_GEMM_CONFIGS`, cold sweeps, best-of-4):**

| Config (BN,BK,warps) | 9B down CTAs | 9B down GB/s | 27B down CTAs | 27B down GB/s |
|---|---|---|---|---|
| 128,128,w8 (default pick) | 32 | 242 | 40 | 239-244 |
| 64,128,w4 | 64 | 235 | 80 | 239 |
| 64,64,w4 | 64 | 190 | 80 | 237 |
| 64,64,w8 | 64 | 209 | 80 | 216 |
| 64,128,w8 | 64 | 220 | 80 | 259 |
| 32,64,w4 | 128 | 196 | 160 | — |
| 32,128,w4 | 128 | 265 | 160 | 252 |
| 32,256,w4 | 128 | **280** | 160 | — |
| 64,256,w4 | 64 | 260 | 80 | — |
| (gate/up best: 128,128,w8 / 64,256,w4) | 96 | 406-425 | 136 | 329-330 |

**Mechanism verdict:**
- (1) **Memory-level parallelism (CTAs × staged-bytes-per-iteration) is the operative variable — "CTA count" alone is a partial story.** rocprof per-config durations show: gate/up at BN128/BK128 (96 CTAs, 484 GB/s) BEAT BN64/BK128 (192 CTAs, 383 GB/s) — more CTAs but half the staged window per CTA is slower. For down (N=4096-5120), BN128 allows only 32-40 CTAs (<1 wave); shrinking BN restores CTA count and wins (242→265-280 forced sweeps) *provided BK stays ≥128* — the winning down configs are BN32-64 × BK128-256. The starved-N shapes force a BN-vs-parallelism tradeoff no single tile can win.
- (2) **K-loop depth matters only via bytes-in-flight per iteration**: BK64 is uniformly worse than BK128/256 at equal BN (190-209 vs 235-280 GB/s; rocprof: BN128/BK64 down 194 vs BN128/BK128 260).
- (3) **Autotune-at-ramping-clocks is NOT the cause** — but **config-set deficiency IS**: all forced-config runs above were at hot clocks and reproduce the same ordering; the default list (`exl3_triton.py:390-404`) contains no BN<64 tile for the M=1 b4 path (only the bits=3 pair), so the tuner cannot select what wins on starved-N shapes. **In-situ A/B with a tuned list (down: BN32/BK256, BN32/BK128; g/u: keep BN128/BK128 + add BN64/BK256): 9B 17.40 → 16.63 ms/token (57.5 → 60.1 tok/s, +4.5% end-to-end); 27B 56.43 → 55.21 ms/token (17.7 → 18.1 tok/s, +2.2%).**
- Residual: even the best config leaves 9B down at 280 vs 406-425 GB/s for gate/up on the SAME model — small-N shapes structurally lack parallelism at any single-config tiling; candidates: split-K across CTAs (partial sums + tiny reduce), multi-CTA per N-tile, or a persistent-kernel design that pipelines across the three dependent GEMVs of a layer.

## 5. Section C — in-situ vs standalone decomposition

| Measurement | 9B | 27B |
|---|---|---|
| standalone full-MLP sweep (all layers, decode order, cold) | 7.22 ms/pass (335 GB/s) | 29.27 ms/pass (293 GB/s) |
| in-situ whole-MLP ablation delta (parent report) | 7.22 ms | 29.81 ms |
| **inference: interference (L2 pollution from other streams, scheduling, gaps)** | **≈0%** | **+1.8%** |

Per-shape standalone-cold vs in-situ-implied rates agree within ±10% (9B: g 405→438 in-situ-ish, u 409→373, d 242→234). The isolated-single-layer probe (`mlp_keep_L0` − `mlp_no_all3` = 0.60 ms for one layer vs 0.226 ms/layer average) was **rejected as a measurement** — the two patches remove different residual glue (zeros-kernels/silu vs module identity), confounding the delta; reported here as a falsification (§9).

## 6. Section D — gap accounting inside the MLP block

9B ablations (same-session where paired; base 17.36-17.40):

| ablation | Δms | share of MLP 7.22 |
|---|---|---|
| mlp_no_down | 3.45 | 47.8% |
| mlp_no_up | 2.17 | 30.1% |
| mlp_no_gate | 1.85 | 25.6% |
| mlp_no_silu | 0.23 | 3.2% |
| pair g+u | 4.00 (vs 1.85+2.17=4.02) | additive ✓ |
| pair u+d | 4.97 (vs 5.62) | **−12% sub-additive** |
| all (no_MLP) | 7.22 | — |

The u+d non-additivity (−0.65 ms) is the measured inter-kernel overlap/gap correction: removing up also removes the dependency stall its consumer chain imposes on down, or vice versa. Everything else additive to ≤1%. Net: **no hidden gap mass** — the MLP block time is its three GEMVs plus ~3% glue.

## 7. Section E — the 80%-peak extrapolation (bridge table)

27B MLP bytes/token 8.70 GB; other blocks at baseline (57.19 − 29.81 = 27.38 ms):

| MLP aggregate rate | MLP ms | whole ms/token | tok/s | status |
|---|---|---|---|---|
| 292 (measured today) | 29.81 | 57.19 | 17.5 | baseline |
| 335 (9B-MLP rate, bigger-N shapes) | 25.97 | 53.4 | 18.7 | config fix only (9B demonstrated end-to-end +4.5%, 27B +2.2%) |
| 500 | 17.40 | 44.8 | 22.3 | needs down 240→~450 (kernel work) + g/u 330→~500 |
| 550 (best in-situ rate ever measured here: 9B lm_head b6) | 15.82 | 43.2 | 23.2 | honest demonstrated-rate ceiling for this kernel class |
| 768 (80% of peak) | 11.33 | 38.7 | 25.8 | needs ≥2.6× current aggregate; no kernel in this family has shown it on these shapes |
| 899 (bigbuf anchor) | 9.68 | 37.1 | 27.0 | physics bound for ANY reader |

**Honest ceiling under the 80% framing: ~25.8 tok/s from MLP-rate fixes alone** (other blocks baseline) — and that requires rates nothing in the current kernel family has demonstrated on these shapes. Combined with the parent report's overhead cuts (norms/glue ~6-9 ms), 33 tok/s remains reachable only if BOTH the GEMV family is rebuilt to sustain ≥500-550 GB/s on all shapes AND the non-GEMV residual drops to ~3 ms.

**Gap accounting (27B MLP, 29.81 ms):**
- physics floor at 899 GB/s anchor: 9.7 ms (32%)
- gate/up at 330 vs 550 demonstrated-class: 8.66 → 5.19, gap 3.5 ms (12%)
- down at 239 vs 550: 11.93 → 5.19, gap 6.7 ms (23%)
- residual above 550-class rate: 33% (needs kernel-family redesign; the 899 anchor says it is not physics)

## 8. Section F — silu_mul + glue

`mlp_no_silu` (activation → plain copy, GEMVs untouched): 9B −0.23 ms/token = 3.2% of MLP block. Upstream `BC_GatedMLP` fuses silu·gate into the GEMV epilogue (C++, excluded on ROCm); the recoverable upper bound is this 0.23 ms (9B) / ~0.4 ms (27B, 2× layers). Also removed by fusion: one intermediate tensor round-trip (`a` 24 KB/layer — negligible bytes, real latency). Priority: below GEMV work.

## 9. Falsifications (hypotheses tested and killed)

1. **"In-situ L2 interference explains the 292 GB/s"** — killed: standalone cold sweeps reproduce in-situ rates exactly (9B 7.22=7.22 ms; 27B 29.27 vs 29.81).
2. **"DRAM under-feeds the kernels (traffic problem)"** — killed: L2-resident runs only +1-4% faster; bigbuf anchor 899 GB/s; kernel source stages each trellis row once (no rotated re-reads).
3. **"Cold-clock autotune mis-selection locks a bad config"** — killed as primary cause: forced-config hot-clock sweeps reproduce the same per-config ordering; the real defect is the config SET (no small-BN M=1 tiles for b4).
4. **"K-loop length (down's 4.2× serial work per CTA) is the mechanism"** — partially killed: at fixed CTA count, BK does matter (bytes in flight), but the BN×parallelism tradeoff dominates; down@BN32 (128 CTAs, same long K) reaches 280.
5. **"More CTAs is always better"** — killed by rocprof durations: g/u BN64 (192 CTAs, 383 GB/s) loses to BN128 (96 CTAs, 484 GB/s); the staged-window size per CTA trades against CTA count.
6. **"Ordering/warmup artifacts inflate the g-vs-u asymmetry"** — killed: order-permuted sweeps (g,u,d vs d,u,g) agree within 0.5%.
7. **"Single-layer isolation measures per-layer in-situ cost"** — killed (methodologically): keep_L0 probe confounded by asymmetric patch residue; not a valid probe.
8. **"System rocprofv3 is usable"** — killed: LLVM double-registration crash / hang in this container; the **venv** profiler (`$MAIN/.venv/bin/rocprofv3`) works and is what produced §A.
9. **"GL2C byte counters will quantify re-reads directly"** — killed empirically: all GL2C_*/FETCH_SIZE values = 0.000 on this gfx1100 stack (SQ-block counters work; L2 block does not). Documented with reproduction commands; revisit when the stack fixes L2 PMC.

## 10. What to build (suggestions ONLY — no implementation)

1. **Extend the M=1 b4 config set** (`exl3_triton.py:_exl3_gemm_configs()`) with BN32/BK128-256 and BN64/BK256 tiles (guarded by the existing `fast_ok` prune). Measured A/B: **9B +4.5% end-to-end (17.40→16.63 ms/token), 27B +2.2% (56.43→55.21)** — pure config selection, zero kernel changes. Confidence: high (direct in-situ A/B on both models).
2. **Per-shape config pinning at load time** (select by N bucket, skip autotune on starved-N shapes). Confidence: high; removes the cold-clock risk AGENTS warns about.
3. **Kernel: raise memory-level parallelism on starved-N shapes** — split-K (CTA-partial sums + small reduce) or multi-CTA-per-N-tile for down_proj-class shapes; the rocprof table pins the tradeoff no single tile can win. Target: 240→450+ GB/s; would cut 27B MLP by ~5-6 ms. Confidence: medium-high (mechanism verified: CTAs × staged-bytes is the operative variable; not prototyped — deliberately, research-only).
4. **Investigate the 27B gate/up 330-vs-406 GB/s shape penalty** (136 CTAs, K=5120): candidates are DRAM page-locality of the tiles_n-strided 128 B chunks and per-CTA issue-rate limits at longer K. A split-K variant or trellis N-major re-layout experiment (prior negative result on a different axis — AGENTS) would discriminate. Confidence: low-medium (mechanism open).
5. **Fuse silu·gate into the GEMV epilogue** (Triton, mirroring upstream BC_GatedMLP): ≤0.23 ms/token (9B) / ~0.4 (27B). Confidence: high, low priority.
6. **Hadamard micro-fusion** (optional): each linear pays 2×1.8 µs hadamard + ~4 µs gap ≈ 7.6 µs of its ~60 µs composite (13%); folding the input hadamard into the GEMV's x-read or batching the two output hadamards would recover a fraction of 9B ~0.7 ms/token across all linears. Confidence: medium.

---
*Counter tooling note: all rocprofv3 data in §A/B comes from the venv profiler (`$MAIN/.venv/bin/rocprofv3`), single counter per invocation (`--pmc SQ_WAVES_sum` verified non-zero; GL2C block zero-valued on this stack). Raw CSVs: `/tmp/sc9b_*_counter_collection.csv` (7654 GEMV dispatches timed), parsed with `profiling_mlp/parse_counters.py`-style grouping by (Kernel_Id, Grid_Size, Workgroup_Size).*
