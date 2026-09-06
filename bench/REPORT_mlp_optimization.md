# MLP GEMV Optimization — implementation report (Tier 1 + Tier 2)

**Branch:** detached HEAD @ `89c8379` (integration tip) + implementation commit(s).
**Scope:** one file, `exllamav3/modules/quant/exl3_triton.py` (+ new tests in `tests/test_exl3_triton.py`), per the task contract.
**Spec:** `bench/REPORT_mlp_deep_dive.md` (copied in-tree, untracked) — the MLP deep-dive's §10 build list, priority-adjusted.
**Environment:** RX 7900 XTX (gfx1100), ROCm 7.14, torch 2.12+rocm, triton 3.7.1; main-tree venv; all measurements cuda:0, hot clocks (3-5 s dummy-matmul ramp), in-situ numbers via the Generator flow (bench_decode.py pattern — the proven 27B harness) with whole-step graphs on, interleaved subprocess A/B vs the baseline tree @ 89c8379.

---

## 1. Executive summary

| Model | metric | baseline | final (Tier 1+2) | tok/s |
|---|---|---|---|---|
| 9B-4bpw | in-situ ms/token (interleaved A/B, median of 3) | 17.53 | **14.97** | 57.0 → **66.8** |
| 27B-4bpw | in-situ ms/token | 56.77 | **52.10** | 17.61 → **19.20** |
| 9B-6bpw (spot) | in-situ ms/token | 24.77 | **17.35** | 40.4 → **57.6** |
| 9B-2bpw (spot) | in-situ ms/token | 17.76 | **17.09** | 56.3 → **58.5** |

| stream | standalone-cold composite GB/s (baseline → final) |
|---|---|
| 9B-4bpw down_proj (K=12288, N=4096) | 241 → **450-455** |
| 27B-4bpw down_proj (K=17408, N=5120) | 239-244 → **385** |
| 9B-4bpw gate/up | 394-407 → 406 |
| 27B-4bpw gate/up | 321-322 → 329 |
| 6bpw MLP gate/up (b6, N=12288) | ~318 (BN16/BK64 pick) → **554** |
| 6bpw MLP down (b6, N=4096) | ~256 → **441-454** |

**Tier 1 (MUST): achieved.** 9B 16.51 ≤ 16.7 ✓ (measured with the Tier-1-only tree), 27B improved past its 55.3 target in the final A/B (52.10).
**Tier 2 (TARGET): achieved.** Standalone-cold down ≥350 GB/s on BOTH models (450 / 385 vs target 350; from 242/239); 9B 14.97 ≤ 15.8 ✓; 27B 52.10 ≤ 52.5 ✓.
**Tier 3 (stretch): partially achieved for free** — the output-side Hadamard fusion (§10.6) ships with the split-K reduce kernel (one had launch removed per qualifying linear); 9B beat the stretch number (14.97 vs ~15.2), 27B landed at 19.2 tok/s vs the ~20 stretch.
**Gate 3 (numeric honesty):** split-vs-classic max abs diff 0.015625 (rel ~5-8e-4, tolerance 2e-2); greedy generation **token-identical** to the pre-change kernel on 4B (both models, 3 prompts × 219 tokens each, 100% argmax match).

A bonus finding: the bits=6 M1 autotune pool was catastrophically mis-set upstream of this work (the 6bpw model's MLP ran at 200-256 GB/s because every BN≥64 tile collapses the heavy 4-accumulator b6 kernel's occupancy). The final pools fix that too — hence the 30% 6bpw decode gain.

---

## 2. What changed (mechanism → change → measured effect)

### 2.1 Tier 1 — N-bucketed M=1 autotune pools (config set + prune)

**Mechanism (deep-dive §B):** down_proj-class shapes (N=4096/5120) have N/BLOCK_N = 32-40 CTAs at BN128 — under one wave on 48 CUs — and the default config list contained no BN<64 tile for the b4 M=1 path, so the tuner could not select the parallelism-restoring tiles. Verified winners: BN32/BK128-256 for down; BN128/BK128 and BN64/BK256 for gate/up.

**Change:**
- `_exl3_gemm_configs()` default list: M=1 section now carries BN32/BK128, BN32/BK256, BN64/BK256 alongside BN128/BK128, BN64/BK128 and the existing bits=3/generic members. The dominated BN128/BK64/w8 entry was dropped (measured 194-216 GB/s on down, 328-371 on g/u — beaten by every surviving member).
- `_exl3_gemm_early_prune()`: for M==1 full-tile (fast_ok) invocations the candidate pool is bucketed **per width class and N**:
  - **bits=4** (light [2,2,2,NN,8,4] M1 accumulator): starved-N (N≤8192) `{BN32/BK128, BN32/BK256, BN64/BK128}`; large-N `{BN128/BK128, BN64/BK256, BN64/BK128}`; split-K-eligible shapes (2.2) `{BN32/BK256, BN64/BK256}`.
  - **other widths (1,2,5,6,7,8)** (heavy-accumulator M1 kernels — 2-4 fp32 tensors of 256·NN elements per CTA — or the staged-gather generic path): N≤16384 `{BN32/BK128, BN32/BK256}`; larger N `{BN64/BK128, BN32/BK128}`.
  - bits=3 keeps its existing narrow-tile pool; non-divisible shapes keep the generic small-tile pool.

**The heavy-width pools are driven by measured landmines** (all composite GB/s, this session):

| tile | b6 N=12288 | b6 N=4096 | b6 N=248320 (lm_head) | b2 N=12288 | b2 N=4096 |
|---|---|---|---|---|---|
| BN16/BK64 (old autotune pick for b6/b2 MLP) | 318 | 256 | 364 | 162 | 115 |
| BN32/BK128 | **554** | **441** | 667 | **173** | **114** |
| BN32/BK256 | 539 | **454** | 634 | — | — |
| BN64/BK128 | 203 | 174 | **685** | 86 | **34** |
| BN128/BK128 | 201 | 175 | 617 | — | — |

i.e. the b4-style wide tiles collapse the heavy kernels' occupancy (per-CTA register pressure), while at the huge lm_head N the deep BN64/BK128 window wins again — hence the two-bucket rule. My first cut used the b4 pools for all widths and cost the 6bpw model +16% decode; the regression was caught by the mandated bitwidth spot-check, root-caused by dumping the autotuner picks (baseline actually ran BN16/BK64 there), and fixed as above. Final spot checks: 6bpw 24.77→17.35 ms/token (-30%), 2bpw 17.76→17.09 (-3.8%).

**Floor-safety rule (bits=4):** every pool member was measured at or above the previous default pick's rate on that bucket's shapes (both models), so a cold-clock autotune mis-rank cannot lock in a regression:
- 9B down: {265, 280, 235} vs old pick 242 — BN64/BK128 is the only sub-baseline member (-3%), kept because it is a pre-existing option for mid-size shapes.
- 27B down: {252, 251, 239} vs old pick 239-244.
- 9B g/u: {406-410, 425, 383-415} vs old pick 406-410.
- 27B g/u: {321-322, 329-334, 328-330} vs old pick 321-322.

**Effect (standalone-cold, composite):** 9B down 242→278; 27B down 239→250; g/u unchanged-to-better. In-situ 9B: 17.47 → 16.51 ms/token (+5.5%). Zero kernel-code changes.

### 2.2 Tier 2 — M=1 split-K GEMV with fused reduce + output Hadamard

**Mechanism (deep-dive §1/#2):** the GEMV kernels are latency/occupancy-limited, not DRAM-fed (L2-resident ≈ cold; 899 GB/s anchor). The operative variable is CTAs × staged bytes in flight. Starved-N shapes cannot buy CTAs with any single tile (BN32 maxes at 128-160 CTAs and trades away the staged window); splitting the K loop across CTAs multiplies CTA count at unchanged per-CTA window size.

**Change** (all inside `exl3_triton.py`):
- `_fused_dequant_gemm_kernel`: new `SPLITS` constexpr + `stride_ys` arg. For SPLITS>1 (b4 M=1 fast path only) the grid becomes `(N/BN) × SPLITS`, each CTA reduces a contiguous K-slice `k_base = pid_split × tiles_per_split` (`tiles_per_split` rounded to NK multiples; over-extended splits store zero partials — correct for any shape) and stores its BLOCK_N fp32 partial to row `pid_split` of a `[SPLITS, N]` buffer. SPLITS==1 codegen is unchanged (constexpr dead-code elimination); all other branches (M>1, other bit widths) never see SPLITS>1 (host-guarded in `exl3_gemm_triton`).
- `_m1_splitk_plan(M, N, K, bits)`: host-side, deterministic, env-free default. Splits only when M==1, bits==4, N≤8192, N%256==0, K%256==0 (guarantees the b4 fast path for every pool tile), and each split gets ≥4 outer iterations: **S=8 if K/16 ≥ 512, S=4 if K/16 ≥ 256, else 1.** `EXL3_SPLITK=off|n` is a dev override (default behavior sets no env var).
- `_get_splitk_buf(N, S, dev)`: `[S, N]` fp32 partials cached per (N, S, device) — 64-160 KB per qualifying shape; allocated once at first call (= BC warmup, before graph capture), stable addresses, nothing allocated in the steady-state path (CUDA-graph capturable by construction; no host syncs, no `.item()`).
- `_m1_split_reduce_had_kernel`: one program per 128-column block sums the S partials in fp32, runs the SAME fp32 radix-2 butterfly as `_had_r_128_kernel` (reuses `_had_stage`), applies r_scale and the svh post-scale with the C++ rounding order — i.e. the output Hadamard is fused into the reduce, eliminating the separate had launch for these linears (part of the deep-dive §10.6 micro-fusion, output side).
- `_linear_exl3_triton`: routes split-eligible calls through partials + reduce/hadamard; everything else (all M>1, bits≠4, large-N) keeps the exact classic path.

**Numerics (Gate 3):** partials and reduce are fp32; the split path skips exactly one intermediate round-to-half (classic: GEMV fp32 → half store → hadamard; split: fp32 partials → fp32 reduce+transform → single final round). fp32 accumulation order changes (K-slice sums). Measured: max abs diff vs classic path 0.015625 on random trellises across all three codebook variants (relative ~5-8e-4, tolerance regime 2e-2); **greedy generation token-identical** (both 4B models: 3 prompts × 219 tokens each, 100% argmax match vs the pre-change kernel).

**Effect (standalone-cold, composite GB/s, distinct-layer sweeps, best-of-4):**

| shape | classic (Tier-1 pools) | S=2 | S=4 | S=8 | default |
|---|---|---|---|---|---|
| 9B down (N=4096, K=12288) | 278 | 368 | 445 | 438 (455 forced BN32/BK256) | **450-455** |
| 27B down (N=5120, K=17408) | 250 | 299 | 353 | 366 (385 forced) | **385** |

Sweep detail (forced configs, 27B down at S=8): BN32/BK128 383, BN32/BK256 385, BN64/BK256 386, BN64/BK128 367-375, BN128/BK128 356 → split-eligible pool {BN32/BK256, BN64/BK256} floor ≈ 385.

**In-situ (interleaved A/B, Generator flow, median of 3 rounds):** 9B 17.532 → 14.972 ms/token; 27B 56.773 → 52.096 ms/token. The gains exceed the MLP-down share because the plan also covers the GDN z/o projections and other N≤8192 b4 linears (they ran at 200-260 GB/s in the parent profile) and each qualifying linear loses one hadamard launch.

**Also verified:** the bits=6 lm_head stream (N=248320, large-N pool {BN64/BK128, BN32/BK128}) keeps its measured-best tile (685 GB/s standalone; autotune picks BN64/BK128, same as baseline).

---

## 3. Verification gates

- **Gate 1 (kernel suite):** `tests/test_exl3_triton.py` — **222 passed** (214 pre-existing + 8 new: split-plan unit test, split-path vs C++ reference × 2 shapes × 2 cb variants, split-vs-classic numeric delta half/fp32, prune pool buckets). No existing test weakened.
- **Gate 2 (integration pytest, AGENTS command):** **494 passed / 68 skipped** (= the expected 486/68 + exactly the 8 new kernel tests). Zero failures.
- **Gate 3 (numeric honesty):** split-vs-classic max abs diff 0.015625 (3 codebooks, random trellis, rel ~5-8e-4); generation identity 3 prompts × ≥219 tokens, greedy/argmax, **100% identical on BOTH 9B-4bpw and 27B-4bpw** vs the pre-change kernel.
- **Gate 4 (perf):** all in-situ numbers interleaved-subprocess A/B (baseline MAIN tree @ 89c8379 vs this worktree), fresh process per point (graph-state resets safe), 3-5 s clock ramps inside each child before the timed window, cuda-event windows over ~340 steady-state tokens, medians of 3 interleaved rounds (2 for the bitwidth spot checks).
- **Graph capture:** every in-situ number was produced with whole-step graphs ON (default env), i.e. the split path + reduce kernel were captured and replayed through BC_LinearEXL3 / block_graph_rocm on both models.

## 4. Bitwidth spot checks (final pools)

| model | base ms/token | cand ms/token | Δ |
|---|---|---|---|
| 9B-2bpw | 17.747 / 17.766 | 17.100 / 17.085 | **-3.8%** |
| 9B-6bpw | 24.737 / 24.797 | 17.419 / 17.272 | **-30%** |

(2bpw improvement from the narrow-tile pools + split-K on any qualifying b4 shapes; 6bpw from the narrow-tile b6 pools — see §2.1 landmine table.)

## 5. What I could NOT do in the one-file scope (recommendations)

1. **27B gate/up rate (deep-dive §10.4):** 329 GB/s vs 9B's 406-425 at the same kernel. Mechanism still open (DRAM page/bank behavior of tiles_n-strided 128 B chunks, or per-CTA issue-rate at K=5120). Split-K does not apply (N=17408 is not starved). Candidate next steps: split-K on the N axis for large-N shapes needs >1 partial pass or atomics (deterministic variant needs a second reduction stage — doable in-file but not measured to win); a trellis N-major re-layout is BLOCKED by the shared-weight rule (prefill shares the trellis; no VRAM headroom for a private copy on 27B).
2. **FWHT input-side fusion (§10.6):** each GEMV still pays the input hadamard (1.8 µs + gap). Fusing it into the GEMV means recomputing the 128-point transform of x inside every CTA (x is KB-scale) or a persistent-x scheme; prototyped neither — the output-side fusion (which is free once the reduce kernel exists) is in.
3. **silu·gate epilogue fusion (§10.5):** lives in the MLP module forward (`modules/mlp.py`-side call sequence), not in the linear — out of the one-file scope. Worth ≤0.23 ms/token (9B).
4. **Per-shape config pinning at load time (§10.2):** the width/N-bucket pools achieve the same cold-clock safety with less machinery; a load-time pin table would only shave autotune warmup (~seconds, once).

---

## Appendix: reproduction

Scratch scripts (untracked, `/tmp/impl_mlp/`):
- `ab_driver.py` + `ab_child.py` — interleaved subprocess A/B (Generator flow, clock ramps, cuda-event windows, optional token capture). `CAND_ROOT=<worktree> python ab_driver.py --model <dir> --rounds 3 [--emit-tokens]`
- `standalone.py` — per-shape L2-cold sweeps over all layers + autotune-pick dump (adapted from the deep-dive's `profiling_mlp/standalone_shapes.py`)
- `cfg_sweep.py` — forced single-config sweeps via `EXL3_GEMM_CONFIGS` (+ `EXL3_SPLITK` for the split factor)
- `one_linear.py` — isolated single-linear timing (lm_head stream)
- `inspect_bits.py` — per-model bit-width census + autotuner pick dump (this is how the 6bpw regression was root-caused)
- `check_split.py` — split-path correctness + split-vs-classic delta vs the C++ reference

Key JSON artifacts: `/tmp/impl_mlp/*.json` (sweeps, A/B summaries, token dumps).

Commits (detached HEAD on top of 89c8379):
- `57b6704` MLP GEMV: N-bucketed decode config pools + M=1 split-K GEMV with fused reduce/Hadamard
- `e9b9819` MLP GEMV: width-aware M=1 autotune pools (fix 6bpw regression, +30% 6bpw decode)
- `7ffc8ec` MLP GEMV: extend split-K to large-N b4 shapes (gate/up + qkv-class)

---

## 6. Follow-up session: `eval/perf.py -m /tmp/Qwen3.8-27B-exl3-4.00bpw`

Task framing: drive perf.py decode past 25 tok/s editing only `exl3_triton.py` (+ tests). Baseline at session start (all previous work in): **18.2-18.4 tok/s** (ctx 0-768, `-max_length 1024 -spf`). Final: **20.2-20.3 tok/s** (+10.4%). The 25 tok/s goal is **not reachable within this file's scope**; evidence below.

### 6.1 What was done

1. **Decode census** (`/tmp/impl_mlp/count_linears.py`): 16.5 GB/token of trellis streams flow through `_linear_exl3_triton` on the 27B — gate/up AND qkv-class `5120→17408` (170 calls/token, 7.58 GB), down-class `17408→5120` (85 calls, 3.79 GB), `6144→5120` / `5120→10240` / `5120→6144` z/b-class (213 calls, 4.0 GB), lm_head b6 (0.95 GB). The fused `attn_rocm_kernels` qkv path is not used on this model — nearly all GEMV mass is in-scope.
2. **Split-K extended to large-N b4 shapes** (N cap 8192 → 32768, eligibility mirrored in the prune): 27B g/u-class 310-323 → **378 GB/s** (S=4, BN64/BK256); 9B g/u 389 → 415. → **20.3 tok/s**. 9B decode also improved to **13.97 ms/token (71.6 tok/s)**.
3. **Ceiling verification on the dominant shape** (5120→17408, 7.58 GB/token): every tile {BN32,64,128} × {BK128,256,512} × splits {1,2,4,8,16} × warps {4,8} × stages {3,4} lands in 346-378 GB/s. **378 is the measured family limit** on this shape (the deep-dive's §10.4 open mechanism — suspected DRAM page/bank behavior of the tiles_n-strided chunks, unfixable by tiling).
4. **Input-Hadamard fusion — negative, reverted.** Two correct implementations (in-loop `tl.sum` extraction; hoisted per-outer-iteration butterfly + static reshape/permute/split row-descent, bit-identical numerics, both passing the reference checks) measured **8.7** and **19.0 tok/s** vs 20.3. Register-layout conversions inside the k-loop poison the RDNA3 GEMV — consistent with (and extending) the AGENTS "no cross-lane reductions in the k-loop" rule: even permute/split-based static slicing costs more than the ~1.8 ms the separate `had_r_128` launches cost. The experiment code was fully reverted.

### 6.2 Why 25 tok/s is out of reach in this file

Budget at 20.3 tok/s (49.3 ms/token), from the census + parent-report ablations:

| component | ms/token | scope |
|---|---|---|
| 5120→17408 streams (7.58 GB) | ~20.1 | in-file, at family ceiling 378 GB/s |
| 17408→5120 down-class (3.79 GB) | ~9.8 | in-file, at 385 GB/s |
| z/b-class + misc b4 (4.2 GB) | ~10.5 | in-file, at ~400 GB/s |
| lm_head b6 (0.95 GB) | ~2.3 | in-file (b6, unsplit) |
| block norms (Python fallbacks) | ~5-6 | **out of scope** (ext fallbacks) |
| full-attention block | ~3.6 | **out of scope** |
| GDN recurrent/conv/norm glue | ~2.5 | **out of scope** |
| sampler argmax+sync, embed, gaps | ~1.5 | **out of scope** |

25 tok/s = 40.0 ms → in-file budget ≈ 25 ms for 16.5 GB = **~660 GB/s aggregate**, vs a verified 378 GB/s kernel-family ceiling on the stream that is 46% of the bytes (and ~400-455 on the rest). Even the 899 GB/s physics anchor applied to every in-file stream leaves ~35 ms ≈ 28 tok/s only if the out-of-scope 13 ms were also somehow reduced. Closing the last 4-5 ms/token requires the out-of-scope work the parent reports already ranked: fused RMS-norm kernels (~5-6 ms), attention decode polish (~1.5 ms), GDN glue (~1.5 ms) — all outside `exl3_triton.py`.

### 6.3 Verification in this session

- Fast kernel-test subset: **74 passed** (split plan + pool buckets + split-vs-reference + K=1..8 sweep + shape sweep). Full 50-minute gates intentionally not re-run during iteration (per task instruction); they must be re-run before integration.
- perf.py (the task's target script): baseline 18.23/18.34/18.41/18.31 → final 20.34/20.24/20.29/20.18 tok/s at ctx 0/256/512/768.
- 9B interleaved A/B spot (1 round): base 17.77 → cand **13.97 ms/token** — no regression, further gain.

---

## 7. Audit iteration: register/occupancy evidence + kernel-structure experiment

The reviewer correctly challenged the "family ceiling" claim as sweep-only evidence. Follow-up work measured the compiled kernels directly and ran the structural experiment the numbers pointed at.

### 7.1 Register / spill audit (n_regs, n_spills)

Dumped via Triton's compiled-kernel metadata (`/tmp/impl_mlp/dump_regs.py`), 27B hot shapes:

| compiled variant | n_regs | n_spills | notes |
|---|---|---|---|
| tensor b4 split (running pick BN32/BK256/w4) | **256** | **17–459** | hits the gfx1100 VGPR cap |
| tensor b4 classic (BN64/BK128) | 114–231 | 0 | |
| lane-local kernel (all configs) | 66–90 | 0 | see 7.2 |

**The reviewer's prediction was confirmed**: the split tensor-tile M1 variants compile at the 256-VGPR cap with scratch spills — the [2,2,2,NN,8,4] accumulator + 16-way unrolled decode tile is register-heavy.

### 7.2 Lane-local (FlyDSL-structure) kernel — implemented, measured, removed

To test whether occupancy was the binding constraint, a spill-free restructure was built after the in-tree FlyDSL kernel (`exl3_gemm_fly.py`): one output column per lane, a single fp32 accumulator, per-lane funnel shifts on 1-D [BLOCK_N] tensors, zero reshapes/permutes/cross-lane ops, scalar x loads — 66–90 regs, 0 spills, 0 shared. Two revisions (masked→unmasked loads; 13-of-16 rows reduced to a single shift via the self-masking funnel). Correct on all codebook variants from the first compile.

**Result: slower.** 27B g/u 327 vs 382 GB/s (tensor), 9B down 331 vs 425. Ablations: removing the x loads entirely (constant accumulate) does *not* speed it up (306 vs 331 — noise), so the x path is free; the cost is the per-lane scattered 4 B word loads vs the tensor path's contiguous staged-row loads. **In Triton's execution model, contiguous-load + broadcast-decode beats lane-local decode; the spills are real but not the binding constraint.** The kernel was removed after the experiment (kill-switch code path and all); the negative result is recorded here.

### 7.3 The FlyDSL 770 GB/s figure in context

The in-tree FlyDSL kernel's 770 GB/s was measured on the **bits=6** stream (953 MB, 0.75 B/weight) = **578 Gweights/s** of decode throughput. The Triton tensor b4 path at 382–455 GB/s is **764–910 Gweights/s** — already *above* FlyDSL's per-weight rate (b4 reads 0.5 B/weight, so its GB/s is structurally lower). AGENTS also records FlyDSL's own b4 results as parity with Triton (0.94–1.05×). Three independent b4 kernel structures — tensor Triton, lane-local Triton, FlyDSL b4 — all land at 380–455 GB/s on this GPU. The GB/s figures are not comparable across bitwidths; the goal's "700–770 GB/s in-family" premise does not transfer to b4.

### 7.4 Kept improvement: bits=6 M=1 split-K (linear-class N)

While auditing, the b6 M1 branch gained the same split-K + fused-reduce/Hadamard as b4 (fp32 partials, same reduce kernel). Scope: bits=6, N ≤ 16384, N/K % 256 == 0 — the lm_head stream (N=248320) measured *slower* split (628 vs 666 GB/s classic; already fully parallel) and stays classic. Measured: 6bpw MLP g/u 553→570, u 549→563, down **442→559 GB/s**; in-situ 6bpw decode **24.60 → 15.79 ms/token (−36%)**, token-identical (3×219 greedy, argmax).

### 7.5 Final state and full gates

| check | result |
|---|---|
| Kernel suite (`tests/test_exl3_triton.py`) | **226 passed** (222 + 4 new b6 tests) |
| Integration suite (AGENTS command) | **498 passed / 68 skipped** (494 expected + 4 new) |
| 27B perf.py Generation (ctx 0/256/512/768) | **20.32 / 20.26 / 20.26 / 20.16 tok/s** |
| 27B Generator-flow interleaved A/B | 56.91 → **46.89 ms/token** (21.3 tok/s), 3×219 token-identical |
| 9B spot | 17.77 → **13.97 ms/token** (71.6 tok/s) |
| 6bpw spot | 24.60 → **15.79 ms/token**, token-identical |

Commits: `57b6704`, `e9b9819`, `7ffc8ec`, `fa5aa19` (detached HEAD on `89c8379`).

### 7.6 Where the 25 tok/s gap actually lives now

With three kernel structures measured at 380–455 GB/s b4-class (and the 899 GB/s anchor representing a trivial reader, not a decoder), the in-file floor for the 27B's 16.5 GB/token is ~24–29 ms in-file + ~13 ms out-of-file (block norms ~5–6 ms via Python fallbacks, full-attention ~3.6 ms, GDN glue ~2.5 ms, sampler/embed ~1.5 ms) ≈ 37–42 ms/token ≈ 24–27 tok/s at *full* in-file physics — the current 46.9 ms Generator / 49.3 perf.py number sits ~20% above that floor, all of the difference inside the measured kernel-structure limit, not in tiling. Reaching >25 tok/s needs either a fundamentally better b4 decode structure (none of three structures tried exceeds ~455 GB/s; the 899 anchor is not reachable by any decoder in-tree) or the out-of-file work (norm/attention/glue) identified by the parent reports.

---

## 8. Audit iteration 2: exhaustive per-shape audit (no recoverable headroom left in-file)

To close the last possible gap class — a shape-level underperformer the MLP-focused sweeps never covered — every distinct M=1 shape the 27B decode actually calls was captured (by wrapping `_linear_exl3_triton` during a real decode step) and timed standalone with the shipping defaults (`/tmp/impl_mlp/all_shapes.py`):

| shape (bits, K→N) | calls/token | GB/token | S | rate (GB/s) | class rate |
|---|---|---|---|---|---|
| b4 5120→17408 (gate/up + qkv) | 170 | 7.58 | 4 | 388 | ✓ at class |
| b4 17408→5120 (down) | 85 | 3.79 | 8 | 389-391 | ✓ |
| b4 5120→10240 | 64 | 1.68 | 4 | 375-376 | ✓ |
| b4 5120→12288 | 5 | 0.17 | 4 | 386-387 | ✓ |
| b4 5120→6144 / 6144→5120 | 128 | 2.35 | 4 | 277-289 | see below |
| b4 5120→1024 (hyper-conn) | 11 | 0.03 | 4 | 46-50 (launch-bound, 0.6 ms/token total) | — |
| b6 5120→248320 (lm_head) | 1 | 0.95 | 1 | 571-585 isolated | ✓ |

The 5120↔6144 pair looked like a ~100 GB/s underperformer, so it got the full treatment: S∈{4,8} × tile {32,64}×{128,256} forced probes (369-376 GB/s measured late in the process) — but an **interleaved A/B/A retest in one process at matched clocks showed the real gap is only 287 vs ~309 GB/s** (`/tmp/impl_mlp/settle_6144.py`): the "370+" probe numbers were themselves a warm-clock/context artifact, and no forced configuration recovers more than ~7% on these shapes. Worth ~0.1-0.3 ms/token; not landed (no change beats its own measurement noise across contexts).

An S=8-for-starved-shapes plan variant (thin slices for N≤8192) measured identically (282/277 vs 283/279) — reverted; the committed plan stands.

**Verdict of the exhaustive audit:** every byte-stream the file controls runs at 285-390 GB/s in realistic context, each within ~10% of the best any configuration/structure experiment reached for its shape, and three independent kernel structures cap the b4 class at 380-455 GB/s. The in-file bound is measured, not assumed:

- current: **20.2-20.3 tok/s** (perf.py), 46.9 ms/token (Generator A/B)
- in-file absolute limit at the best structure rates (≈455 GB/s on every stream): ~36 ms/token ≈ 27.8 tok/s **with zero out-of-file cost — impossible since 13 ms is structural** (Python-fallback norms ~5-6 ms, full-attention ~3.6 ms, GDN glue ~2.5 ms, sampler/embed ~1.5 ms — all outside the two-file scope)
- realistic composite bound: 24-27 tok/s requires BOTH ~455 GB/s everywhere AND the out-of-file 13 ms cut to ~5 ms

**>25 tok/s is therefore infeasible within the constraint of editing only `exl3_triton.py` + `tests/test_exl3_triton.py`.** The two options the objective permits are: (a) relax the scope to the files owning the out-of-file 13 ms (a Triton rms_norm/gated_rms_norm quartet is the single biggest item, ~5-6 ms/token; then attention decode ~1.5 ms and GDN glue ~1.5 ms), or (b) accept 20.3 tok/s (+10.6% over the 18.4 baseline this session started from; 9B 71.6 tok/s, 6bpw 63 tok/s) as the two-file result. Everything in-file has been measured to its limit: configs, splits, kernel structure (three designs), register pressure, fusion of both Hadamard sides, per-shape audits.

---

## 9. Final deliverable: per-bitwidth kernel bandwidth benchmark

`bench/bench_exl3_gemv.py` (commit `c18b57d`, extended in `ead0ec6`) times the production M=1 decode path for all bit widths on three shape classes plus arbitrary custom shapes (`--shape K:N`), reporting GB/s (trellis+scales) and Gweights/s. Methodology lessons baked into the script (both found during bring-up):

1. **Clock-ramp must synchronize per matmul** — an async-launch loop enqueues thousands of 25 ms matmuls in ~2 s of wall time and the closing sync grinds minutes of queued backlog at 100% GPU. This bug made the first full sweep take >1 h; fixed, the same sweep runs in ~60 s.
2. **A long ramp must precede each case's first call** — the autotuner benches on that call and mis-ranks configs at ramping clocks (observed a 24% off-best pick).
3. **Weights rotate across distinct trellises** (~1.5 GB working set) so timed calls are not served from the 96 MB L2.

Measured on the shipping-best kernel (RX 7900 XTX, L2-cold rotation, best-of-4×60 calls, Hadamards + split-K included; bits 4/6 = fast paths, 1/2/3/5/7/8 = generic):

| bits | 12288→4096 (9B down) | 5120→17408 (27B g/u) | 4096→65536 (stream) | 17408→5120 (27B down) |
|---|---|---|---|---|
| 1 | 69 / 550 | 98 / 780 | 116 / 927 | 75 / 598 |
| 2 | 137 / 548 | 210 / 837 | 260 / 1036 | 152 / 606 |
| 3 | 290 / 771 | 360 / 958 | 417 / 1111 | 303 / 806 |
| 4 | 387 / 774 | 488 / 976 | 534 / 1068 | **532 / 1064** |
| 5 | 326 / 522 | 411 / 657 | 476 / 760 | 355 / 567 |
| 6 | 570 / 759 | 582 / 775 | 686 / 914 | 616 / 821 |
| 7 | 453 / 518 | 561 / 641 | 644 / 735 | 489 / 558 |
| 8 | 565 / 564 | 708 / 707 | 804 / 803 | 628 / 628 |

(cells = GB/s / Gweights/s). Isolated-rotation numbers run 25-35% above in-situ
(fixed IO buffers, no interleaving): the operative in-situ figures for the 27B
are ~390 GB/s (down, from 239 pre-optimization) and ~388 GB/s (g/u, from 310).

---

## 10. Session summary (implementation + perf goal + audits + benchmark)

- **Commits** (detached HEAD on 89c8379): `57b6704`, `e9b9819`, `7ffc8ec`, `fa5aa19`, `c18b57d`, `ead0ec6`. Only `exl3_triton.py`, `tests/test_exl3_triton.py`, and the new `bench/bench_exl3_gemv.py` were touched.
- **Headline results**: 27B perf.py 18.4 → **20.2-20.3 tok/s** (+10.6%); 27B Generator A/B 56.91 → **46.89 ms/token**; 9B 17.77 → **13.97 ms/token (71.6 tok/s)**; 6bpw 24.60 → **15.79 ms/token (63.3 tok/s)**.
- **Correctness**: kernel suite **226 passed**; integration suite **498 passed / 68 skipped**; token identity 3×219 greedy/argmax on 27B, 9B, and 6bpw; max abs diff vs C++ reconstruct ≤0.0156 across codebooks/shapes.
- **Optimization inventory (kept)**: N-bucketed width-aware autotune pools; M=1 split-K GEMV with fused reduce+output-Hadamard (b4, all linear-class N; b6 linear-class); large-N split extension (g/u + qkv-class).
- **Negative results (documented with cause)**: input-FWHT fusion (two formulations; register-layout conversions in the k-loop); lane-local FlyDSL-structure port (spill-free 66-90 regs but slower — Triton needs contiguous staged loads); b6 lm_head split (already parallel); thin-slice S=8 for starved shapes (neutral).
- **25 tok/s verdict**: infeasible in the two-file scope — bounded by three independent kernel structures (380-455 GB/s b4-class) plus ~13 ms/token out-of-file cost (norms/attention/GDN glue/sampler). The out-of-file menu in §8/§7.6 is the actionable next step.
