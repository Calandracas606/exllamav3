# FlyDSL RDNA3 WMMA GEMM as EXL3 prefill accelerator — measurement report

Branch: `fly-prefill` (from `rocm-flydsl`), all work uncommitted. RX 7900 XTX (gfx1100),
ROCm 7.14, torch 2.12+rocm, flydsl wheel 0.3.1 + vendored repo kernels. 5 s clock-ramp
before every timed region.

## Verdict: DO NOT ADOPT for prefill. Verified negative with numbers.

The FlyDSL WMMA GEMM itself is excellent at small M (1.4-1.5x over rocBLAS at M=128) but
prefill never runs at M=128 — generator chunks are PAGE_SIZE=256 multiples (a 2048-token
prompt prefills as 1792 + 255). At those M values the incumbent (reconstruct + hgemm,
which is what LinearEXL3.forward actually runs for rows > 144) is within a few % of the
best possible dense pipeline, and the candidate's extra dequant-to-[N,K] pass makes it
slower. End-to-end headroom from a perfect GEMM-engine swap is ~10%, not 1.5x.

## Phase 1 — dense GEMM: vendored WMMA (tile-swept) vs rocBLAS, f16 in/out

| K x N | M | rocBLAS ms | rocBLAS TF/s | fly ms | fly TF/s | best tile | fly f32-out ms |
|---|---|---|---|---|---|---|---|
| 5120x17408 | 128 | 0.690 | 33.1 | **0.488** | **46.7** | 128x64x64 w128 | 0.531 |
| 5120x17408 | 512 | 0.955 | 95.6 | 1.019 | 89.6 | 128x128x32 w128 | 1.236 |
| 5120x17408 | 2048 | 3.715 | 98.3 | 3.776 | 96.7 | 128x128x32 w128 | 3.770 |
| 17408x5120 | 128 | 0.482 | 47.3 | 0.512 | 44.6 | 64x128x64 w128 | 0.595 |
| 17408x5120 | 512 | 1.014 | 90.0 | 1.177 | 77.5 | 64x128x64 w128 | 1.389 |
| 17408x5120 | 2048 | 3.756 | 97.2 | 3.906 | 93.5 | 128x128x32 w128 | 3.907 |
| 5120x248320 | 128 | 6.233 | 52.2 | **4.252** | **76.6** | 64x128x64 w128 | 4.104 |
| 5120x248320 | 512 | 14.675 | 88.7 | 14.259 | 91.3 | 128x128x32 w128 | 14.174 |
| 5120x248320 | 2048 | 60.104 | 86.6 | **56.854** | **91.6** | 128x128x32 w128 | 57.532 |

- 36-tile sweep per shape (`bench/bench_phase1_fly_gemm.py`, `bench/phase1_results.json`).
- B_T layout [N, K] is exactly what dequantized EXL3 weights can produce for free *if the
  dequant kernel writes n-major* — algebraically fine (the (r,c)->(j,t) bijection factors
  per-axis), but on RDNA3 the resulting 16-element strided stores cost 2.6-5.4x vs
  ext.reconstruct's coalesced [K, N] streaming layout (measured below).
- f32-out is not faster than f16-out (store-bound epilogue unchanged).

## Phase 2 — EXL3 prefill paths, end-to-end had->gemm->had, dequant included

`bench/bench_phase2_prefill.py`, `bench/phase2b_results.json`:

| shape (bits) | M | fused Triton | recon+hgemm (incumbent) | fly cold | fly hot (cached W) |
|---|---|---|---|---|---|
| 4b 5120x17408 | 128 | 0.83 | 0.84 | 1.76 | 0.54 |
| 4b 5120x17408 | 256 | 1.62 | **1.13** | 2.31 | — |
| 4b 5120x17408 | 1792 | 11.22 | **4.11** | 5.21 | — |
| 4b 5120x17408 | 2048 | 12.88 | **4.68** | 5.61 | 4.13 |
| 4b 17408x5120 | 256 | 1.69 | **0.99** | 1.62 | — |
| 4b 17408x5120 | 1792 | 11.39 | **4.25** | 4.84 | — |
| 6b 5120x248320 | 256 | 30.95 | **12.68** | 53.99 | — |
| 6b 5120x248320 | 1792 | 217.10 | **57.42** | 96.15 | — |
| 6b 5120x248320 | 2048 | 248.39 | 65.10 | 103.04 | 59.87 |

- The incumbent at M > 144 is reconstruct+hgemm (exl3.py dispatch), *not* the fused Triton
  kernel (which only covers rows <= AUTO_RECONSTRUCT_THRESHOLD = 144 via BC_LinearEXL3).
- fly-hot (dense W cached in VRAM) beats everything, but caching all 27B dense weights
  needs ~54 GB — impossible on 24 GB. With the LRU budget evicting every call (real
  chunked prefill walks all 48 layers per chunk), the honest number is fly-cold: loses
  at every shape and M.
- Root cause: my dequant-to-dense [N,K] kernel writes at 130-266 GB/s vs ext.reconstruct's
  427-698 GB/s. Manual 72-config sweep at hot clocks (BN/BK/warps/stages) closed only
  part of the gap (1.37 -> 0.84 ms at 4b 5120x17408 vs 0.52 recon; 46.5 -> 26.9 ms vs
  5.0 at the 6b lm_head). The [K,N]-major trellis -> [N,K]-major dense transpose store
  is inherently less coalesced on RDNA3.

## Where 27B prefill time actually goes (2048-token prompt, this branch)

- Wall: 4.15-4.20 s = **~490 tok/s** (2 chunks: iterate() calls 3520 ms + 676 ms).
- Module-hook event profile: L:exl3 linears ~97% of hooked time, BUT the accounted pure
  GEMM+dequant GPU work (sum over the model inventory from Phase 2 numbers) is ~1.6 s;
  the rest of the linear time is eager dispatch overhead + W-materialization traffic.
  Attention 306 ms, GDN core 1.8 s hook-inflated (includes launch gaps), RMSNorm 64 ms.
- Consequence: even a GEMM engine that is INFINITElY fast caps prefill at ~2.3x; the
  realistic 10% linear-time delta from the best dense pipeline projects to ~1.05-1.1x.
  The 1.5x goal is not reachable by swapping the GEMM engine. (Prefill dispatch/CPU
  overhead and the per-chunk W materialization are the actual levers — different task.)

## Decode spot-check (256 tokens)

- 66.55 ms/token steady (15.0 tok/s), dispatch-only 66.61 ms — identical across two runs.
- Note on baselines: this branch (rocm-flydsl) does NOT contain attn_rocm_kernels.py /
  gdn_ba_gemm.py / aiter (those live on attn-decode-opt / gdn-ba-gemv / rocm-aiter lines);
  its decode baseline is the plumbing-era ~15 tok/s, which is exactly what we measure.
  My work adds only untracked files and wires nothing (EXL3_PREFILL_FLY default OFF,
  no tracked file references the new module), so decode is unchanged by construction and
  by measurement. The 17.6 tok/s figure is the integration-branch baseline.

## Suite

`pytest tests/ <same ignores>`: **370 passed / 66 skipped** (matches the gate).
`tests/test_exl3_gemm_triton.py`: **111 passed**.

## Vendored files (Apache-2.0, SPDX headers kept), API notes

- `exllamav3/_flydsl_kernels/`: `rdna3_f16_gemm.py` (446 L), `kernels_common.py` (140),
  `mem_ops.py` (182), `buffer_ops.py` (646) — from ROCm/FlyDSL main@47009a3
  (kernels/gemm + kernels/common closure). Only change: `from kernels.common.*` ->
  relative imports. Zero API mismatches vs wheel 0.3.1 — they compile and run as-is.
- **There is no `rdna3_f16_gemm_autotune.py` in the repo** (docstring mentions it, file
  doesn't exist; only rmsnorm/conv3d have autotune modules). The wheel's generic
  `flydsl.autotune.Autotuner` can't drive it either — it injects kwargs into `@jit`
  functions, while the GEMM is a compile-time module factory. Tile selection is a
  hand-rolled sweep (this repo: `bench/bench_phase1_fly_gemm.py`).
- `flydsl-src/` is a fresh shallow clone kept for reference (safe to delete).
- `exllamav3/exl3_gemm_triton.py` + `tests/test_exl3_gemm_triton.py` are pristine copies
  from the triton-kernels branch (needed to benchmark the incumbent on this line; per the
  file-slice rule they belong on triton-kernels, left uncommitted here).
- `exllamav3/exl3_prefill_fly.py` (411 L): the candidate path — Triton dequant-to-dense
  [N,K] + vendored WMMA GEMM + tile heuristic. All 33 correctness configs pass vs the
  reconstruct reference (rtol 2e-2 / atol 0.5, incl. M=513 padding fallback and b4/b6 x
  cb0/1/2). NOT wired anywhere (env `EXL3_PREFILL_FLY` checked per call, default off;
  module never imported by default paths).

## Where the FlyDSL GEMM WOULD win (if the workload ever appears)

- M <= 128 dense GEMMs: 1.4-1.5x over rocBLAS (46.7 vs 33.1 TF/s at 5120x17408; 76.6 vs
  52.2 at the 6-bit lm_head shape). That regime is real for *decode-sized batches* and
  short-chunk speculative verification, not for this generator's 256-page prefill chunks.
