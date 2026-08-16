# exllamav3 ROCm/Triton work notes

## BRANCH STRUCTURE AND WORKFLOW (read this first)

This fork uses a **stacked-branch structure** managed with git-spice (v0.31.2, `gs` on PATH).
git-spice is used for LOCAL stack management only: branch bases, `gs upstack restack`,
`gs up`/`gs down`, `gs log`. **All pushes are explicit `git push` commands, and all PRs are
opened manually by the maintainer. Never run `gs branch submit`.**

```
upstream/dev (trunk)
├── rocm-gfx1100-integration   ◄── OPEN UPSTREAM PR — FROZEN, READ-ONLY, NEVER PUSH
│     5 commits: HIP compat, build system, ext.py fallback dispatch, test fixes, README
│   └── rocm-plumbing          ROCm glue: bc_rocm.py, block_graph_rocm.py, model.py hook,
│       │                         exl3.py/gated_delta_net.py/attn.py/triton_paged.py wiring
│       ├── rocm-aiter         aiter_kernels.py (Triton RMSNorm bridge) + ext.py hook
│       └── rocm-flydsl        exl3_gemm_fly.py — FlyDSL dequant+GEMV (opt-in,
│                                EXL3_GEMM_FLY=1, default OFF; beats Triton on the
│                                6-bit lm_head stream; optional flydsl wheel dep)
└── triton-kernels             modules/quant/exl3_triton.py + tests/test_exl3_triton.py
                                (the ONLY upstreamable kernel line; plain functions,
                                no torch.library; EXL3_PREFER_TRITON_LINEAR=1)

integration = merge(rocm-aiter, rocm-flydsl, triton-kernels) + this AGENTS.md + fork README
                (attn_rocm_kernels.py / gdn_ba_gemm.py / bc tests live on rocm-plumbing)
```

### Hard rules

1. **NEVER push/rebase/amend `rocm-gfx1100-integration`** — it is the open upstream PR.
   Read it, branch from it, nothing else. If the maintainer updates it during PR review,
   rebase the plumbing line onto the new tip (`git rebase --onto <new> <old> rocm-plumbing`)
   then restack (`gs upstack restack`).
2. **File-slice rule** — which line a change belongs on:
   - `modules/quant/exl3_triton.py`, `tests/test_exl3_triton.py` → **triton-kernels**
     (upstreamable; CUDA-testable; plain-function calls, NO torch.library ops —
     the author rejects op registration for dispatch-overhead reasons)
   - Anything gated on `torch.version.hip`, ext.py dispatch, bc_rocm/block_graph_rocm,
     call-site wiring in exl3.py / gated_delta_net.py / attn.py / triton_paged.py / model.py
     → **rocm-plumbing**
   - AITER bridges → **rocm-aiter**
   - AGENTS.md, fork README → **integration only**
3. **Kernels must stay importable on CUDA** (no torch.version.hip dependencies in kernel
   files). The plumbing line must stay valid WITHOUT the kernels line merged (guards in
   ext.py / exl3.py / attn.py handle this — extend that pattern when adding wiring).
4. **integration is the only branch that is tested end-to-end** (full pytest suite +
   generation-identity + bench_decode). Lines get targeted tests only.

### Everyday workflow (agent loop)

- Agents get worktrees cut from `integration`'s tip (complete state → correct behavior).
- Agents never run `gs`, never push, never commit unless told.
- After verification, distribute the agent's commits by the file-slice rule, then:
  `git checkout integration && git merge rocm-aiter && git merge triton-kernels`
- After a mid-stack edit: `gs upstack restack` on the affected line, re-verify integration.

### Verification gates (mandatory after any restructure/restack)

1. `pytest tests/ --ignore=tests/test_dsv4_cached.py --ignore=tests/test_dsv4_state.py
   --ignore=tests/test_qgemm.py --ignore=tests/test_quant_fn.py
   --ignore=tests/test_recurrent_checkpoint.py --ignore=tests/batch_test_model.py
   --ignore=tests/generator_stresstest.py --ignore=tests/test_model.py
   --ignore=tests/test_ext_norm_.py --ignore=tests/test_mla.py
   --ignore=tests/test_dsa_kernels.py --ignore=tests/test_reconstruct_had.py
   --ignore=tests/test_dsv4_compress_kernel.py
   --ignore=tests/test_triton_paged_overflow.py -q` → expect **370 passed / 66 skipped**
2. Generation identity: 3 prompts × ≥200 tokens, argmax, whole-step graphs on, every
   env kill switch toggled, vs the previous verified state.
3. `bench_decode.py --model 27b --num-tokens 512` — median windowed steady-state.

### Performance history (Qwen3.6-27B, RX 7900 XTX, decode)

| State | tok/s | ms/token |
|---|---|---|
| reconstruct+hgemm fallback (project start) | 4.5 | 222 |
| + Triton 4-bit fast path | 12.0 | 83 |
| + 6-bit fast path + whole-step graphs | 15.0 | 66.6 |
| + fused b/a GEMV | 17.4 | 57.6 |
| + fused attention kernels | ~17.6 | ~57 |

## Environment (RX 7900 XTX, gfx1100, ROCm 7.14, torch 2.12+rocm, triton 3.7.1)
- Main repo venv: `/tmp/conversation-worktrees/395af3ac-0a32-4b2a-b348-819aab4fe32a/exllamav3/.venv`
  (worktrees have no venv). Source env with ROCM_HOME/LD_LIBRARY_PATH, else
  `import exllamav3` JIT-builds the C++ ext with broken absolute paths.
- GPU clocks idle at 398 MHz and ramp slowly (~100+ ms). Any microbenchmark
  needs a long warmup (50+ iterations or several seconds of dummy matmuls)
  or results read 1.5-2x slow. Autotune passes run at ramping clocks and can
  mis-rank configs; keep config lists tight so any pick is near-best.

## EXL3 trellis dequant structure (bits=4) — verified algebra
- Weight W[r, c] of a 16x16 trellis subtile = D[j, t] codebook entry where
  (via `_get_perm` inverse + `_dq_indices`):
    t(r,c) = 4*(c%8) + (r%8)//2        (trellis word index)
    j(r,c) = 4*(c//8) + 2*(r//8) + r%2 (shift index: shift = 28 - 4j)
  This (r,c) <-> (j,t) map is a BIJECTION and factors into independent
  per-axis bits: j = 4ch+2rh+p, t = 4cl+q, r = 8rh+2q+p, c = 16nj+8ch+cl.
  => the permutation can be realized by static reshape/permute (no gather),
  or, for GEMV, folded into a broadcast pattern of the x vector.
- The m1 (neighboring word, t-1) only matters for j < 3; loads of the row
  shifted by -1 word are linear/vectorizable, wrap fixed in registers with
  one tiny [NN] load of word 31.
- Cross-lane `tl.sum` reductions INSIDE the k-loop are poison for RDNA3
  (~2x slowdown); accumulate a full product tile elementwise and reduce
  once after the loop.
- tl.gather exists in triton 3.7 and lowers to LDS ops on AMD, but LLVM
  aborts (Fatal error in make_amdgcn) for gather tiles larger than ~[16, 64]
  with 8 warps — prune such configs for the generic path.

## EXL3 trellis dequant structure (bits=6) — verified algebra
- Element e = 4*tg + jj with tg = 4a + b (jj = e%4, b = (e//4)%4, a = e//16).
- Word pair: word(u), word(u-1 mod 48) where u = 3a + f(b), f = [0,1,2,2]
  (the "high" word is always low-1 mod 48).
- Shift: s = C_b - 6*jj with C = [26, 34, 42, 18] (b=2 and b=3 share the same
  word pair, differing only by shift 24).
- Output position (the _get_perm permutation, verified bijective bit-field map):
    r = 8*j1 + 4*(a&1) + 2*(b>>1) + j0
    c = 8*(b&1) + (a>>1)
  where e's bits are 32*cl + 16*(a&1) + 8*(b>>1) + 4*(b&1) + 2*j1 + j0.
- Only 4 linear word-slice loads (word 3a, 3a+1, 3a+2, 3a-1) feed all 256
  elements of a tile; the funnel's s>=32 case flips base/second word via tl.where.
  M=1 folds the permutation into the x-broadcast (4 separate (j1,j0,nj,cA,a0)
  fp32 accumulators, reduced once after the K-loop); M>1 reorders via static
  reshape/permute/join into a [16, BLOCK_N] tile for tl.dot.
- Results (branch triton-6bit-fastpath): lm_head shape M=1 K=5120 N=248320
  bits=6: 11.0 ms (87 GB/s) -> 1.29 ms (738 GB/s), 8.5x. Faster per byte than
  the 4-bit path (6-bit packs more weights per trellis byte read).
- early_config_prune admits bits in {4,6} to large tiles (both fast paths are
  gather-free); the generic tl.gather cap stays for bits 1/2/3/5/7/8 and
  non-divisible shapes.

## Results (branch triton-kernel-opt, file exllamav3/exl3_gemm_triton.py)
- M=1 K=5120 N=17408 bits=4: 0.645 ms (69 GB/s) -> 0.088-0.095 ms (~490-507 GB/s)
- All 111 tests in tests/test_exl3_gemm_triton.py pass.

## Whole-step CUDA graph capture (branch bc-block-graph-capture, exllamav3/block_graph_rocm.py)
- One torch.cuda.CUDAGraph per (cache, block-table-width, recurrent-slots,
  last_tokens_only) covering first transformer block -> lm_head. CPU embedding
  runs outside (prefer_cpu). cache_seqlens/positions/block_table are copied
  pinned->persistent device buffers each replay. Kill switch: EXL3_BLOCK_GRAPHS=0.
- CRITICAL: no side-stream warmup *execution* before capturing a STATEFUL region
  (GDN conv/recurrent state, KV writes are destructive). Re-running the step
  double-advances state and silently corrupts generation (verified). The 8
  uncaptured decode warmup steps are enough; capture records but never executes.
- Bit-identical argmax generations on/off (verified 300+ tokens, 9B and 27B).
- Decode is NOT CPU-bound on this box: baseline 27B decode has GPU ~100% busy
  (rocm-smi --showuse). Real per-token GPU time ~75-84 ms (four measurement
  methods agree; torch.profiler UNDER-counts on ROCm - distrust its device
  totals, use sync-bracketed or tight-loop timing instead).
- The "474-507 GB/s" GEMV number above is an L2 artifact: 47 MB test weights fit
  the XTX's 96 MB L2, so back-to-back same-kernel benches measure L2 bandwidth.
  In-situ decode streams 13.6 GB/token from DRAM at ~250-330 GB/s.
- Kernel-level floors measured (27B, tight-loop): GDN block 1.16 ms, attn block
  0.99 ms, lm_head 11.0 ms (!). lm_head is 6-bit -> generic tl.gather GEMV path
  at ~87 GB/s; early-prune caps tiles to 64x64 so EXL3_GEMM_CONFIGS can't fix
  it. rocBLAS GDN b/a GEMVs cost 2x93 us x 48 layers ~ 9 ms/token.
- Whole-step vs per-block graphs on gfx1100/ROCm 7.14: identical wall (~76 ms;
  2496 nodes) - graph-node execution has ~14 us/node frontend cost here, so
  graphs cut CPU dispatch (28 ms -> 0.2 ms) but cannot cut GPU-frontend gaps.
- pytest needs EXL_TEST_DEVICE=cuda:0 on this 2-GPU box (default cuda:2 fails
  with "invalid device ordinal").

## GDN b/a projection GEMV (branch gdn-ba-gemv, exllamav3/gdn_ba_gemm.py)
- gdn.cu already contains a merged b/a GEMV kernel (gdn_ba_gemv, used by the
  C++ BC_GatedDeltaNetSplit path) and it IS compiled on ROCm; but
  gated_delta_net_fused_op_3 (which consumes its packed output) has no Python
  binding, so the ROCm path used 2x rocBLAS GEMVs + fused_op_2 (~200 us/layer
  in situ, 9.6 ms/token over 48 layers).
- Fix: Triton kernel gdn_ba_beta_g in exllamav3/gdn_ba_gemm.py — one launch
  computes x@[W_b;W_a] AND the fused_op_2 epilogue (beta bf16 / g fp32),
  reusing the ba_weight_t merged buffer the BC path already fills. 11.4 us/
  layer (0.55 ms/token). Gated in gated_delta_net.py on torch.version.hip,
  rows<=16, x contiguous half; kill switch EXL3_GDN_BA_GEMV=0.
- beta comes out bit-identical to the old path (bf16 absorbs dot-order noise);
  g differs ~1e-5 (fp32 reduction order) — 200-token argmax generations
  identical on 9B and 27B.
- 27B decode: 14.99 -> 17.1 tok/s (66.7 -> 58.5 ms/token). Kernel is
  latency/launch-bound (~12 us regardless of BLOCK_K/warps); no autotune.
- Per-GDN-layer in-situ stage times (27B, events during real decode, graphs
  off): qkv_proj 116 us, z_proj 75, b/a 11 (was 201), transpose+cast 8,
  conv1d 11, recurrent 19, gated RMSNorm 56 (!), o_proj 95. Total 393 us/layer
  = 18.9 ms/token over 48 layers.
- NEXT targets, in order: (1) gated RMSNorm runs the 8-kernel torch fallback
  (ext_fallbacks.gated_rms_norm) at 56 us/layer = 2.7 ms/token — a one-kernel
  Triton replacement is nearly free to write; (2) the three EXL3 GEMVs
  (qkv/z/o = 287 us/layer = 13.8 ms/token) are the real floor; (3) the
  recurrent kernel is only 19 us/layer — NOT the bottleneck (the ~1.16 ms
  "GDN block" floor in earlier tight-loop measurements includes eager overhead).



## Full-attention decode decomposition + optimization (branch attn-decode-opt)
- Methodology upgrade: cross-session tok/s varies +-1.5% (clock/power state) and
  CPU load (YouTube, compiles) can cost 15%+. ONLY trust in-process interleaved
  A/B (ab_bench.py: same model load, toggling paths per round) and ablation
  deltas (Attention.forward monkeypatched to zeros gives the "no-attn" rate).
  Eager per-kernel event timings also include CPU launch time when the GPU
  idles - kernel-proxy numbers with graphs off overstate kernel cost.
- 27B attn layer: 24 q heads / 4 kv heads / head_dim 256, q_norm+k_norm fused
  into ext.rope, interleaved q|g gate in q_proj (N=12288 incl gate), k/v
  N=1024 each, o_proj N=5120 K=6144, all EXL3 bits=4 CB=2. 16 such layers.
- Breakdown (eager, graphs off, per layer): q_proj 116 us, deint 25, k+v 111,
  rope 11, attn_dispatch 192 (kv_update 9 + split 35 + combine 25 pure-kernel,
  rest gaps), gate 14, o_proj 96. Pure decode GEMV kernel times (in-situ):
  q 100, k 69, v 69, o ~75-155 -> attention block total 5.28 ms/token
  (ablation, graphs on).
- Root causes fixed (all bit-identical to the reference path, verified by
  torch.equal on every output tensor):
  1. k/v GEMVs launch only 8-16 CTAs (N=1024) -> latency-bound at ~4 GB/s.
     NEW exllamav3/attn_rocm_kernels.py: had3 (3 sign-vector input Hadamards,
     one launch) + gemv3 (q+k+v GEMV in ONE launch, BLOCK_N=128, output
     Hadamards fused in-register in the epilogue, interleaved q|g store
     remapped per CTA - the deinterleave kernel is eliminated). Wired in
     attn.py (_project_qkv_fused_rocm), gated on torch.version.hip + geometry
     checks, kill switch EXL3_ATTN_QKV_FUSED=0. DRAM-cold interleaved A/B:
     68.5 us vs 125 us for the 3 reference projections (1.8x).
  2. o_proj via a direct exl3_ops::LinearEXL3_triton call into persistent
     buffers (no per-linear BC copy_ + clone): 2 fewer nodes per call.
  3. Paged attention: the combine kernel ran 4 CTAs (one per kv head)
     reading 2 MB of partials -> row-split grid (16 CTAs), per-element
     split-sum order unchanged (bit-identical). Plus
     _paged_attn_decode_fused_kernel: split+combine in one launch with an
     atomic last-block reduction (acq_rel counter, self-resetting); partials
     land in L2 and are re-read hot. Gate: torch.version.hip, fp16 cache, no
     sinks, kill switch EXL3_PAGED_FUSED=0.
- Result (in-process A/B + ablation, idle machine): attention block
  5.28 -> 4.36 ms/token (-17%), full model 58.90 -> 57.89 ms/token
  (17.0 -> 17.3 tok/s in that session; the 17.36 baseline was measured in a
  faster session - absolute tok/s is only comparable within one session).
- FLOOR ANALYSIS (why 18.5 tok/s is not reachable in the attention bucket):
  906 MB/token attention weights (q 31.5 + k 2.6 + v 2.6 + o 20 MB per layer
  x16) + ~38 MB KV = 944 MB/token. Measured block cost 4.36 ms = 216 GB/s
  effective. Best demonstrated rate for these dequant-GEMV kernels DRAM-cold
  standalone (hot clocks): 480-630 GB/s; in-situ (dirty L2 from 13.6 GB/token
  streaming) halves that. Even at 500 GB/s sustained the block would cost
  ~2.2 ms -> best case ~17.9-18.0 tok/s whole-model. 18.5 needs 590 GB/s
  in-situ - beyond anything measured on this GPU for 4bpw trellis streams.
  The GEMV bandwidth itself lives in exl3_gemm_triton.py (read-only here).
- Whole-model stream rate: 13.6 GB / 57.9 ms = 235 GB/s (~1/4 of peak DRAM);
  the systemic bottleneck is the dequant-GEMV access pattern + small-kernel
  latency, not attention-specific structure.
- Trellis-layout experiment (removed): permuting the q/k/v trellises to
  [n_subtile, k_tile, words] (bit-identical values, n-major streaming) was
  verified correct but measured NO faster in situ (3-way A/B: 57.91 vs
  57.98 ms/token, within noise). The ~270 GB/s in-situ GEMV rate is the
  DRAM/queue behavior of the trellis streams, not a coalescing artifact.
  Private-graph kernel microbenches are UNSOUND on ROCm (independent nodes
  overlap inside a graph, evict-kernel subtraction goes negative) - only
  in-model A/B + ablation are trustworthy.
- cuda:1 on this box fails in the model LOADER (async OOM reported at
  _decode_lut's arange during weight load; the same op works in isolation).
  Pre-existing, unrelated to NCCL/RCCL; keep everything on cuda:0.

## FlyDSL EXL3 GEMV (branch rocm-flydsl, exllamav3/exl3_gemm_fly.py)
- FlyDSL (ROCm's Python MLIR DSL) is VIABLE on this box: pip wheel flydsl==0.3.1
  installs into the main venv via uv (venv has no pip), JIT-compiles for gfx1100
  out of the box, launches via DLPack tensors + raw stream ptr (no pybind shim).
  Env: EXL3_GEMM_FLY=1 opt-in hook in modules/quant/exl3.py (M==1 only). flydsl
  cache: ~/.flydsl/cache.
- Kernel: thread owns one column PAIR (c, c+8) of a 16x16 subtile (shares
  cl/cA -> 5-word (b4) / 7-word (b6) window), decodes 32 codes/k-tile fully
  unrolled (shifts are v_alignbit_b32 immediates), 2 fp32 accumulators, ZERO
  cross-lane ops (no tl.sum, no LDS). 1-deep prefetch of row k+1.
- GOTCHAS learned (cost hours):
  1. T[row, col] on a 2D memref expects the DIM0 INDEX as row — passing a
     pre-flattened element offset double-applies the row stride and reads OOB
     (symptom: k-tile 0 bit-exact, k>=1 garbage; one-hot x in tile 0 masked it).
  2. v_alignbit_b32 d, $1, $2, imm == (($1<<32)|$2) >> (imm & 31): $1 is HIGH,
     $2 LOW, and the IMMEDIATE shift is masked to 5 bits. s>=32 needs the exact
     hi >> (s-32) fallback. Verified by probe kernel.
  3. Loop-carried values through the AST rewriter: python lists DO work (pytree
     iter_args), but a name defined under `if const_expr(...)` in one block and
     consumed under a different `if const_expr(...)` block hits UnboundLocal at
     trace time — keep carried names unconditional.
  4. mul1 (cb=2) decode must match C++ __hfma exactly (ONE rounding): compute
     h*k_inv+k_bias as fp32 fma of the exact fp16 operands, then round to fp16.
     mul-then-add in fp16 (two roundings) leaves ~0.004/weight bias that
     accumulates over K=5120 to ~0.6 — beyond atol 0.5.
- Results (5s clock ramp + interleaved in-process A/B, full linear incl. both
  hadamards): b4 5120x17408 (44.6MB): parity (0.94-1.05x). b4 2048x5120: 1.01x.
  b6 5120x248320 (953MB): triton 1.400 (681 GB/s) / fly 1.238 ms (770 GB/s) =
  fly 1.13x FASTER (stretch met). Fly kernel is CUDA-graph capturable
  (capture+replay verified). All 7 correctness configs (b4/b6 x cb0/1/2, both
  shapes) PASS vs reconstruct reference; kernel suite still 111 passed with the
  hook present. End-to-end 9B generation verified with EXL3_GEMM_FLY=1.
- Verdict: FlyDSL is a credible alternative for this kernel class; the inline-
  asm escape hatch (v_alignbit_b32) and static shapes make it expressive
  enough, and it BEATS Triton on the L2-exceeding bits=6 stream where DRAM
  efficiency dominates. Triton stays better for tiny launch-bound shapes.
  flydsl wheel (72 MB) is an OPTIONAL dep — nothing imports it unless
  EXL3_GEMM_FLY=1.

## FlyDSL WMMA GEMM as prefill accelerator (branch fly-prefill) — NEGATIVE, do not retry
- Full report: bench/REPORT_fly_prefill.md on the fly-prefill branch (vendored
  kernels + tested-but-unwired candidate kept there; nothing in the active lines).
- The production GEMM itself is good: 1.4-1.5x over rocBLAS at M=128 (46.7 vs
  33.1 TF/s), parity at M>=512. But prefill never runs at M=128 — generator
  chunks are PAGE_SIZE=256 multiples, and for rows>144 LinearEXL3 dispatches to
  reconstruct+hgemm (coalesced [K,N] at 427-698 GB/s), which wins at every real
  chunk size. Dequant to the WMMA [N,K] layout costs 2.6-5.4x (strided stores).
- Ceiling: GPU GEMM+dequant is only ~1.6 s of the 4.15 s prefill wall (2048
  tokens, ~490-590 tok/s) — prefill is dispatch- and materialization-bound.
  THE prefill lever is graph capture for prefill chunk shapes and/or eliminating
  per-chunk W materialization, NOT a GEMM engine swap (max ~1.1x from GEMMs).
- Upstream FlyDSL repo has NO rdna3_f16_gemm_autotune.py despite the docstring
  referencing it; tile selection must be hand-swept.
