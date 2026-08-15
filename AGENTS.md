# exllamav3 ROCm/Triton work notes

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
