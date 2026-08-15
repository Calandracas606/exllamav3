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

