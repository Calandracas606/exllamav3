# HANDOFF: ROCm gfx1100 Port of exllamav3

## Current Status: COMPLETE — Definition of Done MET

The extension compiles, loads, and produces **coherent inference output**.
End-to-end generation verified on Qwen3.5-9B-exl3 at 4.00bpw.

### Verified Inference Output (Session 4, post-RoPE-norm fix)
```
Prompt: "The capital of France is"
→ " Paris." (correct)

Prompt: "Write a Python function to check if a number is prime:"
→ Correct is_prime() implementation with trial division

Prompt: "The mitochondria is"
→ " the powerhouse of the cell..."

Prompt: "Explain quantum computing in simple terms:"
→ Coherent multi-paragraph explanation with analogies
```

### Test Results
- `test_rope.py`: 60/60 PASS (including ALL norm=True paths — previously broken)
- `test_gated_delta_rule.py`: 21/21 PASS (recurrent + chunk kernels)
- `test_rocm_qgemm.py`: 7/7 PASS (reconstruct_hgemm correctness)
- End-to-end generation: coherent output confirmed

## Branch
`rocm-gfx1100` — **pushed to remote** (`origin/rocm-gfx1100`)

## Environment
- ROCm 7.14 installed in venv via TheRock pip index (`torch 2.12.0+rocm7.14.0`)
- ROCm SDK at `.venv/lib/python3.11/site-packages/_rocm_sdk_devel/`
- GPU: Radeon RX 7900 XTX (gfx1100, RDNA3)
- Use `uv run --env-file .env python` for all commands (the `.env` file sets `ROCM_HOME`)
- Model: `turboderp/Qwen3.5-9B-exl3` at 4.00bpw, downloaded to HF cache at:
  `/home/openhands/.cache/huggingface/hub/models--turboderp--Qwen3.5-9B-exl3/snapshots/01192d8c6d9cbd94f9cf99c0ddb85e9e217ccc01`

### ⚠️ ROCM_HOME (critical build requirement)
When ROCm is installed in the venv (not system-wide at `/opt/rocm`), PyTorch's
`_find_rocm_home()` will find the wrong ROCm (e.g. a stale `/opt/rocm-7.2.4`).
This causes host C++ compilation to fail with `hip/hip_runtime.h: No such file`
because the host compiler doesn't get the venv ROCm include path.

**Fix**: The `.env` file in the repo root sets `ROCM_HOME` to the venv SDK path.
Always build with `uv run --env-file .env python ...`. The `.env` file is
gitignored (contains a machine-specific absolute path).

## What Works (Verified)
- Extension compiles with `uv run --env-file .env python setup.py develop`
  (or JIT-compiled on first import with `uv run --env-file .env python -c "from exllamav3.ext import exllamav3_ext"`)
- Extension loads and imports (`exllamav3_ext`)
- Core kernels produce correct output on ROCm:
  - `rms_norm`, `silu_mul` (activations)
  - `hgemm` (hipBLAS GEMM)
  - `had_r_128` (Hadamard transform)
  - `reconstruct`, `reconstruct_slice` (EXL3 dequantization)
  - `cuda_recurrent_mamba2` (SSM recurrence for decode)
  - `cuda_recurrent_gated_delta_rule` (delta rule for decode)
- `fla` library works on ROCm (Triton backend):
  - `fla.ops.simple_gla.chunk_simple_gla` (Mamba2 prefill)
  - `fla.ops.gated_delta_rule.chunk_gated_delta_rule` (GDN prefill)
- Model loads successfully (Qwen3.5-9B-exl3 at 4.00bpw)

## What's Done
### Build System
- `ext.py`, `setup.py`, `arch_list.py`: `USE_ROCM` guards and ROCm compiler flags
- `compat.cuh`: `__dp4a` emulation, `__nanosleep`, `__ldcs`/`__ldg` overloads, `hipFuncSetAttribute`, `cuda::atomic_ref`; bfloat16 guarded with `__HIPCC__`
- `ptx.cuh`: Full ROCm stub layer (MMA, cp_async, barrier, BFE/FSHF/PRMT, globaltimer, group_barrier)
- `util.cuh`: Conditional includes with `USE_ROCM` guards
- `.gitignore`: Excludes generated hip files

### Runtime Routing (3 files)
- **`exl3.py`** (`LinearEXL3`): `forward()` routes to `reconstruct_hgemm()` on ROCm. **No weight caching** — on-the-fly dequant + hgemm per call, as per PRD requirement "Do NOT cache dequantized weights".
- **`mamba2.py`** (`Mamba2`): `forward()` adds `not torch.version.hip` guard on fused BC decode path → falls back to `fla.chunk_simple_gla` (prefill) / `ext.cuda_recurrent_mamba2` (decode)
- **`gated_delta_net.py`** (`GatedDeltaNet`): Same guard on `self.bc_split` fused path → falls back to `fla.chunk_gated_delta_rule` (prefill) / `ext.cuda_recurrent_gated_delta_rule` (decode)
- **`conv1d.py`** (`causal_conv1d_update`): Added `torch.version.hip is None` guard on CUDA fast path → falls back to PyTorch reference

### Attention
- No code changes needed. `has_fa2=False`, `has_triton=True` on ROCm. The existing dispatch chain automatically tries Triton kernels, then falls through to `fn_torch_sdpa_fallback`.

## Inference — WORKING

Inference is verified working. The key was using the correct API:
```python
from exllamav3 import Config, Model, Cache, Tokenizer
from exllamav3.generator.generator import Generator
from exllamav3.generator.sampler import DefaultSampler

config = Config.from_directory(model_dir)
model = Model.from_config(config)
cache = Cache(model=model, max_num_tokens=8192)
model.load()
tokenizer = Tokenizer.from_config(config)
gen = Generator(model, cache, tokenizer)

output = gen.generate(
    'The capital of France is',
    max_new_tokens=8,
    sampler=DefaultSampler(),
    add_bos=True,
)
print(output)  # "The capital of France is Paris."
```

## Key Files Modified
| File | Change |
|------|--------|
| `exllamav3/exllamav3_ext/compat.cuh` | CUDA→HIP compat macros |
| `exllamav3/exllamav3_ext/ptx.cuh` | PTX intrinsics → ROCm stubs |
| `exllamav3/exllamav3_ext/util.cuh` | Include guards for ROCm |
| `exllamav3/exllamav3_ext/ext.py` | ROCm compile flags |
| `exllamav3/exllamav3_ext/setup.py` | ROCm build setup |
| `exllamav3/exllamav3_ext/arch_list.py` | gfx1100 arch target |
| `exllamav3/exllamav3/modules/quant/exl3.py` | ROCm routes to reconstruct_hgemm (no caching) |
| `exllamav3/exllamav3/modules/mamba2.py` | Skip fused BC on ROCm |
| `exllamav3/exllamav3/modules/gated_delta_net.py` | Skip fused BC_split on ROCm |
| `exllamav3/exllamav3/modules/gated_delta_net_fn/conv1d.py` | Disable CUDA conv1d fast path on ROCm |
| `exllamav3/exllamav3_ext/reduction.cuh` | warpSize replaces hardcoded 32 |
| `exllamav3/exllamav3_ext/quant/exl3_gemv_int8.cu` | Named carveout constant |
| `.gitignore` | Exclude generated hip files |

## Commits (on rocm-gfx1100 branch)
1. `ddbd1db` Set up uv-managed Python environment
2. `78a26da` ROCm gfx1100 functional port - extension compiles/loads
3. `705562e` Route EXL3/Mamba2/GDN to fallback paths
4. `61778dd` Cache reconstructed EXL3 weights (superseded by reconstruct_hgemm routing)
5. `a078718` Disable cooperative kernel paths on ROCm
6. `be3c47d` Address PR review comments (warp size, shfl masks, include guards, MMA stubs)
7. `6954426` Update HANDOFF.md
8. `d46f76e` Replace ALL hardcoded warp sync masks with WARP_FULL_MASK macro (fixed 15 files)

## Design Decisions
- **Tensor-core kernels are STUBBED, not ported**: All MMA/WMMA PTX intrinsics call `abort()`. This means the fused BC single-token decode paths and fused GEMM paths are unavailable. Fallback paths (reconstruct + hgemm, fla, recurrent kernels) are used instead.
- **No weight caching**: Per PRD requirement "Do NOT cache dequantized weights." Each forward call reconstructs on-the-fly via `reconstruct_hgemm()`. This trades speed for correctness and memory efficiency.
- **HIP requires 64-bit shfl masks**: Even on gfx1100 (32-lane wavefronts), HIP's `__shfl_*_Sync` requires 64-bit masks (`0xffffffffffffffffULL`) due to a static assertion in `amd_warp_sync_functions.h`. Using 32-bit `0xffffffff` fails to compile.
- **Wave32 is the default on gfx1100**: No compiler flag needed. `warpSize` is 32 at runtime. Previous session incorrectly added `-mno-wavefrontsize64` — removed as redundant.
- **`fla` library is the prefill workhorse**: The Triton-based `chunk_simple_gla` and `chunk_gated_delta_rule` functions handle prefill correctly on ROCm.

## Research Notes
- HIP API has equivalents for most CUDA intrinsics. Key polyfills were needed for:
  - `__dp4a` (dot-product accumulate) — no HIP equivalent, emulated with shifts
  - `cuda::atomic_ref` — available in HIP as `std::atomic_ref` with `__HIPCC__` guard
  - PTX bitfield ops (BFE/FSHF/PRMT) — HIP has `__hip_lcshift_*` and bitfield intrinsics
- Triton on ROCm works well — `fla` library and attention kernels compile and run correctly
- `fla` is the best available library for linear attention on ROCm. Alternatives considered:
  - Custom Triton kernels (would need significant effort to match fla's correctness)
  - PyTorch reference implementations (too slow for prefill)
  - `flash-linear-attention` is the most mature and already a dependency

## Session 4: Critical Bug Fix — RoPE Norm Race Condition

### Root Cause
`rope.cu`'s `apply_norm()` and `apply_norm_uw()` lambdas had a **shared memory write race condition**:
after `warp_reduce_sum_f()` reduces within a warp, only lane 0 holds the correct sum, but ALL threads
wrote the result to the same `sums[warps * t_head + warp_id]` slot. On NVIDIA GPUs, lane 0's write
wins this race (undefined behavior that happens to work). On AMD GPUs, a different lane's write wins,
producing incorrect normalization values.

This caused `test_rope.py` norm=True tests to fail with 90%+ mismatch.

### Fix (rope.cu, lines 210-213 and 260-262)
```c
// Before (BUG):
sums[warps * t_head + warp_id] = warp_reduce_sum_f(sum);

// After (FIX):
sum = warp_reduce_sum_f(sum);
if (lane_id == 0) sums[warps * t_head + warp_id] = sum;
```

### Also cleaned up in this session:
- Removed redundant `-mno-wavefrontsize64` flag from ext.py (wave32 is default on gfx1100)
- Clarified comments in compat.cuh and util.cuh about 64-bit mask requirement
- Fixed test device assignments (cuda:1/cuda:2 → cuda:0)

### Important: Do NOT edit .hip files
The `.hip`, `*_hip.cuh`, `*_hip.cpp` files are auto-generated by PyTorch's hipify at build time.
Always edit the `.cu`/`.cuh`/`.cpp` source files. The `.gitignore` correctly excludes generated files.

## Session 3 API Audit Results (Comprehensive)

### Polyfills verified against HIP 7.2.4 headers:
| Polyfill | Native HIP equivalent? | Verdict |
|----------|----------------------|---------|
| `rsqrtf` | Yes: `__ocml_rsqrt_f32` (device), math header | Polyfill is host-only, doesn't interfere |
| `__funnelshift_r` | Yes: `amd_device_functions.h:207` | Used directly, no polyfill needed |
| `__ldg`/`__ldcs` | Yes: for half/half2. Templates added for generic types | Correct |
| `hipFuncSetAttribute` | Yes: native HIP API | Wrapper just casts function ptr type |
| `__hmax2`/`__hmin2` (half2) | No: only bfloat162 supported | Polyfill via `__hmax`/`__hmin` loop is correct |
| `__dp4a` | `__ockl_sdot4` exists but needs type conversion | Current byte-extraction polyfill is correct for functional use |
| `WARP_FULL_MASK` | N/A (macro) | 64-bit on ROCm, 32-bit on CUDA — matches API requirements |

### Functions verified as never called at runtime on ROCm:
- `barrier_acquire`/`barrier_release` (cooperative paths disabled via `torch.version.hip` guards)
- `group_barrier` (same)
- `globaltimer_ns` (same; uses `clock64()` — `__builtin_amdgcn_s_memrealtime` requires `s-memrealtime` target feature not available by default on gfx1100)
- All PTX MMA stubs (`ptx_mma_m16n8k16`, `ptx_mma_m8n8k4`) — abort() is safe

### Runtime routing verified:
- All cooperative kernel paths (bc_attn, BC_Mamba2, BC_GatedDeltaNet, BC_LinearFP16) are disabled via `torch.version.hip is None` / `not torch.version.hip` guards in Python
- EXL3 GEMM routes to `reconstruct_hgemm` (on-the-fly dequant + hipBLAS) — no weight caching
- Linear attention (prefill) routes to `fla` library
- Linear attention (decode) uses `cuda_recurrent_mamba2` / `cuda_recurrent_gated_delta_rule` kernels
