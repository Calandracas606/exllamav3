# PRD: ROCm gfx1100 Support for exllamav3 (v2 — PyTorch-Native Strategy)

## Revision History

- **v1** (original): Polyfill-and-hipify strategy. Compile all CUDA files for ROCm, add
  `compat.cuh` polyfill layer for PTX assembly, route GEMM to reconstruct fallback.
  Implemented as PR #1, received extensive review feedback.
- **v2** (this document): PyTorch-native strategy. Don't compile unused CUDA kernels at all.
  Replace kernels with PyTorch native ops where possible. Triton as second choice. Only keep
  C++ kernels that compile with minimal/no patching. Minimize delta from master.

## Introduction

exllamav3 is a high-performance inference engine for LLMs, currently CUDA-only. This PRD
defines a revised approach to adding **functional** (not performance-equivalent) ROCm
support targeting **gfx1100** (AMD Radeon RX 7900 XTX, RDNA3).

The v1 strategy (polyfill-and-hipify) proved problematic: the CUDA code is highly tuned for
NVIDIA hardware, and forcing it through compatibility layers produced fragile, hard-to-review
changes across 40+ files. The v2 strategy takes the opposite approach:

**Don't fight the CUDA code — route around it.**

Instead of compiling every CUDA kernel for ROCm and patching the ones that break, we:
1. **Exclude** CUDA files that can't compile cleanly (PTX tensor-core kernels, multi-GPU
   parallel, quantization comp_units) from the ROCm build entirely.
2. **Replace** runtime-critical kernels with PyTorch native operations where possible.
3. **Use Triton** kernels as a fallback when PyTorch native is insufficient.
4. **Keep** C++ kernels only when they compile cleanly under HIP with minimal `#ifdef` guards
   (no polyfills).

The guiding principle is **minimal delta from master**: changes to shared code should be
small `#ifdef` guards, not invasive polyfill layers. Python-level routing (`if
torch.version.hip`) is preferred over C-level patching.

## Goals

- `turboderp/Qwen3.5-9B-exl3` (4.00 bpw) loads and generates **coherent text** on gfx1100
- All currently-passing tests continue to pass (162 tests)
- Minimal diff between `rocm-gfx1100` branch and `master` — no large polyfill files
- No CUDA build regressions (all ROCm logic behind `#ifdef` / `torch.version.hip` guards)
- PyTorch native ops preferred over C++ kernels for ROCm; Triton as second choice
- Unused CUDA code is excluded from the ROCm build, not patched to compile
- Strict TDD: every change begins with a failing test

## Non-Goals (Out of Scope)

- Performance optimization or tuning (functional correctness only)
- Porting EXL3 tensor-core GEMM to WMMA/MFMA (future work)
- Vision/multimodal inference paths
- Support for architectures other than gfx1100
- Conversion/quantization tooling (inference only)
- Speculative decoding, multi-GPU/parallel inference paths
- Any changes to CUDA code paths beyond minimal `#ifdef` guards

## Architecture Context

### Target Model: Qwen3.5-9B-exl3 @ 4.00 bpw

A **hybrid architecture** (32 layers):
- 24× `linear_attention` layers (Mamba2/GatedDeltaNet-style state-space models)
- 8× `full_attention` layers (standard causal self-attention with GQA)
- EXL3-quantized linear layers throughout
- head_dim=256, 16 Q heads, 4 KV heads, hidden_size=4096, intermediate_size=12288
- vocab_size=248320, max_position_embeddings=262144

### Strategy: Kernel Classification

Every C++ kernel function is classified into one of four categories. The classification
determines the ROCm strategy:

| Category | Strategy | Examples |
|---|---|---|
| **A: PyTorch Native** | Replace with equivalent PyTorch op at Python level | RMS norm, activations, sampling, routing, rope |
| **B: Keep C++ (minimal guard)** | Compiles cleanly under HIP; keep with `#ifdef` if needed | hgemm (hipBLAS), reconstruct, had_r_128, stloader |
| **C: Exclude from build** | Not needed at runtime; don't compile for ROCm | exl3_gemm, exl3_gemv, all comp_units, parallel/, blocksparse |
| **D: Triton replacement** | PyTorch native insufficient, write Triton kernel | (Only if needed — assess case by case) |

### Detailed Kernel Classification

#### Category A — Replace with PyTorch Native (Python-level routing)

These kernels are performance optimizations over simple PyTorch ops. For a
functional-only port, PyTorch native equivalents are sufficient.

| Kernel | PyTorch Equivalent |
|---|---|
| `rms_norm` / `rms_norm_res_in` | `x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)` |
| `gated_rms_norm` | Same RMS norm + gating mask |
| `silu_mul` | `torch.nn.functional.silu(x) * gate` |
| `silu_oai_mul` | `torch.nn.functional.silu(x) * gate` (variant) |
| `gelu_mul` | `torch.nn.functional.gelu(x) * gate` |
| `relu_mul` / `relu2_mul` | `torch.nn.functional.relu(x) * gate` |
| `xielu` | PyTorch expr (piecewise) |
| `softcap` | `torch.tanh(x / cap) * cap` |
| `add` | `torch.add` |
| `add_sigmoid_gate` / `mul_sigmoid_*` | `torch.sigmoid` + elementwise |
| `deinterleave_qg` | `torch.reshape` + `torch.index_select` |
| `rope` / `gen_mrope_pos_ids` | Precompute cos/sin tables, apply via complex multiply |
| `routing_*` | `torch.softmax` + `torch.topk` + gather |
| `argmax_sample` | `torch.argmax` |
| `gumbel_sample` | `torch.multinomial` or Gumbel-max trick |
| `gumbel_noise_*` | `torch.distributions.gumbel.Gumbel` or `(-log(-log(uniform)))` |
| `apply_rep_pens` / `apply_pres_freq_pens` | Elementwise tensor ops |
| `cache_rotate` | `torch.roll` + masking |
| `count_inf_nan` | `torch.isinf(x).sum() + torch.isnan(x).sum()` |
| `histogram` | `torch.histc` |

**Routing mechanism:** In Python, check `torch.version.hip is not None` and call the PyTorch
native path instead of `ext.*`. This is a one-line `if` guard in each calling module. No
changes to C++ files for these kernels.

#### Category B — Keep C++ (compiles under HIP with minimal/no changes)

These kernels either use hipBLAS (which PyTorch auto-hipifies) or perform operations
that can't easily be expressed in pure PyTorch (EXL3-specific binary format manipulation).

| Kernel | Why Keep | Expected Changes |
|---|---|---|
| `hgemm` | Wraps hipBLAS `rocblas_hgemm` | None (auto-hipifies) |
| `reconstruct` / `reconstruct_slice` | EXL3 trellis dequant — custom binary format | None (see § Quantization Deep Dive) |
| `had_r_128` | EXL3 Hadamard transform — warp-cooperative | None (shuffles already wrapped) |
| `had_paley` / `had_paley2` | Hadamard matrix construction | Verify auto-hipifies |
| `stloader_*` | Weight loading (GPU copy) | Minimal `#ifdef` for CUDA API calls |
| `pack_trellis` / `unpack_trellis` | EXL3 binary packing (needed for model load) | Verify auto-hipifies |
| `quant_cache_*` / `dequant_cache_*` | KV cache quantization | Verify auto-hipifies or replace with PyTorch |
| `paged_kv_cache_update` | KV cache update | Verify auto-hipifies or replace with PyTorch |

**Decision rule:** If a Category B kernel requires more than 3-5 lines of `#ifdef` changes,
reclassify to Category A (PyTorch native) or Category D (Triton).

#### Category C — Exclude from ROCm Build Entirely

These files are not needed for single-GPU functional inference. Exclude them from the
source list when building for ROCm.

| File(s) | Why Exclude |
|---|---|
| `quant/exl3_gemm.cu` | PTX tensor-core GEMM — replaced by reconstruct + torch.matmul |
| `quant/exl3_gemv.cu` | PTX tensor-core GEMV — replaced by reconstruct + torch.matmul |
| `quant/exl3_gemv_int8.cu` | PTX int8 GEMV — not needed |
| `quant/exl3_moe.cu` | PTX MoE batched GEMM — not needed for non-MoE model |
| `quant/coop_autotune.cu` | Cooperative kernel autotuning — PTX only |
| `quant/comp_units/*.cu` | All compilation units for tensor-core kernels (50+ files) |
| `quant/exl3_kernel_map.cu` | Maps shapes to tensor-core kernels |
| `quant/quantize.cu` | Quantization tooling (not needed for inference) |
| `parallel/*.cu` | Multi-GPU communication (single GPU only for now) |
| `libtorch/gated_delta_net.cpp` | C++ linear attention — replaced by `fla` library |
| `libtorch/blocksparse_mlp.cpp` | Blocksparse MoE — not needed for Qwen3.5 |
| `cuda_drv.cpp` | NVIDIA driver API — not available on ROCm |
| `sam.cpp` | Speculative attention manager — not needed |

**Build system change:** `setup.py` / `ext.py` must maintain a separate source list for
ROCm builds that excludes these files. The CUDA source list stays unchanged (uses `os.walk`).

#### Category D — Triton Replacement (if needed)

Only used when PyTorch native is insufficient AND the C++ kernel can't compile cleanly.
Assess on a case-by-case basis during implementation. Likely candidates:

| Kernel | Why Triton Might Be Needed |
|---|---|
| `bighead_attn` | If PyTorch SDPA fallback doesn't handle the model's attention config |
| `gdn_ba_gemv` | GatedDeltaNet batched gemv — if fla library doesn't cover it |

**Preference order is always:** PyTorch native → keep C++ (minimal guard) → Triton → polyfill.

### Quantization Deep Dive

The EXL3 quantization format is the core innovation of exllamav3. These kernels operate on
a custom binary trellis format and cannot be replaced by standard PyTorch ops. They are the
most complex part of the port and require the most careful analysis.

**Key finding:** The quantization kernels needed for inference have a **limited and
manageable** set of PTX dependencies. They do NOT use tensor-core operations (`mma.sync`,
`cp.async`, `ldmatrix`). Their CUDA-specific dependencies are limited to bitfield/shuffle
operations that already have clean C++ equivalents. This makes them viable Category B
candidates — keep as C++ with minimal `#ifdef` guards.

#### Quantization Kernel Dependency Map

```
reconstruct.cu
  ├── exl3_dq.cuh (dequant dispatch)
  │     ├── codebook.cuh (codebook decode)
  │     │     ├── __dp4a          — HIP native ✓
  │     │     ├── asm("lop3.b32") — C++ XOR replacement already exists ✓
  │     │     └── mul_const_u32   — asm only on sm_86, plain multiply elsewhere ✓
  │     ├── __funnelshift_r       — HIP native ✓
  │     ├── FSHF_IMM (ptx.cuh)    — C++ __funnelshift_r equivalent ✓
  │     ├── BFE16_IMM (ptx.cuh)   — C++ shift+mask equivalent ✓
  │     ├── bfe64 (ptx.cuh)       — C++ uint64 shift equivalent ✓
  │     └── fshift                — pure C++ (uint64 merge) ✓
  ├── exl_shfl_down (util.cuh)    — wrapper with 32/64-bit mask handling ✓
  └── util.cuh / util.h           — standard helpers ✓

hadamard.cu / hadamard_inner.cuh
  ├── exl_shfl_xor (util.cuh)     — wrapper with 32/64-bit mask handling ✓
  ├── exl_shfl_down (util.cuh)    — wrapper with 32/64-bit mask handling ✓
  └── (no PTX assembly)           — pure warp-level arithmetic ✓

pack.cu / pack_trellis
  └── (bitfield packing ops)      — verify, likely plain C++ ✓

hgemm.cu
  └── rocblas_hgemm               — hipBLAS, auto-hipified ✓
```

#### What This Means

All `✓` items above either are HIP-native builtins or already have clean C++ replacements.
The quantization kernels should compile under HIP with **zero polyfills** — only the
existing `exl_shfl_*` wrappers in `util.cuh` (which correctly handle the 32-bit vs 64-bit
mask difference) and the existing bitfield macros in `ptx.cuh` (which have `#if
defined(USE_ROCM)` C++ branches).

**Files to verify compile cleanly (investigate during US-005):**
1. `reconstruct.cu` → `exl3_dq.cuh` → `codebook.cuh`
2. `hadamard.cu` → `hadamard_inner.cuh`
3. `hgemm.cu`
4. `pack.cu`
5. `hadamard.cpp` (Paley matrix construction)

If any of these fail to compile, the fix should be a targeted `#ifdef USE_ROCM` guard, not
a polyfill layer. If the fix is too complex, reclassify to Category A or D.

#### Risk Assessment

The main risk is that some of these kernels use CUDA intrinsics I haven't identified yet
(e.g., warp-level intrinsics in hadamard_inner.cuh that I haven't fully traced). The
investigation step (US-005) is designed to surface these early. If a kernel proves too
complex to port with minimal guards, it becomes a Category D (Triton) candidate.

### Python-Level Fallback Architecture

Instead of modifying C++ code, most routing happens in Python:

```python
# Example: in a module's forward method
if torch.version.hip is not None:
    # ROCm: use PyTorch native
    return torch.nn.functional.silu(x) * gate
else:
    # CUDA: use optimized C++ kernel
    ext.silu_mul(x, gate, x)
```

This approach:
- Zero changes to C++ files for Category A kernels
- Easy to review (one `if` statement per kernel)
- No polyfills, no compat layers
- Trivially correct (PyTorch ops are well-tested)

### Build System Changes

The key build system change is **source list filtering** for ROCm:

```python
# setup.py / ext.py
if torch.version.hip:
    # Exclude Category C files from compilation
    excluded_dirs = ['parallel/', 'quant/comp_units/', ...]
    excluded_files = ['exl3_gemm.cu', 'exl3_gemv.cu', ...]
    sources = [s for s in all_sources if not should_exclude(s)]
```

Plus:
- `-DUSE_ROCM` flag
- `-Wno-register` flag (C++20 deprecation)
- `PYTORCH_ROCM_ARCH=gfx1100` (via `arch_list.py`, using `rocminfo`)
- Guard nvcc-only flags (`-Xcudafe`, `--diag_suppress`)

### What Gets Deleted from the v1 PR

The v1 PR introduced extensive polyfill infrastructure that should be **removed** under the
v2 strategy:

| v1 Artifact | v2 Action |
|---|---|
| `compat.cuh` (polyfill layer) | **Delete or gut** — only keep genuinely needed minimal helpers |
| All `*_hip.cpp` / `*_hip.cuh` files | **Delete** — these are hipified copies that duplicate the CUDA files |
| `bindings_hip.cpp` | **Delete** — use single `bindings.cpp` with conditional compilation |
| PTX polyfills in `ptx.cuh` | **Revert** — files that include `ptx.cuh` for tensor-core ops are excluded from ROCm build. Keep only the bitfield C++ equivalents that the quantization kernels need. |
| `__grid_constant__` removals | **Revert** — keep CUDA code untouched |
| `__shfl_xor_sync` blanket mask changes across 40 files | **Revert** — not needed if those kernels are excluded. The `exl_shfl_*` wrappers in util.cuh handle this correctly for the kernels we keep. |
| `WARP_FULL_MASK` additions | **Revert** — not needed |

The v2 diff from master should be dramatically smaller than v1.

---

## Technical Constraints

- **uv-only dependency management**: All Python execution via `uv run`. uv-managed Python.
- **CUDA paths must not break**: ROCm logic behind `#ifdef` and `torch.version.hip` guards
  only. No changes to CUDA code paths beyond minimal `#ifdef` where a kernel is kept.
- **Strict TDD**: Every change must begin with a failing test.
- **Coherent output is mandatory**: The model must produce readable, sensible text. This is
  verified at every stage, not just at the end.
- **No polyfills**: If a C++ kernel needs a polyfill to compile under HIP, either replace it
  with PyTorch native or exclude it from the build. Do not write compatibility layers.

## Definition of Done (NON-NEGOTIABLE)

**This task can ONLY be considered finished or complete when the following is true:**

> The model `turboderp/Qwen3.5-9B-exl3` at 4.00 BPW
> (https://huggingface.co/turboderp/Qwen3.5-9B-exl3/tree/4.00bpw) **loads and generates
> coherent output** on the target 24GB AMD Radeon RX 7900 XTX (gfx1100).

Additionally:
- All 162 currently-passing tests continue to pass
- The diff between `rocm-gfx1100` and `master` is minimal (no large polyfill files)
- No CUDA build regressions

"Coherent output" means the model produces sensible, readable text responses to prompts —
not garbage tokens, not crashes, not silent corruption. Speed does not matter.

---

## User Stories

### US-001: Clean v1 polyfill artifacts from the branch

**Description:** As a developer, I need to start from a clean base by removing the v1
polyfill infrastructure (compat.cuh, _hip.cpp files, PTX stubs) so the v2 work begins from
a minimal delta against master.

**Acceptance Criteria:**
- [ ] `compat.cuh` polyfill layer removed or gutted (only keep genuinely needed minimal helpers)
- [ ] All `*_hip.cpp` / `*_hip.cuh` duplicate files removed
- [ ] `bindings_hip.cpp` removed — single `bindings.cpp` with `#ifdef USE_ROCM` guards
- [ ] PTX polyfills in `ptx.cuh` reverted to master state (keep only bitfield C++ equivalents
      that quantization kernels need)
- [ ] `__grid_constant__`, `__shfl_xor_sync` mask, `WARP_FULL_MASK` changes reverted
- [ ] CUDA build still compiles and all tests pass after cleanup
- [ ] TDD: verify tests pass before and after cleanup

### US-002: Build system excludes unused files for ROCm

**Description:** As a developer, I need `setup.py` / `ext.py` to maintain a filtered source
list for ROCm that excludes Category C files (tensor-core kernels, comp_units, parallel,
quantization, etc.) so they are never compiled under HIP.

**Acceptance Criteria:**
- [ ] `setup.py` and `ext.py` detect `torch.version.hip` and use a filtered source list
- [ ] Excluded files: all `quant/comp_units/`, `quant/exl3_gemm.cu`, `quant/exl3_gemv*.cu`,
      `quant/exl3_moe.cu`, `quant/coop_autotune.cu`, `quant/quantize.cu`,
      `parallel/*.cu`, `cuda_drv.cpp`, `libtorch/gated_delta_net.cpp`,
      `libtorch/blocksparse_mlp.cpp`, `sam.cpp`
- [ ] `-DUSE_ROCM` and `-Wno-register` flags passed for ROCm builds
- [ ] nvcc-only flags guarded
- [ ] CUDA build source list unchanged (still `os.walk`)
- [ ] `arch_list.py` uses `rocminfo` for arch detection (already implemented)
- [ ] TDD: failing test verifying ROCm source list excludes Category C files

### US-003: Extension compiles and loads on gfx1100

**Description:** As a developer, I need the filtered extension to compile under HIP and
load successfully, exposing the reduced set of Python bindings.

**Acceptance Criteria:**
- [ ] `uv run python -c "from exllamav3.ext import exllamav3_ext as ext; print(dir(ext))"`
      succeeds
- [ ] All Category B kernels (hgemm, reconstruct, had_r_128, stloader, etc.) are present
- [ ] Category C kernels (exl3_gemm, exl3_gemv, etc.) may or may not be present (stubs if needed)
- [ ] Category A kernels (rms_norm, activations, etc.) may be present if they compile cleanly,
      or absent if excluded (PyTorch native is used instead)
- [ ] A simple kernel call (e.g., `ext.hgemm`) works and produces correct output
- [ ] TDD: test that imports the extension and calls a basic function

### US-004: Core elementwise ops replaced with PyTorch native

**Description:** As a developer, I need the Python modules to route to PyTorch native
operations for elementwise kernels (norms, activations, softcap, etc.) on ROCm.

**Acceptance Criteria:**
- [ ] RMS norm: PyTorch equivalent matches C++ kernel output within tolerance
- [ ] Activations (silu_mul, gelu_mul, relu_mul): PyTorch equivalents match
- [ ] Softcap: PyTorch equivalent matches
- [ ] RoPE: PyTorch equivalent matches
- [ ] Routing/softmax: PyTorch equivalent matches
- [ ] Each replacement verified via TDD: write test comparing C++ output (on CUDA or cached
      reference) vs PyTorch native, observe ROCm path fails, implement, verify pass
- [ ] No changes to C++ kernel files — routing is purely in Python

### US-005: Quantization kernels compile and produce correct output

**Description:** As a developer, I need the EXL3 quantization kernels (reconstruct,
had_r_128, hgemm, pack_trellis) to compile under HIP with minimal `#ifdef` guards and
produce numerically correct output. These are the most complex kernels in the port and
operate on the custom EXL3 trellis binary format.

**Acceptance Criteria:**
- [ ] `reconstruct.cu` compiles under HIP (via `exl3_dq.cuh` → `codebook.cuh` dependency chain)
- [ ] `hadamard.cu` compiles under HIP (via `hadamard_inner.cuh`)
- [ ] `hgemm.cu` compiles under HIP (hipBLAS auto-hipification)
- [ ] `pack.cu` compiles under HIP
- [ ] `hadamard.cpp` (Paley matrix) compiles under HIP
- [ ] Any PTX assembly in these files is guarded with `#ifdef USE_ROCM` / C++ equivalent
      (e.g., `lop3.b32` → XOR, `FSHF_IMM` → `__funnelshift_r`)
- [ ] `ext.reconstruct` produces correct dequantized weights vs reference
- [ ] `ext.had_r_128` produces correct Hadamard transform vs reference
- [ ] `ext.hgemm` produces correct matmul vs `torch.matmul`
- [ ] No polyfill layer (`compat.cuh`) needed for these kernels — only targeted `#ifdef`s
- [ ] TDD: write correctness tests comparing against PyTorch references, observe failure,
      fix incrementally until correct

### US-006: EXL3 quantized linear layers work via reconstruct + torch.matmul

**Description:** As a user, I need EXL3-quantized linear layers to dequantize weights
(`ext.reconstruct`) and multiply (`torch.matmul` or `ext.hgemm`) on ROCm.

**Acceptance Criteria:**
- [ ] `LinearEXL3.forward()` routes to `reconstruct_hgemm` when `torch.version.hip` is set
- [ ] `ext.reconstruct` / `ext.reconstruct_slice` produce correct dequantized weights
- [ ] `ext.had_r_128` produces correct Hadamard transform output
- [ ] `ext.hgemm` (or `torch.matmul`) produces correct matmul output
- [ ] Test: create a small EXL3 layer, forward random input, compare against reference
- [ ] TDD: write test first, observe failure, implement

### US-007: Full attention dispatches to PyTorch SDPA on ROCm

**Description:** As a user, I need full-attention layers to use PyTorch's
`scaled_dot_product_attention` on ROCm.

**Acceptance Criteria:**
- [ ] Attention module routes to `fn_torch_sdpa_fallback_*` on ROCm
- [ ] SDPA produces correct output for the model's attention configuration (GQA, head_dim=256)
- [ ] No OOM or shape errors during attention computation
- [ ] Test: verify attention output matches a reference
- [ ] TDD: write test first, observe failure, implement

### US-008: Linear attention dispatches to fla library on ROCm

**Description:** As a user, I need Mamba2/GatedDeltaNet layers to route to the
`flash-linear-attention` library on ROCm.

**Acceptance Criteria:**
- [ ] Mamba2 layer falls back to `fla.ops.simple_gla.chunk_simple_gla`
- [ ] GatedDeltaNet layer falls back to `fla` or pure-torch reference
- [ ] C++ linear attention files (`libtorch/gated_delta_net.cpp`) excluded from ROCm build
- [ ] Output matches reference within tolerance
- [ ] Test: create a small linear-attention layer, forward input, compare against reference
- [ ] TDD: write test first, observe failure, implement

### US-009: Sampling works via PyTorch native on ROCm

**Description:** As a user, I need token sampling (argmax, Gumbel, repetition penalty, etc.)
to work via PyTorch native operations on ROCm.

**Acceptance Criteria:**
- [ ] Argmax sampling works via `torch.argmax`
- [ ] Gumbel sampling works via PyTorch Gumbel noise + argmax
- [ ] Repetition/presence/frequency penalties work via elementwise tensor ops
- [ ] Output matches C++ kernel output within tolerance (or is functionally equivalent)
- [ ] Test: verify sampling produces expected token distributions
- [ ] TDD: write test first, observe failure, implement

### US-010: End-to-end model load and generation on gfx1100

**Description:** As a user, I want to load `turboderp/Qwen3.5-9B-exl3` at 4.00 bpw and
generate coherent text on my Radeon RX 7900 XTX.

**Acceptance Criteria:**
- [ ] Model loads successfully with no errors
- [ ] Tokenizer works correctly
- [ ] Generation produces coherent text (not garbage tokens) for multiple test prompts
- [ ] No crashes or GPU errors during multi-token generation
- [ ] All 162 existing tests still pass
- [ ] TDD: end-to-end generation test written first, observed failing, implemented until passing

### US-011: Verify minimal delta from master

**Description:** As a maintainer, I need the final diff between `rocm-gfx1100` and `master`
to be small and reviewable — no large polyfill files, no duplicated _hip.cpp files.

**Acceptance Criteria:**
- [ ] No `compat.cuh` polyfill layer (or extremely minimal)
- [ ] No `*_hip.cpp` / `*_hip.cuh` duplicate files
- [ ] No PTX polyfills (only targeted `#ifdef` C++ equivalents for bitfield ops)
- [ ] Changes to shared C++ files are limited to small `#ifdef USE_ROCM` guards
- [ ] Python changes are `if torch.version.hip:` routing guards
- [ ] Build system change is source list filtering + flags
- [ ] `git diff master...rocm-gfx1100 --stat` shows a small, comprehensible diff

---

## Functional Requirements

- **FR-1:** The build system (`setup.py` and `ext.py`) must detect `torch.version.hip` and:
  (a) define `-DUSE_ROCM`, (b) set ROCm compiler flags, (c) use a filtered source list that
  excludes Category C files, (d) guard nvcc-only flags.
- **FR-2:** `arch_list.py` must detect gfx architecture via `rocminfo` (already implemented).
- **FR-3:** Category A kernels (norms, activations, sampling, routing, rope) must be replaced
  by PyTorch native operations via `torch.version.hip` routing in Python modules. No changes
  to C++ files for these kernels.
- **FR-4:** Category B kernels (hgemm, reconstruct, had_r_128, stloader) must compile under
  HIP with minimal `#ifdef` guards only. No polyfills.
- **FR-5:** Category C files must be excluded from the ROCm source list entirely. They are
  never compiled under HIP.
- **FR-6:** `LinearEXL3.forward()` must route to `reconstruct_hgemm` on ROCm.
- **FR-7:** Mamba2 and GatedDeltaNet must route to `fla` library fallbacks on ROCm.
- **FR-8:** Attention must dispatch to PyTorch SDPA on ROCm.
- **FR-9:** All v1 polyfill artifacts (compat.cuh, _hip.cpp files, PTX stubs) must be removed.
- **FR-10:** All Python execution must use `uv run`.
- **FR-11:** All development must follow TDD: failing test observed before implementation.
- **FR-12:** The diff from master must be minimal — no large generated/polyfill files.

## Technical Considerations

### Build System: Source List Filtering

`setup.py` currently uses `os.walk` to collect all `.c`/`.cpp`/`.cu` files. For ROCm, we
need to filter this list:

```python
ROCM_EXCLUDE_DIRS = ['parallel/', 'quant/comp_units/']
ROCM_EXCLUDE_FILES = {
    'quant/exl3_gemm.cu', 'quant/exl3_gemv.cu', 'quant/exl3_gemv_int8.cu',
    'quant/exl3_moe.cu', 'quant/coop_autotune.cu', 'quant/quantize.cu',
    'cuda_drv.cpp', 'sam.cpp',
    # libtorch C++ linear attention (replaced by fla)
    'libtorch/gated_delta_net.cpp', 'libtorch/blocksparse_mlp.cpp',
}
```

The `bindings.cpp` file may need `#ifdef USE_ROCM` guards to avoid referencing excluded
functions, or we maintain a separate binding registration that only includes available
functions.

### PyTorch Native Equivalents — Implementation Notes

**RMS Norm:**
```python
def rms_norm_torch(x, weight, eps):
    variance = x.float().pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight * x).to(x.dtype)
```

**RoPE:** Precompute frequency tables, apply via complex multiply or direct sin/cos rotation.
The exact variant (standard, mrope, etc.) must match the model config.

**Sampling:** Most sampling is already partially in Python. The C++ kernels are fused
optimizations. For functional correctness, the unfused PyTorch path works.

### Dependencies

| Package | CUDA-specific? | ROCm-compatible? |
|---|---|---|
| torch | Yes | ROCm wheels via TheRock index |
| flash-linear-attention | Partial | fla-org/flash-linear-attention (ROCm fixes) |
| kbnf, formatron | No (Rust/CPU) | Yes |
| tokenizers, safetensors | No (Rust) | Yes |
| ninja | No | Yes |
| triton | Partial | Works on ROCm (for Category D if needed) |

### Environment

- ROCm 7.14, TheRock pip index, clang 22
- gfx1100 detected via `rocminfo` in `arch_list.py`
- uv-managed Python (system Python has ROCm issues)

---

## Success Metrics

- Model loads and generates coherent text (primary)
- All 162 existing tests pass (non-negotiable)
- Diff from master is small and reviewable (target: <500 lines excluding deletions)
- No polyfill files (compat.cuh, _hip.cpp) in the final diff
- Extension compiles in under 10 minutes (fewer files to compile)

## Open Questions

1. **Which Category B kernels actually compile cleanly under HIP?** Need to test each one
   after excluding Category C files. Reclassify to A or D as needed. The quantization kernels
   (reconstruct, had_r_128) are the highest-risk items — their PTX dependencies look clean
   on paper but must be verified empirically.
2. **Does PyTorch SDPA handle head_dim=256?** Standard SDPA may have limits on head
   dimension. If it fails, a Triton attention kernel may be needed (Category D).
3. **Does `fla`'s `chunk_simple_gla` handle Qwen3.5's exact Mamba2/GDN config?** Verify
   num_heads, head_dim, conv1d dimensions.
4. **bindings.cpp approach:** Should we use `#ifdef USE_ROCM` guards on individual `m.def()`
   lines, or maintain a separate ROCm bindings file? The former is simpler and has less delta.
5. **stloader:** Does the GPU-deferred weight loading path (`stloader_deferred_cuda`) work
   on ROCm? If not, can we fall back to CPU loading (`stloader_deferred_cpu`)?
6. **Memory budget:** Qwen3.5-9B at 4bpw is ~7 GB on disk. Reconstruct path allocates fp16
   weights (~18 GB) temporarily per layer. With 24 GB VRAM, this should fit but needs
   verification during US-010.