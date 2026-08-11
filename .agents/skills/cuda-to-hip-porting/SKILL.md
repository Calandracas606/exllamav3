---
name: cuda-to-hip-porting
description: Patterns and anti-patterns for porting CUDA code to HIP/ROCm, learned from PR #1 code review. Apply when adding ROCm support, writing #ifdef USE_ROCM branches, or translating CUDA builtins to HIP.
---

# CUDA-to-HIP Porting Guidelines

Distilled from the PR #1 (ROCm gfx1100 support) code review. Each rule below
originates from a concrete bug or design flaw caught in review.

## 1. Never Remove CUDA Builtins — Macro or Polyfill Them

**The iron law: never break upstream CUDA.** When a CUDA-specific attribute or
builtin (`__grid_constant__`, `__shfl_*_sync`, etc.) needs a HIP equivalent, do
NOT delete it from the CUDA path. Instead, define a portable abstraction that
preserves the CUDA semantics exactly.

**Wrong:** Removing `__grid_constant__` from a function parameter so it compiles
on HIP. This degrades CUDA codegen and silently breaks upstream.

```cpp
// WRONG: CUDA builtin removed entirely
const Offsets all_offsets,
```

**Right:** Define a macro that is a no-op on HIP but preserves the attribute on
CUDA:

```cpp
#ifdef USE_ROCM
  #define __grid_constant__
#endif

const Offsets __grid_constant__ all_offsets,
```

Or use `compat.cuh` polyfills guarded by `#ifdef USE_ROCM`.

## 2. Use Wrapper Functions Over Mechanical sed Replacements

**Never do blanket find-and-replace across dozens of files without understanding
semantics.** The review caught a mechanical `0xffffffff → 0xffffffffffffffffULL`
replacement across ~40 files — the riskiest change in the PR.

For warp shuffle operations, define a forced-inline wrapper instead of changing
every call site:

```cpp
// compat.cuh
__device__ __forceinline__ float shfl_xor_all(float var, int laneMask) {
#if defined(USE_ROCM)
    return __shfl_xor_sync(0xffffffffffffffffULL, var, laneMask);  // 64-bit mask required by HIP
#else
    return __shfl_xor_sync(0xffffffff, var, laneMask);  // 32-bit mask for CUDA
#endif
}
```

This centralizes the platform difference in one place. The mask is always the
same (full warp participation), so it doesn't need to be a parameter.

**Key insight:** HIP requires 64-bit unsigned masks for `__shfl_*_sync` even on
32-lane wavefronts (compile-time static_assert enforces this). The upper bits
are simply ignored on wave32 hardware.

## 3. Map CUDA API Semantics Correctly — Don't Guess

Every CUDA API function has a HIP equivalent, but the semantics may differ. Look
up the mapping in the [HIPIFY compatibility table](https://rocm.docs.amd.com/projects/HIPIFY/en/latest/reference/tables/CUDA_Runtime_API_functions_supported_by_HIP.html)
before writing polyfills.

### Timer functions

- **CUDA `%globaltimer`**: wall-clock nanoseconds.
- **HIP `clock64()`**: GPU shader cycles (~2.6 GHz on gfx1100). Per AMD docs,
  **does not work properly on RDNA3 (GFX11)**.
- **HIP `wall_clock64()`**: constant-frequency counter (queryable via
  `hipDeviceAttributeWallClockRate`). Best equivalent for monotonic timing.

If the function name says `_ns`, it must return nanoseconds or be documented
otherwise. Do not use `clock64()` for a `globaltimer_ns()` polyfill.

### Shared memory carveout

- `cudaSharedmemCarveoutMaxShared == 100` on CUDA.
- On HIP, `hipFuncAttributePreferredSharedMemoryCarveout` takes a percentage
  (0–100).

Use a named constant, not a bare literal:
```cpp
constexpr int kMaxSharedCarveout = 100;
```

### Atomic 128-bit operations

CUDA's `stg.wt`/`ldg.cv` are whole-128-bit operations. The HIP emulation using
`__atomic_store_n` on only `p->x` does NOT provide 128-bit atomicity — it only
provides release/acquire ordering on the first 32 bits. Document explicitly
that atomicity of the full payload must be provided by a separate flag/seqlock
at the call site.

## 4. Don't Polyfill What HIP Already Provides

Before writing a polyfill, check whether HIP has the function natively:

- HIP has `__dp4a` — don't reimplement it.
- HIP has `_rn` (round-to-nearest) variants of all intrinsics. If a CUDA function
  uses `_rz` (round-toward-zero), check whether `_rz` is semantically required or
  whether `_rn` would work. HIP lacks `_rz` for some types but has `_rn` for all.
- HIP's `__builtin_amdgcn_s_sleep(int)` provides a low-power NOP for spin-loop
  backoff. Don't use volatile busy-wait loops with arbitrary iteration counts.

## 5. Use rocminfo for Architecture Detection, Not Device Properties

**Never derive gfx arch from `torch.cuda.get_device_properties().major/minor`.**
These are CUDA-style compute capability numbers, not gfx ISA identifiers.

**Correct approach:** Parse `rocminfo` output for proper `gfxXXX` identifiers:

```python
result = subprocess.run(['rocminfo'], capture_output=True, text=True, timeout=5)
for line in result.stdout.splitlines():
    if line.strip().startswith('Name:'):
        name = line.split('Name:')[1].strip()
        if name.startswith('gfx') and name[3:].isdigit():
            arch_list.append(name)
```

Filter to `gfx` + digits only — `rocminfo` also lists ISA target triples like
`amdgcn-amd-amdhsa--gfx11-generic` which are NOT valid compile targets.

For unknown devices, raise a clear `RuntimeError` with instructions to set
`PYTORCH_ROCM_ARCH`. Never emit a malformed arch string.

## 6. Dead Code Must Still Be Correct — Use abort() Not Wrong Emulation

If a code path is unreachable on ROCm (e.g., tensor-core MMA stubs), use
`abort()` or `#error`, never a "plausible-looking but incorrect" emulation.

**Why:** "Dead + wrong" is worse than "dead + crash." If someone later removes
the short-circuit that makes the path unreachable, wrong emulation produces
silent data corruption. `abort()` fails loudly.

```cpp
// WRONG: FMA emulation of warp-cooperative MMA
frag_c.elems[0] = __hfma2(a[0], b, frag_c.elems[0]);
frag_c.elems[1] = __hfma2(a[0], b, frag_c.elems[0]);  // copy-paste bug!
```

```cpp
// RIGHT
#if defined(USE_ROCM)
// Warp-level MMA cannot be emulated with per-thread scalar code.
// All GEMM paths are routed through reconstruct_hgemm (hipBLAS) at runtime.
abort();
#endif
```

Also: check for copy-paste bugs in any emulation code. If `elems[1]` references
`a[0]/a[1]` when it should reference `a[2]/a[3]`, the output is garbage.

## 7. Warp Size: RDNA3 (gfx1100) is Wave32 Under HIP

RDNA3 (gfx10xx/gfx11xx) is natively wave32. `warpSize == 32` at runtime on
gfx1100 under the HIP runtime. GCN/CDNA (gfx9xx/MI-series) uses wave64.

- HIP still requires 64-bit mask *types* for `__shfl_*_sync` even on wave32
  (compile-time static assert). Upper bits are ignored.
- The hardcoded `32` / `16` / `warpSize / 2` reduction widths are correct for
  gfx1100.
- Using `warpSize` (runtime) instead of literal `32` is a harmless defensive
  refactor that works for both wave sizes.

## 8. Test Portability — No Machine-Specific Paths or Import-Time Side Effects

- **Never hardcode machine paths** in tests. Use env vars (`EXL_TEST_MODEL`)
  with `pytest.mark.skipif` guards when the resource is absent.
- **Never load models at import time.** Use `@pytest.fixture(scope="module")`
  so collection doesn't fail when hardware/models are missing.
- Tests must be able to be *collected* in CI even if they are *skipped* due to
  missing hardware.

## 9. Never Cite Non-Existent Constraints

Do not attribute design decisions to PRD requirements or documentation that
don't exist. If there is a real constraint (e.g., VRAM budget), state it with
the actual numbers. Reviewers will check.

**Wrong:** "Removed caching (violates PRD requirement 'Do NOT cache weights')"
when the PRD says no such thing.

**Right:** "Caching is not used because a reconstructed fp16 weight for a 9B
model is ~18 GB, exceeding the 24 GB VRAM budget of the target GPU. Long-term
fix is a native WMMA kernel." Document this in a TODO comment.

## 10. Development Artifacts Don't Ship in the Repo Root

Files like `HANDOFF.md`, `prd-*.md`, and other agent/session working docs contain
machine-specific paths, stale debug narrative, and potentially incorrect claims.
They must not ship in the repo root. Move them to `docs/`, a branch, or delete
before merge. Incorrect technical assertions in version-controlled docs will be
trusted by future contributors and agents.

## 11. compat.cuh Include Order

When adding `#include "compat.cuh"` to a header file, `#pragma once` must come
FIRST, before any includes:

```cpp
#pragma once                    // <-- guard first
#include "compat.cuh"           // <-- then includes
```

If the include precedes `#pragma once`, multiple TUs can include `compat.cuh`
multiple times, defeating the guard.
