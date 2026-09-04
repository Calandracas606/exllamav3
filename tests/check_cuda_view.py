#!/usr/bin/env python3
"""
CUDA-view no-op checker: every file shared with upstream, preprocessed for the CUDA
branch (USE_ROCM undefined), must be byte-identical to upstream/master, except for an
enumerated list of justified deltas. Run from the repo root:

    python tests/check_cuda_view.py [--base upstream/master] [--ref <worktree>]

The reference must be a clean checkout of the base commit; the script creates a
temporary git worktree for it when --ref is not given. tests/test_cuda_view.py wraps
this as a pytest test so the suite runs it.
"""

import argparse
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Files whose CUDA view is allowed to differ from upstream, with justifications.
EXPECTED_DELTAS = {
    "exllamav3/exllamav3_ext/norm.cu": "half-gate dispatch rows unconditional (cross-platform)",
    "exllamav3/exllamav3_ext/rope.cu": "warp-reduce race fix (cross-platform bugfix)",
    # compat.cuh: device helpers guarded to the HIP compiler passes so host .cpp TUs
    # (libtorch/attention.cpp) parse; CUDA builds compile them unchanged, plus the
    # host-parse __align__ macro and intrinsic declaration from CUDA's host_defines
    "exllamav3/exllamav3_ext/compat.cuh": "device helpers hidden from host parses; host-defines macros",
    # graph seam: see cuda_drv_compat.cuh
    "exllamav3/exllamav3_ext/graph.cuh": "HIP compat include for unmapped driver types",
    "exllamav3/exllamav3_ext/cuda_drv.h": "one include + comment line (compat header)",
    "exllamav3/exllamav3_ext/cuda_drv.cpp": "one dlopen candidate (libamdhip64 probe)",
    # exl3_gemm stack: the ptx tensor-core inner is CUDA-only; the kernel wrapper
    # selects the ROCm split-K streaming inner behind the identical template
    # signature (same pattern as ptx.cuh's ptx_rocm_compat branch), and the two
    # cudaFuncSetAttribute call sites pass the function pointer through the
    # (const void*) parameter type both vendors' APIs declare
    "exllamav3/exllamav3_ext/quant/exl3_gemm_kernel.cuh": "platform inner seam: ptx pipeline vs split-K streaming inner",
    "exllamav3/exllamav3_ext/quant/exl3_gemm.cu": "cudaFuncSetAttribute explicit (const void*) cast (valid on both)",
    "exllamav3/exllamav3_ext/quant/exl3_moe.cu": "cudaFuncSetAttribute explicit (const void*) cast (valid on both)",
    "exllamav3/exllamav3_ext/quant/quantize.cu": "cudaFuncSetAttribute explicit (const void*) cast (valid on both)",
    # q_cache.cu: hipify's launcher regex mis-parses 'fn_ptr[idx]<<<...>>>'; the
    # six sites hoist the fn pointer to a local first (plain C++, both vendors)
    "exllamav3/exllamav3_ext/cache/q_cache.cu": "kernel fn-pointer table launches hoisted to a local var (hipify-safe)",
    # lmq.cuh: lm_clamp_ gains __host__ __device__ (clang HIP's device pass does
    # not define __CUDA_ARCH__, so the host-only static inline was unreachable
    # from device code)
    "exllamav3/exllamav3_ext/cache/lmq.cuh": "lm_clamp_ __host__ __device__ (device pass sees it)",
    # exl3_moe_kernel.cuh: same platform-inner seam as exl3_gemm_kernel.cuh; the
    # scheduler atomics compile untouched via the cuda::atomic_ref shim
    "exllamav3/exllamav3_ext/quant/exl3_moe_kernel.cuh": "platform inner seam (see exl3_gemm_kernel.cuh)",
    # moe_handoff.cu: MemOps wraps both platforms' stream-wait ops behind one
    # write/wait interface (dlsym'd driver API on CUDA, direct runtime API on
    # HIP, which has a different arity); call sites are platform-neutral
    "exllamav3/exllamav3_ext/cpu/moe_handoff.cu": "MemOps::write/wait wrap cu-dlsym vs hip runtime stream ops",
    # cuda_fp16.hpp (operator-overload companion) is CUDA-only; tested with __has_include
    # so a future HIP-side equivalent is picked up automatically
    "exllamav3/exllamav3_ext/activation.cu": "cuda_fp16.hpp via __has_include (feature test, not platform test)",
    "exllamav3/exllamav3_ext/gdn.cu": "cuda_fp16.hpp via __has_include (feature test, not platform test)",
    # torch's JIT loader hipifies in place on ROCm and cannot rewrite kernel launches
    # on function-pointer expressions; the local copy is behavior-identical on CUDA
    "exllamav3/exllamav3_ext/quant/pack.cu": "function-pointer launch via local copy (hipify requirement)",
    "exllamav3/exllamav3_ext/quant/reconstruct.cu": "function-pointer launch via local copy (hipify requirement)",
    # ptx.cuh: self-gating include seam (one guard at the top of one file)
    "exllamav3/exllamav3_ext/ptx.cuh": "self-gating include seam",
    # util.cuh: rocBLAS/hipBLAS lack CUBLAS_STATUS_LICENSE_ERROR; guarded on macro
    # availability so the same code compiles against both headers
    "exllamav3/exllamav3_ext/util.cuh": "CUBLAS_STATUS_LICENSE_ERROR availability guard",
}

# Files whose preprocessed CUDA view should be checked (everything the branch touches
# under exllamav3_ext plus setup.py/ext.py are excluded: Python is not preprocessed).
PP_DIR = "exllamav3/exllamav3_ext"


def git(*args, cwd = ROOT):
    return subprocess.run(["git", *args], cwd = cwd, check = True,
                          capture_output = True, text = True).stdout


def strip_conditionals(text):
    """Remove lines inside USE_ROCM-only conditional blocks (the branch's platform
    guards) and the guard lines themselves, so the CUDA preprocessor view of the file
    can be compared as text. Conditionals on anything else pass through verbatim
    (identical on both sides), including nested USE_ROCM guards inside them."""
    out = []
    stack = []  # frames: dicts {drop, passthrough}
    for line in text.splitlines():
        s = line.strip()
        m = re.match(r"#\s*(if|ifdef|ifndef|else|elif|endif)\b(.*)", s)
        if m:
            directive, rest = m.group(1), m.group(2).strip()
            dropping = any(f["drop"] for f in stack)
            if directive in ("if", "ifdef", "ifndef"):
                cond = re.sub(r"\s+", "", rest)
                if directive == "ifdef": cond = f"defined({cond})"
                elif directive == "ifndef": cond = f"!defined({cond})"
                if cond in ("defined(USE_ROCM)", "!defined(USE_ROCM)"):
                    stack.append({"drop": cond == "defined(USE_ROCM)", "passthrough": False})
                else:
                    stack.append({"drop": dropping, "passthrough": True})
                    if not dropping: out.append(line)
                continue
            if directive == "else":
                if stack and not stack[-1]["passthrough"]:
                    stack[-1]["drop"] = not stack[-1]["drop"]
                else:
                    if not dropping: out.append(line)
                continue
            if directive == "elif":
                # The CUDA-view strip only models if/else on USE_ROCM guards; an elif
                # on or inside one would be mis-stripped, so fail loudly instead
                if any(not f["passthrough"] for f in stack):
                    raise SystemExit(
                        f"check_cuda_view: #elif inside a USE_ROCM-only guard is not "
                        f"supported by strip_conditionals: {line!r}")
                if not dropping: out.append(line)
                continue
            if directive == "endif":
                frame = stack.pop() if stack else None
                if frame is not None and frame["passthrough"] and not dropping:
                    out.append(line)
                continue
        if not any(f["drop"] for f in stack):
            out.append(line)
    stripped = "\n".join(out).rstrip("\n") + "\n"
    # a removed platform block can leave a doubled blank line; normalize blank runs
    return re.sub(r"\n{3,}", "\n\n", stripped)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default = "upstream/master")
    ap.add_argument("--ref", default = None, help = "worktree of the base commit; created if missing")
    args = ap.parse_args()

    ref = args.ref
    tmp = None
    if ref is None:
        tmp = tempfile.mkdtemp(prefix = "exl3-cudaview-")
        ref = os.path.join(tmp, "ref")
        subprocess.run(["git", "worktree", "add", "--detach", ref, args.base],
                       cwd = ROOT, check = True, capture_output = True)

    modified = [f for f in git("diff", "--name-only", args.base, "--", PP_DIR).splitlines() if f]
    failures = []
    checked = 0
    for f in modified:
        if not f.endswith((".cu", ".cuh", ".cpp", ".h", ".c")):
            continue
        ours = open(os.path.join(ROOT, f), encoding = "utf-8").read()
        ref_path = os.path.join(ref, f)
        if not os.path.exists(ref_path):
            # new file with no upstream counterpart: nothing to preserve in the CUDA
            # view; list it so the additions stay visible
            print(f"  (new file, ROCm-side) {f}")
            continue
        theirs = open(ref_path, encoding = "utf-8").read()
        checked += 1
        if strip_conditionals(ours) != strip_conditionals(theirs):
            if f not in EXPECTED_DELTAS:
                failures.append(f"{f}: CUDA view differs and is not an expected delta")
            else:
                print(f"  (expected delta) {f}: {EXPECTED_DELTAS[f]}")

    if tmp:
        subprocess.run(["git", "worktree", "remove", "--force", ref],
                       cwd = ROOT, check = True, capture_output = True)
        os.rmdir(tmp)

    print(f"Checked {checked} preprocessed files against {args.base}")
    if failures:
        print("\nFAILURES:")
        for f in failures:
            print(" ", f)
        return 1
    print("CUDA view clean")
    return 0


if __name__ == "__main__":
    sys.exit(main())
