"""Source list for the C++ extension, shared by setup.py (precompiled) and the JIT loader.

On ROCm, tensor-core and other sources that do not build on HIP are excluded; the
exclusion list is kept consistent with the ROCm binding table in bindings.cpp by
tests/test_build_config.py.
"""

import os

ROCM_EXCLUDE_DIRS = set()

# The whole extension builds on HIP: the per-(K,cb) comp_units, the exl3_gemm
# stack (kernels route through exl3_gemm_inner_rocm.cuh), quantize.cu,
# attention.cu, dsv4_compress.cu, sampling_fused.cu, every libtorch BC host TU,
# the batched MoE dispatcher (cuda::atomic_ref via the libcu++-shaped shim in
# ptx_rocm_compat.cuh over __hip_atomic builtins), the quantized KV cache and
# the CPU->GPU MoE handoff (stream-wait ops through the MemOps interface).
# The two warp-mma GEMV engines are the only exclusion pending a port;
# exl3_rocm_stubs.cpp provides declining launchers so the shared exl3_gemm.cu
# dispatcher links and falls through to the regular templated kernel.
ROCM_EXCLUDE_FILES = {
    'quant/exl3_gemv.cu', 'quant/exl3_gemv_int8.cu',
}


def get_sources(sources_dir, is_rocm, base_dir = None):
    """Walk the extension source directory and return the source file list. Stale
    hipify intermediates (*.hip, *_hip.*) are skipped. With base_dir the paths are
    relative to it (setup.py), otherwise absolute (JIT loader)."""

    # CUDA-only kernel instantiations whose host dispatchers are excluded above:
    # the fused MoE kernels (cuda::atomic_ref scheduler) and the int8 GEMV
    # (cp.async) comp units. Nothing references these kernel pointers on ROCm.
    rocm_exclude_prefixes = ('exl3_gemv_int8_inst_',)

    sources = []
    for root, _, files in os.walk(sources_dir):
        for file in files:
            if not file.endswith(('.c', '.cpp', '.cu')): continue
            if file.endswith('.hip') or '_hip' in file: continue
            if is_rocm and file.startswith(rocm_exclude_prefixes): continue
            rel_path = os.path.relpath(os.path.join(root, file), start = sources_dir)
            norm_rel = rel_path.replace('\\', '/')
            if is_rocm:
                parts = norm_rel.split('/')
                if any(d in parts for d in ROCM_EXCLUDE_DIRS): continue
                if norm_rel in ROCM_EXCLUDE_FILES: continue
            full = os.path.join(root, file)
            if base_dir is not None:
                sources.append(os.path.relpath(full, start = base_dir))
            else:
                sources.append(os.path.abspath(full))
    return sources
