# Vendored from https://github.com/ROCm/FlyDSL (Apache-2.0) at main@47009a3.
# The pip wheel flydsl==0.3.1 ships only the compiler/runtime; the production
# kernels live in the repo's kernels/ package. These are the files needed for
# the RDNA3 WMMA GEMM:
#   kernels/gemm/rdna3_f16_gemm.py  -> ./rdna3_f16_gemm.py
#   kernels/common/kernels_common.py -> ./kernels_common.py
#   kernels/common/mem_ops.py        -> ./mem_ops.py
#   kernels/common/buffer_ops.py     -> ./buffer_ops.py
# Only the `from kernels.common...` imports were rewritten to relative imports.
