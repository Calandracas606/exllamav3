// HIP definitions for the two warp-mma GEMV engine launchers (exl3_gemv.cu,
// exl3_gemv_int8.cu), whose kernels stay excluded from the ROCm build
// (build_config.py). Same declarations as the upstream headers, so the shared
// exl3_gemm.cu dispatcher and bindings.cpp link unchanged. Both always decline
// (return false / null op); exl3_gemm_gr's split-K streaming inner
// (exl3_gemm_inner_rocm.cuh) covers those shapes instead, including the
// M == 1 decode case.

#include <ATen/Tensor.h>
#include "quant/exl3_gemm.cuh"
#include "quant/exl3_gemv.cuh"
#include "quant/exl3_gemv_int8.cuh"
#include "graph.cuh"

#if defined(USE_ROCM)

// quant/exl3_gemv.cu on CUDA. Always declines; when it does, A-hadamard staging is
// the caller's responsibility.
bool exl3_gemv_try_launch
(
    void**,
    int, int, int, int, int,
    bool, bool,
    int,
    cudaStream_t,
    void**,
    bool
)
{
    return false;
}

void exl3_gemv
(
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    bool,
    bool
)
{
    TORCH_CHECK(false, "exl3_gemv: direct GEMV entry point is not built on ROCm (the regular exl3_gemm kernel covers these shapes)");
}

// quant/exl3_gemv_int8.cu on CUDA
bool exl3_gemv_int8_enabled() { return false; }

bool exl3_gemv_int8
(
    const at::Tensor&,
    const at::Tensor&,
    at::Tensor&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    const c10::optional<at::Tensor>&,
    cudaStream_t,
    Graph*
)
{
    return false;
}

int exl3_gemv_int8_max_k(int) { return 0; }

#endif
