#pragma once

#include <cstdlib>

// Launch wrapper for the exl3 grid-synchronized kernels. Included only by host .cu TUs that
// already pull in the CUDA/HIP runtime headers.
//
// On ROCm the exl3 GEMM kernel carries a lock-based sense-reversing grid barrier (group_barrier
// on the per-device locks buffer) in addition to cooperative-groups grid.sync(), so it can run
// under either launch mode: EXL3_COOP_LAUNCH=0 opts out of hipLaunchCooperativeKernel, which
// intermittently segfaults inside libamdhip64 under large multi-device process footprints
// (observed on the 7.14 wheel SDK with a 262144-token paged cache; identical launches succeed
// repeatedly before one faults). The opt-out launches the kernel plainly with the concurrency
// dimension clamped to 1, which is safe for a lock-based barrier as long as every block is
// simultaneously resident (num_sms never exceeds the physical SM count and the launches are
// stream-serialized). Cooperative launch remains the default on all platforms; CUDA kernels
// always use grid.sync().
static inline bool exl3_coop_launch_enabled()
{
#if defined(USE_ROCM)
    static const bool enabled = []
    {
        const char* env = std::getenv("EXL3_COOP_LAUNCH");
        return !(env && atoi(env) == 0);
    }();
    return enabled;
#else
    return true;
#endif
}

static inline cudaError_t exl3_launch_grid_sync_kernel
(
    void* kernel,
    dim3 grid,
    dim3 block,
    void** args,
    size_t smem,
    cudaStream_t stream
)
{
#if defined(USE_ROCM)
    if (!exl3_coop_launch_enabled())
    {
        grid.z = 1;
        return cudaLaunchKernel((const void*) kernel, grid, block, args, smem, stream);
    }
#endif
    return cudaLaunchCooperativeKernel(kernel, grid, block, args, smem, stream);
}
