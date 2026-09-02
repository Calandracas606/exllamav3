#pragma once
#include "context.cuh"

#if defined(USE_ROCM)

// gfx1100 clang hides __builtin_amdgcn_s_memrealtime from __has_builtin and gates direct
// use on the s-memrealtime target feature (which the driver does not enable), so the
// deadline falls back to clock64() shader ticks. The tick rate is budgeted at a high-side
// 5 GHz constant: clock64() never exceeds the boost clock, so the abort can only fire
// late, never early. (A calibrated rate would need a device symbol shared across TUs,
// which requires -fgpu-rdc; the build is -fno-gpu-rdc like the CUDA one.)

__device__ __forceinline__ uint64_t sync_deadline()
{
#if __has_builtin(__builtin_amdgcn_s_memrealtime)
    // Wall clock at a constant 100 MHz (10 ns/tick)
    return __builtin_amdgcn_s_memrealtime() + SYNC_TIMEOUT * 100000000ull;
#else
    return (uint64_t) clock64() + SYNC_TIMEOUT * 5000000000ull;
#endif
}

__device__ __forceinline__ uint32_t check_timeout(PGContext* ctx, uint64_t deadline, const char* name)
{
#if __has_builtin(__builtin_amdgcn_s_memrealtime)
    uint32_t timeout = __builtin_amdgcn_s_memrealtime() >= deadline ? 1 : 0;
#else
    uint32_t timeout = clock64() >= deadline ? 1 : 0;
#endif
    // No device printf: it breaks kernel launch on the second GPU of some
    // multi-GPU setups; the name goes to the context, printed host-side.
    // Name first, flag last (release): the flag store publishes the name bytes
    if (timeout && threadIdx.x == 0)
    {
        char* dst = ctx->sync_timeout_name;
        int i = 0;
        #pragma unroll 8
        for (; i < 63 && name[i]; ++i) dst[i] = name[i];
        dst[i] = 0;
        __threadfence_system();
        stg_release_sys_u32(&ctx->sync_timeout, 1);
    }
    return timeout;
}

#else

__device__ __forceinline__ uint64_t sync_deadline()
{
    return globaltimer_ns() + SYNC_TIMEOUT * 45000000000ull;
}

__device__ __forceinline__ uint32_t check_timeout(PGContext* ctx, uint64_t deadline, const char* name)
{
    uint32_t timeout = globaltimer_ns() >= deadline ? 1 : 0;
    if (timeout && threadIdx.x == 0)
    {
        stg_release_sys_u32(&ctx->sync_timeout, 1);
        printf(" ## Synchronization timeout in kernel: %s\n\n", name);
    }
    return timeout;
}

#endif
