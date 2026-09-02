#pragma once

#include <hip/hip_runtime.h>

// The polyfills below use device-only intrinsics and are guarded to the HIP compiler
// passes (see the guard in compat.cuh): hidden from host parses, compiled for hipcc
// and on CUDA

#if !defined(USE_ROCM) || defined(__HIPCC__)

#define __shfl_xor_sync(mask, var, ...) __shfl_xor(var, __VA_ARGS__)
#define __shfl_sync(mask, var, ...) __shfl(var, __VA_ARGS__)
#define __shfl_down_sync(mask, var, ...) __shfl_down(var, __VA_ARGS__)
#define __shfl_up_sync(mask, var, ...) __shfl_up(var, __VA_ARGS__)
#define __ballot_sync(mask, ...) __ballot(__VA_ARGS__)

namespace polyfill
{

__device__ __forceinline__ int dp4a(uint32_t a, uint32_t b, int c)
{
    int result = c;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        uint32_t va = static_cast<uint32_t>(static_cast<uint8_t>((a >> (i * 8)) & 0xFF));
        uint32_t vb = static_cast<uint32_t>(static_cast<uint8_t>((b >> (i * 8)) & 0xFF));
        result += va * vb;
    }
    return result;
}

__device__ __forceinline__ uint32_t dp4a(uint32_t a, uint32_t b, uint32_t c)
{
    return static_cast<uint32_t>(dp4a(a, b, static_cast<int>(c)));
}

__device__ __forceinline__ half2 hmax2(half2 a, half2 b)
{
    return __halves2half2(__hmax(__low2half(a), __low2half(b)),
                          __hmax(__high2half(a), __high2half(b)));
}

__device__ __forceinline__ half2 hmin2(half2 a, half2 b)
{
    return __halves2half2(__hmin(__low2half(a), __low2half(b)),
                          __hmin(__high2half(a), __high2half(b)));
}

#if defined(__HIPCC__)
__device__ __forceinline__ __hip_bfloat16 float2bfloat16_rz(float f)
{
    uint32_t u = __float_as_uint(f);
    uint16_t r = static_cast<uint16_t>(u >> 16);
    return __ushort_as_bfloat16(r);
}

__device__ __forceinline__ __hip_bfloat16 float2bfloat16_rn(float f)
{
    return __float2bfloat16(f);
}
#endif

} // namespace polyfill

#ifndef __dp4a
#define __dp4a polyfill::dp4a
#endif
#ifndef __hmax2
#define __hmax2 polyfill::hmax2
#endif
#ifndef __hmin2
#define __hmin2 polyfill::hmin2
#endif
#if defined(__HIPCC__)
#ifndef __float2bfloat16_rz
#define __float2bfloat16_rz polyfill::float2bfloat16_rz
#endif
#ifndef __float2bfloat16_rn
#define __float2bfloat16_rn polyfill::float2bfloat16_rn
#endif
#endif

#endif  // !defined(USE_ROCM) || defined(__HIPCC__)
