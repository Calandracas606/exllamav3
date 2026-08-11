#pragma once

// ============================================================================
// ROCm polyfills for CUDA intrinsics not provided by HIP.
// Included by compat.cuh when building for ROCm (USE_ROCM defined).
// ============================================================================

#include <hip/hip_runtime.h>

// HIP has non-sync __shfl_* variants that don't require a mask argument.
// Map the CUDA __shfl_*_sync(mask, ...) calls to the simpler HIP equivalents
// by dropping the mask. The C preprocessor's non-recursive expansion rule
// ("blue paint") prevents infinite recursion.
#define __shfl_xor_sync(mask, var, ...) __shfl_xor(var, __VA_ARGS__)
#define __shfl_sync(mask, var, ...) __shfl(var, __VA_ARGS__)
#define __shfl_down_sync(mask, var, ...) __shfl_down(var, __VA_ARGS__)
#define __shfl_up_sync(mask, var, ...) __shfl_up(var, __VA_ARGS__)
#define __ballot_sync(mask, ...) __ballot(__VA_ARGS__)

// Functions are defined in a polyfill namespace and mapped via #define,
// avoiding direct definition of reserved __-prefixed symbols.
namespace polyfill
{

__device__ __forceinline__ int dp4a(uint32_t a, uint32_t b, int c)
{
    int result = c;
    #pragma unroll
    for (int i = 0; i < 4; i++)
    {
        int8_t va = (int8_t)((a >> (i * 8)) & 0xFF);
        int8_t vb = (int8_t)((b >> (i * 8)) & 0xFF);
        result += va * vb;
    }
    return result;
}

__device__ __forceinline__ uint32_t dp4a(uint32_t a, uint32_t b, uint32_t c)
{
    return (uint32_t)dp4a(a, b, (int)c);
}

// HIP provides __hmax2/__hmin2 for bfloat162 but NOT for half2.
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
// HIP lacks these float-to-bfloat16 conversion variants used by gdn.cu.
__device__ __forceinline__ __hip_bfloat16 float2bfloat16_rz(float f)
{
    uint32_t u = __float_as_uint(f);
    uint16_t r = (uint16_t)(u >> 16);
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

// Bitfield macros for reconstruct/dequant path. On CUDA these come from ptx.cuh
// via PTX inline asm. On ROCm, use plain C++ equivalents.
struct FragB { half2 elems[2]; __device__ half2& operator[](int i) { return elems[i]; } };

#define FSHF_IMM(dst, lo, hi, imm) \
    do { uint64_t _m = ((uint64_t)(hi) << 32) | (uint32_t)(lo); (dst) = (uint32_t)(_m >> (imm)); } while(0)
#define BFE16_IMM(dst, src, imm) (dst) = ((src) >> (imm)) & 0xFFFFu

// 64-bit bitfield extract (CUDA uses PTX bfe.u64; ROCm uses plain C++)
static __forceinline__ __device__ uint32_t bfe64(uint32_t lo, uint32_t hi, int offset, int length)
{
    uint64_t value = (static_cast<uint64_t>(hi) << 32) | static_cast<uint64_t>(lo);
    return static_cast<uint32_t>((value >> offset) & ((1ULL << length) - 1));
}
