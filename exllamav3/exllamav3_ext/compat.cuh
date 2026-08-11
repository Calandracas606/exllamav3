#pragma once

// Shared helpers (both CUDA and ROCm)

__forceinline__ __device__ float copysignf_pos(float a, float b)
{
    float r;
    r = __int_as_float(__float_as_int(a) | (__float_as_int(b) & 0x80000000));
    return r;
}

#if defined(USE_ROCM) || (defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 750 || CUDART_VERSION < 11000))

__inline__ __device__ float tanh_opt(float x)
{
    const float exp_val = -1.f * fabs(2 * x);
    return copysignf_pos((1.0f - __expf(exp_val)) / (__expf(exp_val) + 1.0f), x);
}

#else

__inline__ __device__ float tanh_opt(float x)
{
    float r;
    asm("tanh.approx.f32 %0,%1; \n\t" : "=f"(r) : "f"(x));
    return r;
}

#endif

// Bitfield macros and FragB type for EXL3 dequant path.
// On CUDA these use PTX inline asm; on ROCm they use plain C++ equivalents.
#if defined(USE_ROCM)
#include "compat_rocm.cuh"
#else
// On CUDA, ptx.cuh provides these via PTX asm. Define them here so
// reconstruct.cu no longer needs to include ptx.cuh (which has many
// tensor-core definitions irrelevant to the dequant path).
struct FragB { half2 elems[2]; __device__ half2& operator[](int i) { return elems[i]; } };

#define FSHF_IMM(dst, lo, hi, imm) asm("shf.r.wrap.b32 %0, %1, %2, " #imm ";" : "=r"(dst) : "r"(lo), "r"(hi))
#define BFE16_IMM(dst, src, imm) asm("bfe.u32 %0, %1, " #imm ", 16;" : "=r"(dst) : "r"(src))
#endif
