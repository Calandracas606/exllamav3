#pragma once

// Approximate tanh

// The JIT loader compiles .cpp files with the host compiler, where HIP's headers provide
// neither the device-only intrinsics these helpers use nor CUDA's host_defines macros.
// Guard the device helpers to the HIP compiler passes (see AMD's porting guide,
// "Identifying host or device compilation pass": __HIPCC__ is defined for hipcc/amdclang
// and undefined for a standard host compiler); device TUs still compile them.

#if !defined(USE_ROCM) || defined(__HIPCC__)

// Clang's HIP math declares rsqrtf __device__-only (CUDA's is __host__ __device__), so
// host code calling it (e.g. attention.cu's scale computation) needs a host overload.
// The overload must be declared unconditionally, not under a host-pass guard: clang's
// HIP device pass also semantically checks __host__ function bodies, and a host-only
// #if would leave those calls unresolved there. Device callers are unaffected - the
// __device__-only declaration is not callable from them, so the library's serves them
// and overload resolution is never ambiguous.
#if defined(USE_ROCM)
__host__ inline float rsqrtf(float x) { return 1.0f / sqrtf(x); }
#endif

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

#endif  // !defined(USE_ROCM) || defined(__HIPCC__)

#if defined(USE_ROCM)
#if !defined(__HIPCC__)
// Host parse (standard compiler): provide CUDA's host_defines macros so upstream headers'
// typedefs parse; HIP's headers define them only for the HIP compiler passes
#define __align__(x) __attribute__((aligned(x)))
// Declaration only: the half-combining intrinsic appears in device-constructor bodies,
// which a host parse still reads but HIP's headers declare only for the HIP compiler
__half2 __halves2half2(__half a, __half b);
#endif
#include "compat_rocm.cuh"
// Warp-sync builtins take a mask on CUDA but a 64-bit mask on HIP (templates
// static_assert on smaller integers). Like vLLM's HIP compat, the mask is
// dropped: on AMD the non-sync variants act with a full warp mask, which is
// what every call site here means (masks are always full-warp or advisory).
// Do NOT widen to HIP's masked 64-bit templates: the masked __syncwarp
// hardware-faults on wave32 gfx1100 once the barrier executes.
#define __syncwarp(...) __syncwarp()
#endif
