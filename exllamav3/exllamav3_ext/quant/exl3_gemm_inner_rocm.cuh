#pragma once

// HIP body behind the shared kernel wrappers (exl3_gemm_kernel.cuh,
// exl3_moe_kernel.cuh); dispatcher, kernel tables, autotuner and comp units are
// shared with CUDA. Same contract as the CUDA inner: blockIdx.x slices the
// (k x n) tile space, each column tile assembled by a lock cascade in reverse k
// order (highest-k segment acquires first and accumulates into C in place, the
// k == 0 block writes the final result), no grid-wide sync, and locks used only
// as locks so every caller's disjoint lock range stays disjoint. Decode tiers on
// (bits, cb): 4/6 with mul1 decode lane-local through funnel/byte-dot intrinsics,
// everything else through the shared dq_dispatch. NB: never name a ROCm-side
// header *_hip.cuh - torch's in-place hipify clobbers files with that pattern.

#include "../ptx.cuh"
#include "exl3_kernel_map.cuh"
#include "hadamard_inner.cuh"
#include "exl3_dq.cuh"
#include "exl3_devctx.cuh"

// The #undef matters: exl3_moe_common.cuh's 90 KB fallback would exceed
// gfx1100's 64 KB LDS limit.
#undef SMEM_MAX
#define SMEM_MAX (8 * 1024)

// staged in shared memory, declared once per kernel: per-instantiation __shared__
// would request the LDS once per instantiation
#define EXL3_INNER_SH_FLOATS(ts_n) (16 * (ts_n))

namespace exl3_rocm_inner
{

// funnel shift is native on both vendors
__device__ __forceinline__ uint32_t alignbit16(uint32_t hi, uint32_t lo, int imm)
{
    return __funnelshift_r(lo, hi, imm) & 0xFFFFu;
}

// byte lanes sum to <= 1020, so the low 16 bits are exact
__device__ __forceinline__ uint32_t dot4_add(uint32_t x, uint32_t c)
{
    return __builtin_amdgcn_udot4(x, 0x01010101u, c, false);
}

// same arithmetic as codebook.cuh's decode_mul1_product_2
__device__ __forceinline__ float decode_w_mul1(uint32_t code)
{
    uint32_t u = dot4_add(code * 0x83DCD12Du, 0x6400u) & 0xFFFFu;
    __half h = __ushort_as_half(static_cast<unsigned short>(u));
    __half ki = __ushort_as_half(static_cast<unsigned short>(0x1EEE));
    __half kb = __ushort_as_half(static_cast<unsigned short>(0xC931));
    float v = __half2float(h) * __half2float(ki) + __half2float(kb);
    return __half2float(__float2half(v));
}

// Column lock (ptx.cuh protocol): counts completed k-tiles of one column tile;
// the topmost block resets it to 0 for the next call

__device__ __forceinline__ void lock_acquire(int* lock, int stage)
{
    if (threadIdx.x == 0)
    {
        unsigned int* a = (unsigned int*) lock;
        unsigned int state;
        do
        {
            state = __hip_atomic_load(a, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        }
        while (state != (unsigned int) stage);
    }
    __syncthreads();
}

__device__ __forceinline__ void lock_release(int* lock, int val, bool reset)
{
    __syncthreads();
    if (threadIdx.x == 0)
    {
        unsigned int* a = (unsigned int*) lock;
        if (reset)
        {
            __hip_atomic_store(a, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
            return;
        }
        __hip_atomic_fetch_add(a, (unsigned int) val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
}

// phase functions must stay __forceinline__: a __shared__ array whose address
// escapes into a non-inlined callee is demoted to local memory
struct SegCtx
{
    const half* __restrict__ A;
    const uint32_t* __restrict__ B;
    float* __restrict__ sh_c;       // [16][cols] segment partial
    int size_m;
    int size_k;
    int kt0, kt1;                   // trellis k-subtile range of the segment (16 wide each)
    int nsub_total;                 // subtiles per k row of the whole matrix (size_n / 16)
    int subs_tile;                  // subtiles in this column tile (TILESIZE_N / 16)
    int col;                        // column tile index
};

// Tier A: bits 4, cb 2. 32 words per subtile, 8 lanes per (subtile, m row) item

__device__ __forceinline__ void phase1_b4c2(SegCtx& c)
{
    int lane = threadIdx.x & 7;
    int rem = threadIdx.x >> 3;
    int groups = blockDim.x >> 3;
    int items = c.subs_tile * c.size_m;

    for (int idx = rem; idx < items; idx += groups)
    {
        int s = idx % c.subs_tile;
        int n = c.col * c.subs_tile + s;        // global subtile
        int m = idx / c.subs_tile;

        const uint32_t* base =
            (const uint32_t*) c.B + (size_t) c.kt0 * (c.nsub_total * 32) + (size_t) n * 32;
        const half* x = c.A + (size_t) m * c.size_k;
        float acc0 = 0.f, acc1 = 0.f;

        #define PAIR4(HI, LO, XK)                                                     \
        acc0 += decode_w_mul1(alignbit16(HI, LO, 28)) * (XK)[0]                       \
              + decode_w_mul1(alignbit16(HI, LO, 24)) * (XK)[1]                       \
              + decode_w_mul1(alignbit16(HI, LO, 20)) * (XK)[8]                       \
              + decode_w_mul1(alignbit16(HI, LO, 16)) * (XK)[9];                      \
        acc1 += decode_w_mul1(alignbit16(HI, LO, 12)) * (XK)[0]                       \
              + decode_w_mul1(alignbit16(HI, LO, 8))  * (XK)[1]                       \
              + decode_w_mul1(alignbit16(HI, LO, 4))  * (XK)[8]                       \
              + decode_w_mul1(alignbit16(HI, LO, 0))  * (XK)[9];

        for (int t = c.kt0; t < c.kt1; ++t)
        {
            const uint32_t* p32 = base + (size_t) (t - c.kt0) * (c.nsub_total * 32);
            uint32_t w0 = p32[4 * lane];
            uint32_t w1 = p32[4 * lane + 1];
            uint32_t w2 = p32[4 * lane + 2];
            uint32_t w3 = p32[4 * lane + 3];
            uint32_t prev = p32[(4 * lane + 31) & 31];

            float xf[16];
            const half* xk = x + t * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) xf[i] = __half2float(xk[i]);

            PAIR4(prev, w0, xf)
            PAIR4(w0, w1, xf + 2)
            PAIR4(w1, w2, xf + 4)
            PAIR4(w2, w3, xf + 6)
        }
        #undef PAIR4

        float* out = c.sh_c + (size_t) m * (c.subs_tile * 16) + (size_t) s * 16;
        out[lane] = acc0;
        out[lane + 8] = acc1;
    }
}

// Tier B: bits 6, cb 2. 48 words per subtile, 6-word window algebra

__device__ __forceinline__ void phase1_b6c2(SegCtx& c)
{
    int lane = threadIdx.x & 7;
    int rem = threadIdx.x >> 3;
    int groups = blockDim.x >> 3;
    int items = c.subs_tile * c.size_m;

    for (int idx = rem; idx < items; idx += groups)
    {
        int s = idx % c.subs_tile;
        int n = c.col * c.subs_tile + s;
        int m = idx / c.subs_tile;

        const uint32_t* base =
            (const uint32_t*) c.B + (size_t) c.kt0 * (c.nsub_total * 48) + (size_t) n * 48;
        const half* x = c.A + (size_t) m * c.size_k;
        float acc0 = 0.f, acc1 = 0.f;

        #define CA(HI, LO, SH) \
            ((SH) < 32 ? alignbit16(HI, LO, (SH)) : ((HI) >> ((SH) - 32)) & 0xFFFFu)

        for (int t = c.kt0; t < c.kt1; ++t)
        {
            const uint32_t* p32 = base + (size_t) (t - c.kt0) * (c.nsub_total * 48);
            uint32_t w0 = p32[6 * lane];
            uint32_t w1 = p32[6 * lane + 1];
            uint32_t w2 = p32[6 * lane + 2];
            uint32_t w3 = p32[6 * lane + 3];
            uint32_t w4 = p32[6 * lane + 4];
            uint32_t w5 = p32[6 * lane + 5];
            uint32_t m1 = p32[(6 * lane + 47) % 48];

            float xf[16];
            const half* xk = x + t * 16;
            #pragma unroll
            for (int i = 0; i < 16; ++i) xf[i] = __half2float(xk[i]);

            acc0 += decode_w_mul1(CA(m1, w0, 26)) * xf[0]
                  + decode_w_mul1(CA(m1, w0, 20)) * xf[1]
                  + decode_w_mul1(CA(m1, w0, 14)) * xf[8]
                  + decode_w_mul1(CA(m1, w0, 8))  * xf[9];
            acc1 += decode_w_mul1(CA(w0, w1, 34)) * xf[0]
                  + decode_w_mul1(CA(w0, w1, 28)) * xf[1]
                  + decode_w_mul1(CA(w0, w1, 22)) * xf[8]
                  + decode_w_mul1(CA(w0, w1, 16)) * xf[9];
            acc0 += decode_w_mul1(CA(w1, w2, 42)) * xf[2]
                  + decode_w_mul1(CA(w1, w2, 36)) * xf[3]
                  + decode_w_mul1(CA(w1, w2, 30)) * xf[10]
                  + decode_w_mul1(CA(w1, w2, 24)) * xf[11];
            acc1 += decode_w_mul1(CA(w1, w2, 18)) * xf[2]
                  + decode_w_mul1(CA(w1, w2, 12)) * xf[3]
                  + decode_w_mul1(CA(w1, w2, 6))  * xf[10]
                  + decode_w_mul1(CA(w1, w2, 0))  * xf[11];
            acc0 += decode_w_mul1(CA(w2, w3, 26)) * xf[4]
                  + decode_w_mul1(CA(w2, w3, 20)) * xf[5]
                  + decode_w_mul1(CA(w2, w3, 14)) * xf[12]
                  + decode_w_mul1(CA(w2, w3, 8))  * xf[13];
            acc1 += decode_w_mul1(CA(w3, w4, 34)) * xf[4]
                  + decode_w_mul1(CA(w3, w4, 28)) * xf[5]
                  + decode_w_mul1(CA(w3, w4, 22)) * xf[12]
                  + decode_w_mul1(CA(w3, w4, 16)) * xf[13];
            acc0 += decode_w_mul1(CA(w4, w5, 42)) * xf[6]
                  + decode_w_mul1(CA(w4, w5, 36)) * xf[7]
                  + decode_w_mul1(CA(w4, w5, 30)) * xf[14]
                  + decode_w_mul1(CA(w4, w5, 24)) * xf[15];
            acc1 += decode_w_mul1(CA(w4, w5, 18)) * xf[6]
                  + decode_w_mul1(CA(w4, w5, 12)) * xf[7]
                  + decode_w_mul1(CA(w4, w5, 6))  * xf[14]
                  + decode_w_mul1(CA(w4, w5, 0))  * xf[15];
        }
        #undef CA

        float* out = c.sh_c + (size_t) m * (c.subs_tile * 16) + (size_t) s * 16;
        out[lane] = acc0;
        out[lane + 8] = acc1;
    }
}

// Tier C: everything else. One warp per subtile through dq_dispatch

template <int bits, int cb>
__device__ __forceinline__ void phase1_dq(SegCtx& c)
{
    int lane_id = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    int warps = blockDim.x >> 5;

    int r0 = 2 * (lane_id & 3);
    int c0 = 2 * (lane_id >> 3) + ((lane_id & 7) >> 2);
    int rows[4] = { r0, r0 + 1, r0 + 8, r0 + 9 };

    for (int idx = warp; idx < c.subs_tile; idx += warps)
    {
        int n = c.col * c.subs_tile + idx;
        const uint32_t* sub_base =
            (const uint32_t*) c.B + (size_t) c.kt0 * (c.nsub_total * (8 * bits)) + (size_t) n * (8 * bits);

        float acc[16][2];
        #pragma unroll
        for (int m = 0; m < 16; ++m) { acc[m][0] = 0.f; acc[m][1] = 0.f; }

        for (int t = c.kt0; t < c.kt1; ++t)
        {
            const uint32_t* ptr = sub_base + (size_t) (t - c.kt0) * (c.nsub_total * (8 * bits));

            FragB frag[2];
            dq_dispatch<bits, cb>(ptr, lane_id * 8, frag[0], frag[1]);

            #pragma unroll
            for (int m = 0; m < 16; ++m)
            {
                if (m < c.size_m)
                {
                    const half* x = c.A + (size_t) m * c.size_k + (size_t) t * 16;
                    float x0 = __half2float(x[rows[0]]);
                    float x1 = __half2float(x[rows[1]]);
                    float x2 = __half2float(x[rows[2]]);
                    float x3 = __half2float(x[rows[3]]);
                    acc[m][0] += __half2float(frag[0][0].x) * x0
                               + __half2float(frag[0][0].y) * x1
                               + __half2float(frag[0][1].x) * x2
                               + __half2float(frag[0][1].y) * x3;
                    acc[m][1] += __half2float(frag[1][0].x) * x0
                               + __half2float(frag[1][0].y) * x1
                               + __half2float(frag[1][1].x) * x2
                               + __half2float(frag[1][1].y) * x3;
                }
            }
        }

        #pragma unroll
        for (int m = 0; m < 16; ++m)
        {
            if (m >= c.size_m) continue;
            float a0 = acc[m][0], a1 = acc[m][1];
            a0 += __shfl_xor_sync(0xffffffffu, a0, 1);
            a1 += __shfl_xor_sync(0xffffffffu, a1, 1);
            a0 += __shfl_xor_sync(0xffffffffu, a0, 2);
            a1 += __shfl_xor_sync(0xffffffffu, a1, 2);

            if ((lane_id & 3) == 0)
            {
                float* out = c.sh_c + (size_t) m * (c.subs_tile * 16) + (size_t) idx * 16;
                out[c0] = a0;
                out[c0 + 8] = a1;
            }
        }
    }
}

}  // namespace exl3_rocm_inner

// shared inner entry point; C is [m][n] with row stride size_n. post_scale
// applies only when shmem_out_had is set

template<EXL3_GEMM_T_ARGS, bool shmem_out_had>
__device__ void exl3_gemm_kernel_inner
(
    const half* __restrict__  A,
    const uint16_t* __restrict__ B,
    void* __restrict__ C,
    const int size_m,
    const int size_k,
    const int size_n,
    int* __restrict__ locks,
    const half* post_scale,
    int size_n_stride = 0,       // full width of B and C rows when computing a column slice (0: = size_n)
    float* __restrict__ sh = nullptr
)
{
    using namespace exl3_rocm_inner;

    if (size_n_stride == 0) size_n_stride = size_n;
    const int n_full = size_n_stride;    // B rows and C rows span the full matrix width
    constexpr int TS_N = TILESIZE_N;
    float* sh_c = sh;

    int tiles_k = size_k / 16;                 // trellis k-subtiles (16 wide)
    int tiles_n = size_n / TS_N;
    int units = tiles_k * tiles_n;
    int num_slices = gridDim.x;
    int beg = (int) ((int64_t) units * blockIdx.x / num_slices);
    int end = (int) ((int64_t) units * (blockIdx.x + 1) / num_slices);

    SegCtx c;
    c.A = A;
    c.B = (const uint32_t*) B;
    c.sh_c = sh_c;
    c.size_m = size_m;
    c.size_k = size_k;
    c.nsub_total = n_full / 16;
    c.subs_tile = TS_N / 16;

    while (beg < end)
    {
        int col = beg / tiles_k;
        int seg_k0 = beg % tiles_k;
        int seg_k1 = ((end - 1) / tiles_k == col) ? ((end - 1) % tiles_k) + 1 : tiles_k;

        c.kt0 = seg_k0;
        c.kt1 = seg_k1;
        c.col = col;
        int cols = c.subs_tile * 16;

        if constexpr (cb == 2 && bits == 4)
        {
            phase1_b4c2(c);
        }
        else if constexpr (cb == 2 && bits == 6)
        {
            phase1_b6c2(c);
        }
        else if constexpr (bits > 0)
        {
            phase1_dq<bits, cb>(c);
        }
        // bits == 0 never reaches the inner (the caller switches on K first)

        __syncthreads();

        int lock_i = tiles_k - seg_k1;
        int lock_d = seg_k1 - seg_k0;
        int* lock = &locks[col];
        lock_acquire(lock, lock_i);

        bool first = (lock_i == 0);
        bool last = (lock_i + lock_d == tiles_k);

        if (!first)
        {
            for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x)
            {
                int m = i / cols;
                int n = i % cols;
                if constexpr (c_fp32)
                    sh_c[i] += ((const float*) C)[(size_t) m * n_full + (size_t) col * TS_N + n];
                else
                    sh_c[i] += __half2float(((const half*) C)[(size_t) m * n_full + (size_t) col * TS_N + n]);
            }
            __syncthreads();
        }

        if (!last)
        {
            for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x)
            {
                int m = i / cols;
                int n = i % cols;
                if constexpr (c_fp32)
                    ((float*) C)[(size_t) m * n_full + (size_t) col * TS_N + n] = sh_c[i];
                else
                    ((half*) C)[(size_t) m * n_full + (size_t) col * TS_N + n] = __float2half(sh_c[i]);
            }
        }
        else if (shmem_out_had)
        {
            // final block: output Hadamard (without post_scale only the
            // 1/sqrt(128) factor, no per-column scale)
            int nb = cols / 128;
            int warp = threadIdx.x >> 5;
            int lane = threadIdx.x & 31;
            int warps = blockDim.x >> 5;
            int slots = size_m * nb;
            for (int slot = warp; slot < slots; slot += warps)
            {
                int m = slot / nb;
                int b = slot % nb;
                const float* p = sh_c + m * cols + b * 128;
                float v0 = p[lane * 4 + 0];
                float v1 = p[lane * 4 + 1];
                float v2 = p[lane * 4 + 2];
                float v3 = p[lane * 4 + 3];

                float s0 = v0 + v1, d0 = v0 - v1;
                float s1 = v2 + v3, d1 = v2 - v3;
                float h0 = s0 + s1, h1 = d0 + d1, h2 = s0 - s1, h3 = d0 - d1;

                shuffle_had_f4x32(h0, h1, h2, h3, lane);

                const float rs = 0.088388347648f;   // 1/sqrt(128)
                if (post_scale)
                {
                    const half* sb = post_scale + ((size_t) col * TS_N + b * 128) % n_full;
                    h0 *= rs * __half2float(sb[lane * 4 + 0]);
                    h1 *= rs * __half2float(sb[lane * 4 + 1]);
                    h2 *= rs * __half2float(sb[lane * 4 + 2]);
                    h3 *= rs * __half2float(sb[lane * 4 + 3]);
                }
                else
                {
                    h0 *= rs; h1 *= rs; h2 *= rs; h3 *= rs;
                }

                size_t n0 = (size_t) m * n_full + (size_t) col * TS_N + b * 128 + lane * 4;
                if constexpr (c_fp32)
                {
                    float* out = (float*) C + n0;
                    out[0] = h0; out[1] = h1; out[2] = h2; out[3] = h3;
                }
                else
                {
                    half* out = (half*) C + n0;
                    out[0] = __float2half(h0); out[1] = __float2half(h1);
                    out[2] = __float2half(h2); out[3] = __float2half(h3);
                }
            }
        }
        else
        {
            for (int i = threadIdx.x; i < size_m * cols; i += blockDim.x)
            {
                int m = i / cols;
                int n = i % cols;
                if constexpr (c_fp32)
                    ((float*) C)[(size_t) m * n_full + (size_t) col * TS_N + n] = sh_c[i];
                else
                    ((half*) C)[(size_t) m * n_full + (size_t) col * TS_N + n] = __float2half(sh_c[i]);
            }
        }

        lock_release(lock, lock_d, last);

        // phase 1 must store fresh values: the accumulate path adds into sh_c
        beg = (col + 1) * tiles_k;
    }
}

// M=1 fused dequant-GEMV kernels (plain HIP: single-rounding hfma decode, uint4
// strip loads, split-K atomics, and the b6 lane-layout algebra — no LUT, no
// dependencies). Compiled only in TUs that define EXL3_ROCM_GEMV_M1
// before including this header (currently exl3_gemm.cu) so the comp-unit TUs
// don't instantiate them.
// ============================================================================
#if defined(EXL3_ROCM_GEMV_M1)

#include <ATen/cuda/CUDAContext.h>
#include <set>
#include <tuple>
#include <c10/cuda/CUDAGuard.h>
#include "hadamard.cuh"

namespace exl3_rocm_gemv
{


// vN: configurable amortized GEMV. SUBS = subtiles per thread (multiple of
// 2); each thread computes a 16 x (16*SUBS) output tile: SUBS*16 fp32 chains,
// SUBS*16 fdot2 per k-tile, x pairs and decode constants amortized SUBS/2x
// more than v11. Cross-pair emission order per LLVM #186179; explicit stores.
__device__ __forceinline__ __half dec_half(uint32_t v);

template <int SUBS>
__device__ __forceinline__ void gemv_vN_dev
(
    const uint16_t* __restrict__ g_tr16,
    const __half* __restrict__ g_xh,
    float* __restrict__ g_y,
    int n_subs, int k_tiles, int splits, int y_row_stride
)
{
    static_assert(SUBS % 2 == 0, "SUBS must be even (subtile pairs)");
    constexpr int W = 32;
    constexpr int SLOTS = 32;                 // 256 threads / 8 cA
    constexpr int SUBS_PER_BLOCK = SLOTS * SUBS;
    const uint32_t* g_tr = reinterpret_cast<const uint32_t*>(g_tr16);
    const int t = threadIdx.x;
    const int cA = t % 8;
    const int slot = t / 8;
    const int nt0 = blockIdx.x * SUBS_PER_BLOCK + slot * SUBS;
    if (nt0 + SUBS - 1 >= n_subs) return;

    const int k_per = (k_tiles + splits - 1) / splits;
    const int kt0 = blockIdx.y * k_per;
    const int kt1 = min(kt0 + k_per, k_tiles);

    const uint32_t* sub0 = g_tr + (int64_t)nt0 * W;
    float a[SUBS][16];
    #pragma unroll
    for (int j = 0; j < SUBS; ++j)
        #pragma unroll
        for (int i = 0; i < 16; ++i) a[j][i] = 0.0f;

    for (int kt = kt0; kt < kt1; ++kt)
    {
        const uint32_t* sub = sub0 + (int64_t)kt * n_subs * W;
        uint4 sv[SUBS];
        uint32_t wrap[SUBS];
        // DPP8 wrap fetch: each lane needs the PREVIOUS lane-of-octet's last
        // word (lane i <- lane (i+7)%8). The octet shares one nt0 (same slot),
        // so source and dest lanes are always equally active - DPP8's direct
        // VGPR cross-lane read is safe and replaces an LDS round-trip
        // (ds_bpermute) per subtile per k-tile.
        #pragma unroll
        for (int j = 0; j < SUBS; ++j)
        {
            sv[j] = *reinterpret_cast<const uint4*>(sub + j * W + 4 * cA);
            wrap[j] = __builtin_amdgcn_mov_dpp8(sv[j].w, 0xd63447u);
        }
        const __half2* xh2 = reinterpret_cast<const __half2*>(g_xh + kt * 16);

        #pragma unroll
        for (int rh = 0; rh < 4; ++rh)
        {
            // per-subtile (word-current, word-prev) for this rh
            uint32_t wc[SUBS], wr[SUBS];
            #pragma unroll
            for (int j = 0; j < SUBS; ++j)
            {
                switch (rh)
                {
                    case 0:  wc[j] = sv[j].x; wr[j] = wrap[j];  break;
                    case 1:  wc[j] = sv[j].y; wr[j] = sv[j].x;  break;
                    case 2:  wc[j] = sv[j].z; wr[j] = sv[j].y;  break;
                    default: wc[j] = sv[j].w; wr[j] = sv[j].z;  break;
                }
            }
            const __half2 xlo = xh2[rh];
            const __half2 xhi = xh2[rh + 4];
            #pragma unroll
            for (int pr = 0; pr < SUBS / 2; ++pr)
            {
                const int j0 = 2 * pr, j1 = 2 * pr + 1;
                const __half2 wA0 = __halves2half2
                (
                    dec_half(__funnelshift_r(wc[j0], wr[j0], 28u) & 0xffffu),
                    dec_half(__funnelshift_r(wc[j0], wr[j0], 24u) & 0xffffu)
                );
                const __half2 wA1 = __halves2half2
                (
                    dec_half(__funnelshift_r(wc[j0], wr[j0], 20u) & 0xffffu),
                    dec_half(__funnelshift_r(wc[j0], wr[j0], 16u) & 0xffffu)
                );
                const __half2 wB0 = __halves2half2
                (
                    dec_half(__funnelshift_r(wc[j0], wr[j0], 12u) & 0xffffu),
                    dec_half(__funnelshift_r(wc[j0], wr[j0], 8u) & 0xffffu)
                );
                const __half2 wB1 = __halves2half2
                (
                    dec_half(__funnelshift_r(wc[j0], wr[j0], 4u) & 0xffffu),
                    dec_half(__funnelshift_r(wc[j0], wr[j0], 0u) & 0xffffu)
                );
                const __half2 vA0 = __halves2half2
                (
                    dec_half(__funnelshift_r(wc[j1], wr[j1], 28u) & 0xffffu),
                    dec_half(__funnelshift_r(wc[j1], wr[j1], 24u) & 0xffffu)
                );
                const __half2 vA1 = __halves2half2
                (
                    dec_half(__funnelshift_r(wc[j1], wr[j1], 20u) & 0xffffu),
                    dec_half(__funnelshift_r(wc[j1], wr[j1], 16u) & 0xffffu)
                );
                const __half2 vB0 = __halves2half2
                (
                    dec_half(__funnelshift_r(wc[j1], wr[j1], 12u) & 0xffffu),
                    dec_half(__funnelshift_r(wc[j1], wr[j1], 8u) & 0xffffu)
                );
                const __half2 vB1 = __halves2half2
                (
                    dec_half(__funnelshift_r(wc[j1], wr[j1], 4u) & 0xffffu),
                    dec_half(__funnelshift_r(wc[j1], wr[j1], 0u) & 0xffffu)
                );
                // cross-pair emission (LLVM #186179): adjacent dot2 differ in
                // dst AND src0 AND src1 across the two subtiles of the pair
                a[j0][rh * 2 + 0] = __builtin_amdgcn_fdot2(wA0, xlo, a[j0][rh * 2 + 0], false);
                a[j1][rh * 2 + 1] = __builtin_amdgcn_fdot2(vA1, xhi, a[j1][rh * 2 + 1], false);
                a[j0][rh * 2 + 1] = __builtin_amdgcn_fdot2(wA1, xhi, a[j0][rh * 2 + 1], false);
                a[j1][rh * 2 + 0] = __builtin_amdgcn_fdot2(vA0, xlo, a[j1][rh * 2 + 0], false);
                a[j0][8 + rh * 2 + 0] = __builtin_amdgcn_fdot2(wB0, xlo, a[j0][8 + rh * 2 + 0], false);
                a[j1][8 + rh * 2 + 1] = __builtin_amdgcn_fdot2(vB1, xhi, a[j1][8 + rh * 2 + 1], false);
                a[j0][8 + rh * 2 + 1] = __builtin_amdgcn_fdot2(wB1, xhi, a[j0][8 + rh * 2 + 1], false);
                a[j1][8 + rh * 2 + 0] = __builtin_amdgcn_fdot2(vB0, xlo, a[j1][8 + rh * 2 + 0], false);
            }
        }
    }

    // explicit per-subtile stores (macro-free; v11 lesson)
    #pragma unroll
    for (int j = 0; j < SUBS; ++j)
    {
        float s0c = 0.0f, s1c = 0.0f;
        #pragma unroll
        for (int i = 0; i < 8; ++i) { s0c += a[j][i]; s1c += a[j][8 + i]; }
        const int base = nt0 + j;
        if (y_row_stride)
        {
            float* row = g_y + (size_t) blockIdx.y * y_row_stride;
            row[base * 16 + cA] = s0c;
            row[base * 16 + cA + 8] = s1c;
        }
        else if (splits == 1)
        {
            g_y[base * 16 + cA] = s0c;
            g_y[base * 16 + cA + 8] = s1c;
        }
        else
        {
            atomicAdd(g_y + base * 16 + cA, s0c);
            atomicAdd(g_y + base * 16 + cA + 8, s1c);
        }
    }
}

template <int SUBS>
__global__ __launch_bounds__(256) void gemv_vN
(
    const uint16_t* __restrict__ g_tr16,
    const __half* __restrict__ g_xh,
    float* __restrict__ g_y,
    int n_subs, int k_tiles, int splits, int y_row_stride
)
{
    gemv_vN_dev<SUBS>(g_tr16, g_xh, g_y, n_subs, k_tiles, splits, y_row_stride);
}


__device__ __forceinline__ uint32_t dp4a_u(unsigned x, unsigned y, unsigned c)
{
    return __builtin_amdgcn_udot4(x, y, c, false);
}

__device__ __forceinline__ uint32_t fshift_u64(uint32_t b, uint32_t a, uint32_t s)
{
    return static_cast<uint32_t>(((static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b)) >> s);
}

__device__ __forceinline__ uint32_t lut_h2(const uint32_t* lut, uint32_t s0, uint32_t s1)
{
    uint32_t a = lut[s0 - 0x6400u];
    uint32_t b = lut[s1 - 0x6400u];
    return __byte_perm(a, b, 0x5410);
}

// legacy warp-per-subtile kernel, bits=3 (dq8 align=4)
__device__ __forceinline__ void decode_lane_b3(const uint32_t* __restrict__ sub,
                                               uint32_t lane,
                                               uint32_t (&ws)[8])
{
    constexpr int W3 = 24;
    uint32_t idx = lane * 8;
    uint32_t b1 = (idx + 257) * 3;
    uint32_t b0 = b1 - 16;
    uint32_t b2 = b1 + 21;
    uint32_t i0 = b0 / 32;
    uint32_t i2 = (b2 - 1) / 32;
    uint32_t s2 = (i2 + 1) * 32 - b2;
    uint32_t a = sub[i0 % W3];
    uint32_t b = sub[i2 % W3];
    uint64_t m = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
    uint32_t w7 = static_cast<uint32_t>(m >> s2);
    uint32_t w5 = static_cast<uint32_t>(m >> (s2 + 6));
    uint32_t w3 = static_cast<uint32_t>(m >> (s2 + 12));
    uint32_t w1 = static_cast<uint32_t>(m >> (s2 + 18));
    ws[0] = (w1 >> 3) & 0xffffu;
    ws[1] = w1 & 0xffffu;
    ws[2] = (w3 >> 3) & 0xffffu;
    ws[3] = w3 & 0xffffu;
    ws[4] = (w5 >> 3) & 0xffffu;
    ws[5] = w5 & 0xffffu;
    ws[6] = (w7 >> 3) & 0xffffu;
    ws[7] = w7 & 0xffffu;
}

// ROOT CAUSE NOTE (the historical huge-N corruption, fixed):
// The even rh-slots' FIRST window is ((w[-1]:w[0]) >> 28) & 0xffff — a 16-bit field whose
// top 4 bits live in the PREVIOUS 32-bit word (w[-1], i.e. index (4*cA+31) % 32). The
// original implementation masked a 32-bit word BEFORE the cross-word shift, silently
// dropping those high bits for the even slots (odd slots happened to stay in-word).
// With ~50% of windows affected by a 3-4 bit error, errors appeared only when the
// corrupted weight products failed to cancel — hence "state-dependent" corruption that
// surfaced at particular trellis contents/large N. The fix (below): materialize the FULL
// 32-bit shifted word from the 64-bit (hi:lo) pair FIRST, then extract each 16-bit window
// from it. Verified: bit-flip oracle (every word x bit -> affected output elements
// consistent with reconstruct_slice) and 9-run multi-seed huge-N stress.
__global__ __launch_bounds__(256) void gemv_b3
(
    const uint16_t* __restrict__ g_tr16,
    const __half* __restrict__ g_xh,
    float* __restrict__ g_y,
    int n_subs, int k_tiles
)
{
    constexpr int W3 = 24;
    __shared__ uint32_t s_lut3[1024];
    {
        const int t = threadIdx.x;
        for (int i = t; i < 1024; i += 256)
        {
            __half h = __ushort_as_half(static_cast<uint16_t>(0x6400 + i));
            const __half k_inv = __ushort_as_half(0x1eee);
            const __half k_bias = __ushort_as_half(0xc931);
            __half v = __hfma(h, k_inv, k_bias);
            s_lut3[i] = __half_as_ushort(v) | (static_cast<uint32_t>(__half_as_ushort(v)) << 16);
        }
    }
    __syncthreads();

    const uint32_t* g_tr = reinterpret_cast<const uint32_t*>(g_tr16);
    const int t = threadIdx.x;
    const int lane = t % 32;
    const int warp = t / 32;
    const int nt = blockIdx.x * 8 + warp;
    if (nt >= n_subs) return;

    const int j = lane % 4;
    const int q = lane / 4;
    float acc0 = 0.0f, acc1 = 0.0f;
    const uint32_t* sub0 = g_tr + (int64_t)nt * W3;

    for (int kt = 0; kt < k_tiles; ++kt)
    {
        uint32_t wv[8];
        decode_lane_b3(sub0 + (int64_t)kt * n_subs * W3, lane, wv);
        uint32_t s8[8];
        #pragma unroll
        for (int k = 0; k < 8; ++k)
            s8[k] = dp4a_u(wv[k] * 0x83DCD12Du, 0x01010101u, 0x6400u);
        uint32_t w01 = lut_h2(s_lut3, s8[0], s8[1]);
        uint32_t w23 = lut_h2(s_lut3, s8[2], s8[3]);
        uint32_t w45 = lut_h2(s_lut3, s8[4], s8[5]);
        uint32_t w67 = lut_h2(s_lut3, s8[6], s8[7]);
        uint32_t xlo = *reinterpret_cast<const uint32_t*>(g_xh + kt * 16 + 2 * j);
        uint32_t xhi = *reinterpret_cast<const uint32_t*>(g_xh + kt * 16 + 8 + 2 * j);
        __half2 xl = __halves2half2(__ushort_as_half(xlo & 0xffff), __ushort_as_half(xlo >> 16));
        __half2 xh2 = __halves2half2(__ushort_as_half(xhi & 0xffff), __ushort_as_half(xhi >> 16));
        __half2 p01 = __hmul2(*reinterpret_cast<__half2*>(&w01), xl);
        __half2 p23 = __hmul2(*reinterpret_cast<__half2*>(&w23), xh2);
        __half2 p45 = __hmul2(*reinterpret_cast<__half2*>(&w45), xl);
        __half2 p67 = __hmul2(*reinterpret_cast<__half2*>(&w67), xh2);
        float s0 = __low2float(p01) + __high2float(p01) + __low2float(p23) + __high2float(p23);
        float s1 = __low2float(p45) + __high2float(p45) + __low2float(p67) + __high2float(p67);
        #pragma unroll
        for (int off = 2; off > 0; off >>= 1)
        {
            s0 += __shfl_down(s0, off, 4);
            s1 += __shfl_down(s1, off, 4);
        }
        if (j == 0) { acc0 += s0; acc1 += s1; }
    }

    if (j == 0)
    {
        g_y[nt * 16 + q] = acc0;
        g_y[nt * 16 + q + 8] = acc1;
    }
}

// ---------------- v5 kernels (bits=4 one-subtile reference, bits=6 production) ----------------

__device__ __forceinline__ float dec_val(uint32_t w, const __half2* lut2)
{
    uint32_t s = dp4a_u(w * 0x83DCD12Du, 0x01010101u, 0x6400u);
    __half2 v = lut2[s - 0x6400u];
    return __low2float(v);
}

__device__ __forceinline__ __half dec_half(uint32_t w)
{
    uint32_t s = dp4a_u(w * 0x83DCD12Du, 0x01010101u, 0x6400u);
    __half h = __ushort_as_half(static_cast<uint16_t>(s));
    return __hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}

template <int bits>
__device__ __forceinline__ void gemv_lut_body
(
    const uint16_t* __restrict__ g_tr16,
    const __half* __restrict__ g_xh,
    float* __restrict__ g_y,
    int n_subs, int k_tiles, int splits, int y_row_stride
)
{
    constexpr int W = bits * 8;
    __shared__ __half2 s_lut[1024];
    if constexpr (bits == 4)
    {
        const int t = threadIdx.x;
        for (int i = t; i < 1024; i += 256)
        {
            __half h = __ushort_as_half(static_cast<uint16_t>(0x6400 + i));
            const __half k_inv = __ushort_as_half(0x1eee);
            const __half k_bias = __ushort_as_half(0xc931);
            __half v = __hfma(h, k_inv, k_bias);
            s_lut[i] = __halves2half2(v, v);
        }
        __syncthreads();
    }

    const uint32_t* g_tr = reinterpret_cast<const uint32_t*>(g_tr16);
    const int t = threadIdx.x;
    const int nt = blockIdx.x * 32 + t / 8;
    if (nt >= n_subs) return;
    const int cA = t % 8;

    const int k_per = (k_tiles + splits - 1) / splits;
    const int kt0 = blockIdx.y * k_per;
    const int kt1 = min(kt0 + k_per, k_tiles);

    const uint32_t* sub0 = g_tr + (int64_t)nt * W;
    [[maybe_unused]] const __half2* lut2 = s_lut;

    // half2 SIMD accumulation, ROW-PAIRED: rows (2rh, 2rh+1) and
    // (2rh+8, 2rh+9) are natural x half2 pairs; lane = row parity, register = column
    // (cA for call 0, cA+8 for call 1); fp32 flush every 4 k-tiles
    __half2 hA = __floats2half2_rn(0.0f, 0.0f);
    __half2 hB = __floats2half2_rn(0.0f, 0.0f);
    float accA = 0.0f, accB = 0.0f;
    int since_flush = 0;

    if constexpr (bits == 4)
    {
        for (int kt = kt0; kt < kt1; ++kt)
        {
            const uint32_t* sub = sub0 + (int64_t)kt * n_subs * W;
            uint32_t w0 = sub[(4 * cA + 31) % 32];
            uint32_t w1 = sub[4 * cA];
            uint32_t w2 = sub[4 * cA + 1];
            uint32_t w3 = sub[4 * cA + 2];
            uint32_t w4 = sub[4 * cA + 3];
            uint64_t m01 = (static_cast<uint64_t>(w0) << 32) | w1;
            uint64_t m12 = (static_cast<uint64_t>(w1) << 32) | w2;
            uint64_t m23 = (static_cast<uint64_t>(w2) << 32) | w3;
            uint64_t m34 = (static_cast<uint64_t>(w3) << 32) | w4;
            const float4 xlo = *reinterpret_cast<const float4*>(g_xh + kt * 16);
            const float4 xhi = *reinterpret_cast<const float4*>(g_xh + kt * 16 + 8);
            const uint32_t xw[8] = {
                reinterpret_cast<const uint32_t&>(xlo),
                (&reinterpret_cast<const uint32_t&>(xlo))[1],
                (&reinterpret_cast<const uint32_t&>(xlo))[2],
                (&reinterpret_cast<const uint32_t&>(xlo))[3],
                reinterpret_cast<const uint32_t&>(xhi),
                (&reinterpret_cast<const uint32_t&>(xhi))[1],
                (&reinterpret_cast<const uint32_t&>(xhi))[2],
                (&reinterpret_cast<const uint32_t&>(xhi))[3],
            };
            float x[16];
            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                x[2 * i]     = __half2float(__ushort_as_half(static_cast<uint16_t>(xw[i] & 0xffff)));
                x[2 * i + 1] = __half2float(__ushort_as_half(static_cast<uint16_t>(xw[i] >> 16)));
            }
            #pragma unroll
            for (int rh = 0; rh < 4; ++rh)
            {
                uint32_t wc, wr;
                uint64_t mpair;
                switch (rh)
                {
                    case 0: wc = w1; wr = w0; mpair = m01; break;
                    case 1: wc = w2; wr = w1; mpair = m12; break;
                    case 2: wc = w3; wr = w2; mpair = m23; break;
                    default: wc = w4; wr = w3; mpair = m34; break;
                }
                (void) wr;
                uint32_t win[8] = {
                    static_cast<uint32_t>(mpair >> 28) & 0xffffu, static_cast<uint32_t>(mpair >> 24) & 0xffffu,
                    static_cast<uint32_t>(mpair >> 20) & 0xffffu, wc >> 16,
                    (wc >> 12) & 0xffffu, (wc >> 8) & 0xffffu, (wc >> 4) & 0xffffu, wc & 0xffffu,
                };
                #pragma unroll
                for (int k = 0; k < 8; ++k)
                {
                    float wv = dec_val(win[k], lut2);
                    int r = 8 * ((k % 4) / 2) + 2 * rh + (k % 2);
                    if (k / 4 == 0) accA += wv * x[r];
                    else            accB += wv * x[r];
                }
            }
        }
    }
    else if constexpr (bits == 6)
    {
        // layout empirically verified vs reconstruct_slice (b6_layout_solve.py):
        // element (r, c) = window wi of dq4 call (c/8) at lane L,
        //   L = 4*(c%8) + (r%8)/2,  wi = (r%2) + 2*(r/8),  t = L*8 + 4*(c/8)
        // dq4 window extraction (exl3_dq.cuh): b0 = (t+257)*6-16, b2 = b0+34,
        //   a = word[b0/32], b = word[(b2-1)/32] (a HIGH), w_{3-k} = ((a:b) >> (s2+6k)) & 0xffff
        for (int kt = kt0; kt < kt1; ++kt)
        {
            const uint32_t* sub = sub0 + (int64_t)kt * n_subs * W;
            // x tile as half pairs — the row-paired __hfma2 consumes halves directly;
            // NO f32 converts in the k-loop at all (verified in the shipped ISA: the
            // loop is 32 v_pk_fma_f16 + 32 v_dot4 + 32 v_mul_lo_u32, zero v_cvt)
            const __half2* xh2 = reinterpret_cast<const __half2*>(g_xh + kt * 16);

            // per rh (the lane-group index (r%8)/2 -> lane 4*cA + rh), two calls.
            // wi pairs (0,1) -> rows (2rh, 2rh+1) [xa]; (2,3) -> rows (2rh+8, 2rh+9)
            // [xb]. One __hfma2 per pair; call 0 accumulates column cA, call 1 cA+8.
            #pragma unroll
            for (int rh = 0; rh < 4; ++rh)
            {
                const int L = 4 * cA + rh;
                const __half2 xa = xh2[rh];
                const __half2 xb = xh2[4 + rh];
                uint32_t wq[2][4];
                #pragma unroll
                for (int call = 0; call < 2; ++call)
                {
                    const int t = L * 8 + 4 * call;
                    const int b0 = (t + 257) * 6 - 16;
                    const int b2 = b0 + 34;
                    const int i0 = b0 / 32;
                    const int i2 = (b2 - 1) / 32;
                    const int s2 = (i2 + 1) * 32 - b2;
                    const uint32_t a = sub[i0 % W];
                    const uint32_t b = sub[i2 % W];
                    const uint64_t m = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
                    wq[call][0] = static_cast<uint32_t>(m >> (s2 + 18)) & 0xffffu;
                    wq[call][1] = static_cast<uint32_t>(m >> (s2 + 12)) & 0xffffu;
                    wq[call][2] = static_cast<uint32_t>(m >> (s2 + 6)) & 0xffffu;
                    wq[call][3] = static_cast<uint32_t>(m >> s2) & 0xffffu;
                }
                const __half2 wA0 = __halves2half2(dec_half(wq[0][0]), dec_half(wq[0][1]));
                const __half2 wA1 = __halves2half2(dec_half(wq[0][2]), dec_half(wq[0][3]));
                const __half2 wB0 = __halves2half2(dec_half(wq[1][0]), dec_half(wq[1][1]));
                const __half2 wB1 = __halves2half2(dec_half(wq[1][2]), dec_half(wq[1][3]));
                hA = __hfma2(wA0, xa, hA);
                hA = __hfma2(wA1, xb, hA);
                hB = __hfma2(wB0, xa, hB);
                hB = __hfma2(wB1, xb, hB);
            }
            // fp32 flush every 4 k-tiles: packed convert + lane sum
            if (++since_flush == 4)
            {
                const float2 fA = __half22float2(hA);
                const float2 fB = __half22float2(hB);
                accA += fA.x + fA.y;
                accB += fB.x + fB.y;
                since_flush = 0;
                hA = __floats2half2_rn(0.0f, 0.0f);
                hB = __floats2half2_rn(0.0f, 0.0f);
            }
        }
        {
            const float2 fA = __half22float2(hA);
            const float2 fB = __half22float2(hB);
            accA += fA.x + fA.y;
            accB += fB.x + fB.y;
        }
    }

    if (y_row_stride)
    {
        float* row = g_y + (size_t) blockIdx.y * y_row_stride;
        row[nt * 16 + cA] = accA;
        row[nt * 16 + cA + 8] = accB;
    }
    else if (splits == 1)
    {
        g_y[nt * 16 + cA] = accA;
        g_y[nt * 16 + cA + 8] = accB;
    }
    else
    {
        atomicAdd(g_y + nt * 16 + cA, accA);
        atomicAdd(g_y + nt * 16 + cA + 8, accB);
    }
}

template <int bits>
__global__ __launch_bounds__(256) void gemv_lut
(
    const uint16_t* __restrict__ g_tr16,
    const __half* __restrict__ g_xh,
    float* __restrict__ g_y,
    int n_subs, int k_tiles, int splits, int y_row_stride
)
{
    gemv_lut_body<bits>(g_tr16, g_xh, g_y, n_subs, k_tiles, splits, y_row_stride);
}

// cb=2 decode WITHOUT the shared LUT: the byte-sum's fp16 bits feed a direct hfma
// (breaks the LDS-throughput wall that capped the LUT version at ~560 GB/s on b4)
__device__ __forceinline__ float dec_fast(uint32_t w)
{
    uint32_t s = dp4a_u(w * 0x83DCD12Du, 0x01010101u, 0x6400u);
    __half h = __ushort_as_half(static_cast<uint16_t>(s));
    return __half2float(__hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931)));
}

// ---------------- v6: b4, two adjacent subtiles per thread ----------------

// wide packed-half vector (8 halves = 4 __hfma2 lanes) with a __half2 union view:
// build/consume elementwise or pairwise without conversion ops (HIP has no __half16)
typedef _Float16 f16x8 __attribute__((ext_vector_type(8)));
union h8u { f16x8 v; __half2 h2[4]; };

// __half and _Float16 share the f16 bit format: union bit-cast, zero ops
__device__ __forceinline__ _Float16 h2f16(__half h)
{
    union { __half h; _Float16 f; } u;
    u.h = h;
    return u.f;
}





// rocWMMA (parsed on HIP only; the CUDA build never includes this header).
// TUs that also pull the torch/ATen chain must include rocwmma BEFORE it
// (exl3_gemm.cu does); kernel-table TUs have no ATen, so this direct include
// is clean there.
#if defined(__HIPCC__)
// rocWMMA under torch's __HIP_NO_HALF_{CONVERSIONS,OPERATORS}__ flags:
// - the __half vector registrations static_cast float->__half (deleted ctor) -> break
// - HIP_NO_HALF skips them, but this SDK build still references hfloat16_t in
//   wmma_impl's gfx11 derivative regardless of the enable gate -> also breaks
// Bridge: staged includes so the alias exists (float16_t is bit-identical to
// __half) before the rest of rocwmma parses. Must run BEFORE any torch/ATen
// header (the float8 trait glue also breaks after the torch chain).
#ifndef EXL3_ROCWMMA_BRIDGE
#define EXL3_ROCWMMA_BRIDGE
#define HIP_NO_HALF 1
#include <rocwmma/internal/config.hpp>
#include <rocwmma/internal/types.hpp>
namespace rocwmma { using hfloat16_t = float16_t; }
#include <rocwmma/rocwmma.hpp>
#undef HIP_NO_HALF
#endif
#endif

// ---------------- WMMA: M-blind GEMM, port of rocwmma perf_hgemm (gfx11) ----------------
// Follows AMD's samples/perf_hgemm.cpp structure - macro tile / warp tile /
// cooperative LDS staging with transposed B frags, apply_data_layout
// transforms, double-buffered k-step prefetch - adapted to the decode GEMV:
// the M dim is the batch (<= 16, zero-padded), the N dim is the output
// elements, and the B "global" source is the trellis decode into a
// row-major [k][n] scratch tile (the GEMV's verified window map), which then
// enters the sample's GRFragB -> LWFragB -> LRFragB -> mma pipeline unchanged.
// gfx11 tuning (sample table): 16x16x16 frags, wave32.

namespace
{

using namespace rocwmma;

constexpr uint32_t ROCWMMA_M = 16u, ROCWMMA_N = 16u, ROCWMMA_K = 16u;
constexpr uint32_t BLOCKS_M  = 1u,  BLOCKS_N  = 2u;   // M is the (tiny) batch dim
constexpr uint32_t TBLOCK_X  = 32u, TBLOCK_Y  = 4u;   // 4 independent wave32 warps per CTA
constexpr uint32_t WARP_SIZE = 32u;

using InputT   = float16_t;
using ComputeT = float32_t;

using DataLayoutA   = row_major;
using DataLayoutB   = row_major;   // scratch is [e][k] = W_recon[e][k]; row-major read: B[k][n] = scratch[k*16+n] = W_recon[k][n]
using DataLayoutC   = row_major;
using DataLayoutLds = col_major;

constexpr uint32_t WARP_TILE_M = BLOCKS_M * ROCWMMA_M;   // 16
constexpr uint32_t WARP_TILE_N = BLOCKS_N * ROCWMMA_N;   // 32

constexpr uint32_t WARPS_M      = TBLOCK_X / WARP_SIZE;  // 1
constexpr uint32_t WARPS_N      = TBLOCK_Y;              // 4
constexpr uint32_t MACRO_TILE_M = WARPS_M * WARP_TILE_M; // 16
constexpr uint32_t MACRO_TILE_N = WARPS_N * WARP_TILE_N; // 128

using MmaFragA   = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, row_major>;
using MmaFragB   = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, col_major>;
using MmaFragAcc = fragment<accumulator, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, ComputeT>;

using GRFragA = fragment<matrix_a, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutA>;
using GRFragB = fragment<matrix_b, ROCWMMA_M, ROCWMMA_N, ROCWMMA_K, InputT, DataLayoutB>;

using LWFragA = apply_data_layout_t<GRFragA, DataLayoutLds>;
using LWFragB = apply_data_layout_t<apply_transpose_t<GRFragB>, DataLayoutLds>;

ROCWMMA_DEVICE auto transformGRFragAToLWFragA(GRFragA const& f) { return apply_data_layout<DataLayoutLds>(f); }
ROCWMMA_DEVICE auto transformGRFragBToLWFragB(GRFragB const& f)
{
    return apply_data_layout<DataLayoutLds>(apply_transpose(f));
}

using LRFragA = apply_data_layout_t<MmaFragA, DataLayoutLds>;
using LRFragB = apply_data_layout_t<MmaFragB, DataLayoutLds>;

ROCWMMA_DEVICE auto transformLRFragAToMmaFragA(LRFragA const& f) { return apply_data_layout<row_major>(f); }
ROCWMMA_DEVICE auto transformLRFragBToMmaFragB(LRFragB const& f)
{
    return apply_data_layout<col_major>(apply_transpose(f));
}

}  // namespace

__device__ __forceinline__ unsigned f16_bits(__half v)
{
    union { __half h; unsigned short u; } c; c.h = v; return c.u;
}

// k-adjacent half2 pack from two funnel windows (low half = lower k)
// (hfma2-packed variant measured neutral-negative on gfx1100 - v_pk_fma_f16
// buys nothing over two scalar fmas here; kept scalar)
__device__ __forceinline__ uint32_t pack_win(uint32_t wc, uint32_t wr, unsigned shLo, unsigned shHi)
{
    return f16_bits(dec_half(__funnelshift_r(wc, wr, shLo) & 0xffffu)) |
           (f16_bits(dec_half(__funnelshift_r(wc, wr, shHi) & 0xffffu)) << 16);
}

// Per-bitcount trellis decoder for the WMMA GEMM. Contract: given the lane's
// octet index cA and its subtile/kt, emit uint32_t P[4][4] in the probed
// gfx11 matrix_b fragment layout (k-adjacent half-pairs per u32; elements
// {cA, cA+8} x k-half). The generic kernel body never mentions bits.
template <int Bits>
struct wmma_decoder;

template <>
struct wmma_decoder<4>
{
    static constexpr int words_per_row = 32;   // u32 trellis words per (kt, subtile)
    static constexpr int words_per_lane = 4;      // lane's word group = uint4

    __device__ static void init_lut(uint32_t*) {}   // no auxiliary state

    // post-mma affine correction: the fragment holds the RAW window bits as
    // f16 VALUES (h = 1024 + sum-of-nibbles), so the true weight
    // w = k_inv * h + k_bias is recovered linearly OUTSIDE the k-loop:
    // y_true = k_inv * acc + k_bias * sum(x). This deletes the udot4+fma
    // decode from the hot loop entirely.
    __device__ static float post_scale() { return __half2float(__ushort_as_half(0x1eee)); }
    __device__ static float post_bias()  { return __half2float(__ushort_as_half(0xc931)); }

    // lane-selective raw-window emission: only the xset this lane's fragment
    // rows need (xset 0..3 -> funnel-shift pairs (28,24) (20,16) (12,8) (4,0)).
    // Lane (o, cA) loads word 4*cA + o (each word loaded exactly once); the
    // group's other 3 words arrive via 3 lane shuffles.
    __device__ static void decode_lane(const uint32_t* g_tr, int lane, int xset,
                                       uint32_t (&fb)[4], uint32_t* = nullptr)
    {
        const int cA = lane & 7;
        const int o = lane >> 3;
        const uint32_t myw = g_tr[4 * cA + o];
        uint32_t svx = __shfl(myw, cA);
        uint32_t svy = __shfl(myw, cA + 8);
        uint32_t svz = __shfl(myw, cA + 16);
        uint32_t svw = __shfl(myw, cA + 24);
        const uint32_t wrap = __builtin_amdgcn_mov_dpp8(svw, 0xd63447u);
        const unsigned shLo = xset == 0 ? 28u : xset == 1 ? 20u : xset == 2 ? 12u : 4u;
        #pragma unroll
        for (int rh = 0; rh < 4; ++rh)
        {
            const uint32_t wc = rh == 0 ? svx : rh == 1 ? svy : rh == 2 ? svz : svw;
            const uint32_t wr = rh == 0 ? wrap : rh == 1 ? svx : rh == 2 ? svy : svz;
            fb[rh] = pack_win(wc, wr, shLo, shLo - 4u);
        }
    }
};

template <>
struct wmma_decoder<3>
{
    static constexpr int words_per_row = 24;   // u32 trellis words per (kt, subtile)
    static constexpr int words_per_lane = 3;

    __device__ static void init_lut(uint32_t* lut)
    {
        // 1024-entry byte-indexed LUT, halves duplicated (from gemv_b3)
        for (int i = threadIdx.x; i < 1024; i += WARP_SIZE)
        {
            __half hh = __ushort_as_half(static_cast<uint16_t>(0x6400 + i));
            const __half k_inv = __ushort_as_half(0x1eee);
            const __half k_bias = __ushort_as_half(0xc931);
            __half v = __hfma(hh, k_inv, k_bias);
            lut[i] = __half_as_ushort(v) | (static_cast<uint32_t>(__half_as_ushort(v)) << 16);
        }
    }

    // b3 algebra (from gemv_b3): the ORIGINAL lane q*4+j window yields
    //   w01 = (elem q,   k 2j / 2j+1)    w23 = (elem q,   k 8+2j / 9+2j)
    //   w45 = (elem q+8, k 2j / 2j+1)    w67 = (elem q+8, k 8+2j / 9+2j)
    // so P[s][rh] = {w01, w23, w45, w67} of the window at lane-index cA*4 + rh
    // lane-selective: window q*4+j (q = cA, j = rh) yields {w01,w23,w45,w67};
    // xset picks which of the four this lane emits
    __device__ static void decode_lane(const uint32_t* g_tr, int lane, int xset,
                                       uint32_t (&fb)[4], uint32_t* lut = nullptr)
    {
        const int cA = lane & 7;
        #pragma unroll
        for (int rh = 0; rh < 4; ++rh)
        {
            uint32_t wv[8];
            decode_lane_b3(g_tr, (uint32_t)(cA * 4 + rh), wv);
            uint32_t s8[8];
            #pragma unroll
            for (int k = 0; k < 8; ++k)
                s8[k] = dp4a_u(wv[k] * 0x83DCD12Du, 0x01010101u, 0x6400u);
            fb[rh] = lut_h2(lut, s8[2 * xset], s8[2 * xset + 1]);
        }
    }
};

template <>
struct wmma_decoder<6>
{
    static constexpr int words_per_row = 48;   // u32 trellis words per (kt, subtile)
    static constexpr int words_per_lane = 6;

    __device__ static void init_lut(uint32_t*) {}   // no auxiliary state

    // b6 algebra (empirically verified in gemv_lut<6>): element (r, c) comes
    // from window wi of dq4 call (c/8) at lane L = 4*(c%8) + (r%8)/2 with
    // wi = (r%2) + 2*(r/8); the dq4 gather is t = L*8 + 4*call, b0 = (t+257)*6-16.
    // The P sets are exactly the shipped half2 pairs:
    //   P[0][rh] = (elem cA,   k 2rh/2rh+1) P[1][rh] = (elem cA,   k 8+2rh/9+2rh)
    //   P[2][rh] = (elem cA+8, k 2rh/2rh+1) P[3][rh] = (elem cA+8, k 8+2rh/9+2rh)
    __device__ static void decode_lane(const uint32_t* g_tr, int lane, int xset,
                                       uint32_t (&fb)[4], uint32_t* = nullptr)
    {
        constexpr int W = 48;
        const int cA = lane & 7;
        const int call = xset >> 1;
        const int pair = xset & 1;
        #pragma unroll
        for (int rh = 0; rh < 4; ++rh)
        {
            const int L = 4 * cA + rh;
            const int t = L * 8 + 4 * call;
            const int b0 = (t + 257) * 6 - 16;
            const int b2 = b0 + 34;
            const int i0 = b0 / 32;
            const int i2 = (b2 - 1) / 32;
            const int s2 = (i2 + 1) * 32 - b2;
            const uint32_t a = g_tr[i0 % W];
            const uint32_t b = g_tr[i2 % W];
            const uint64_t m = (static_cast<uint64_t>(a) << 32) | static_cast<uint64_t>(b);
            const uint32_t wlo = static_cast<uint32_t>(m >> (s2 + 18 - 12 * pair)) & 0xffffu;
            const uint32_t whi = static_cast<uint32_t>(m >> (s2 + 12 - 12 * pair)) & 0xffffu;
            const __half2 wp = __halves2half2(dec_half(wlo), dec_half(whi));
            fb[rh] = *reinterpret_cast<const uint32_t*>(&wp);
        }
    }
};

template <int Bits>
__global__ __launch_bounds__(TBLOCK_X * TBLOCK_Y, 16) void gemm_wmma
(
    const uint16_t* __restrict__ g_tr16,
    const __half* __restrict__ g_A,
    float* __restrict__ g_y,
    int n_subs, int k_tiles, int splits, int y_row_stride, int size_m
)
{
    // Fragment-direct WMMA. gfx11 16x16x16 fragment register map (empirically
    // probed; identical for matrix_a row_major and matrix_b col_major): lane L
    // holds matrix row e = L & 15, columns k = 8 * (L >> 4) + [0..8), packed
    // as k-adjacent half pairs per u32 (low half = lower k). The bit-specific
    // decoder emits exactly that layout; the body below is bit-agnostic.
    using Dec = wmma_decoder<Bits>;
    constexpr int W = Dec::words_per_row;
    const uint32_t* g_tr = reinterpret_cast<const uint32_t*>(g_tr16);

    const int lane = threadIdx.x;
    const int warp = threadIdx.y;
    const int cA = lane & 7;

    // warp-private subtile band (warps never share data - no cross-warp sync)
    const int n0 = blockIdx.x * int(MACRO_TILE_N) + warp * int(WARP_TILE_N);
    const int k_per = (k_tiles + splits - 1) / splits;
    const int kt0 = blockIdx.y * k_per;
    const int kt1 = min(kt0 + k_per, k_tiles);

    __shared__ alignas(16) uint32_t dec_scratch[1024];   // decoder aux (b3 LUT)

    fragment<matrix_a, 16, 16, 16, InputT, row_major> aFrag;
    fragment<matrix_b, 16, 16, 16, InputT, col_major> bFrag[BLOCKS_N];
    MmaFragAcc acc[BLOCKS_N];
    #pragma unroll
    for (int b = 0; b < BLOCKS_N; ++b) fill_fragment(acc[b], 0.0f);

    const uint32_t* gx = reinterpret_cast<const uint32_t*>(g_A);
    // lane->fragment-set table: rows e = L&15, k-half = L>>4 selects P-set
    const int xset = (lane < 8) ? 0 : (lane < 16) ? 2 : (lane < 24) ? 1 : 3;

    // M-generic A fragment: lane L owns batch row e = L & 15 at k-offset
    // 8 * (L >> 4) (probed A layout). Rows m >= size_m are dead - zeroed
    // ONCE here; live rows load one uint4 (k-adjacent pairs) per k-tile.
    const int arow = lane & 15;
    const uint32_t* arow_base = gx + (int64_t)arow * k_tiles * 8;   // hoisted row base
    uint32_t* fa = static_cast<uint32_t*>(static_cast<void*>(&aFrag.x));
    if (arow >= size_m)
    {
        fa[0] = 0u; fa[1] = 0u; fa[2] = 0u; fa[3] = 0u;
    }

    Dec::init_lut(dec_scratch);
    __syncthreads();

    for (int kt = kt0; kt < kt1; ++kt)
    {
        if (arow < size_m)
        {
            const uint32_t* xp = arow_base + kt * 8 + 4 * (lane >> 4);
            fa[0] = xp[0]; fa[1] = xp[1]; fa[2] = xp[2]; fa[3] = xp[3];
        }

        // lane-local fragment build: each lane decodes ONLY the set its rows
        // need (4 live u32 per subtile - no exchange, no barriers, no spills)
        #pragma unroll
        for (int b = 0; b < BLOCKS_N; ++b)
        {
            uint32_t fbv[4] = {0u, 0u, 0u, 0u};
            const int nt = (n0 + b * 16) / 16;
            if (nt < n_subs)
            {
                const uint32_t* sub = g_tr + (int64_t)kt * n_subs * W + (int64_t)nt * W;
                Dec::decode_lane(sub, lane, xset, fbv, dec_scratch);
            }
            uint32_t* fb = static_cast<uint32_t*>(static_cast<void*>(&bFrag[b].x));
            fb[0] = fbv[0]; fb[1] = fbv[1]; fb[2] = fbv[2]; fb[3] = fbv[3];
            mma_sync(acc[b], aFrag, bFrag[b], acc[b]);
        }
    }

    // epilogue: direct accumulator store. Probed gfx11 acc layout: reg i of
    // lane L holds C[m = 2*i + (L >> 4)][n = L & 15] - one scalar store per
    // live row (m < size_m), consecutive n across lanes (coalesced). No sC,
    // no store_matrix_sync, no barriers.
    {
        #pragma unroll
        for (int b = 0; b < BLOCKS_N; ++b)
        {
            const int nt = (n0 + b * 16) / 16;
            if (nt >= n_subs) continue;
            const float* fr = static_cast<const float*>(static_cast<const void*>(&acc[b].x));
            const int ncol = lane & 15;
            const int m0 = lane >> 4;
            #pragma unroll
            for (int i = 0; i < 8; ++i)
            {
                const int m = 2 * i + m0;
                if (m >= size_m) break;
                const float v = fr[i];
                const int elem = nt * 16 + ncol;
                if (y_row_stride)
                {
                    float* rowp = g_y + (size_t) blockIdx.y * y_row_stride;
                    rowp[m * (n_subs * 16) + elem] = v;
                }
                else if (splits == 1)
                {
                    g_y[m * (n_subs * 16) + elem] = v;
                }
                else
                {
                    atomicAdd(g_y + m * (n_subs * 16) + elem, v);
                }
            }
        }
    }
}

static inline int m1_pick_splits(int bits, int n_subs, int k_tiles)
{
    int splits = 1;
    {
        const int blocks_x0 = (n_subs + (bits == 4 ? 63 : 31)) / (bits == 4 ? 64 : 32);
        if (blocks_x0 >= 64) splits = 4;
        else while (splits < 128 && blocks_x0 * splits < 640) splits <<= 1;
        if (splits > k_tiles) splits = k_tiles;
    }
    return splits;
}

static inline bool gemm_wmma_launch
(
    hipStream_t stream,
    const uint16_t* p_tr,
    const __half* p_A,
    float* p_y,
    int bits, int n_subs, int k_tiles, int size_m
)
{
    int splits = m1_pick_splits(bits, n_subs, k_tiles);
    // row-store split-K on the wide class (splits == 4), same policy as the
    // legacy path: no memset, no atomics; the caller's (splits x N) scratch is
    // summed by the fused epilogue (m1_had_out_f32_to_h16, production)
    const int row_stride = 0;   // atomic accumulate (row-store epilogue under debug)
    if (splits > 1)
        hipMemsetAsync(p_y, 0, (size_t)n_subs * 16 * 4 * size_m, stream);
    const int lds_bytes = 0;
    dim3 grid((n_subs * 16 + int(MACRO_TILE_N) - 1) / int(MACRO_TILE_N), splits);
    switch (bits)
    {
        case 3:
            hipLaunchKernelGGL((gemm_wmma<3>), grid, dim3(32, 4), lds_bytes, stream,
                               p_tr, p_A, p_y, n_subs, k_tiles, splits, row_stride, size_m);
            return true;
        case 4:
            hipLaunchKernelGGL((gemm_wmma<4>), grid, dim3(32, 4), lds_bytes, stream,
                               p_tr, p_A, p_y, n_subs, k_tiles, splits, row_stride, size_m);
            return true;
        case 6:
            hipLaunchKernelGGL((gemm_wmma<6>), grid, dim3(32, 4), lds_bytes, stream,
                               p_tr, p_A, p_y, n_subs, k_tiles, splits, row_stride, size_m);
            return true;
        default:
            return false;
    }
    return false;
}

void launch
(
    hipStream_t stream,
    const uint16_t* p_tr,
    const __half* p_x,
    float* p_y,
    int bits,
    int n_subs,
    int k_tiles
)
{
    if (bits == 3)
    {
        const int blocks3 = (n_subs + 7) / 8;
        hipLaunchKernelGGL(gemv_b3, dim3(blocks3), dim3(256), 0, stream,
                           p_tr, p_x, p_y, n_subs, k_tiles);
        return;
    }

    
// split-K policy: fill the machine (aim for >=1024 CTAs), capped by k_tiles
    // split-K policy: fill the machine (>=1024 CTAs) — narrow-N shapes (down_proj:
    // 5 x-blocks) need up to 128 splits (measured optimum), wide-N shapes hit the
    // target by 4-8; cap by k_tiles
    int splits = m1_pick_splits(bits, n_subs, k_tiles);

    // wide shapes (splits <= 4) use ROW-STORE partials: each split writes its own
    // row of a (splits x N) scratch (caller-provided when splits > 1) and the output
    // epilogue sums the rows — no memset, no atomics. Deep splits (narrow shapes)
    // keep the zero+atomicAdd tail, where an epilogue-side sum would read too deep.
    const int row_stride = (splits == 4) ? n_subs * 16 : 0;
    if (row_stride == 0 && splits > 1)
        hipMemsetAsync(p_y, 0, (size_t)n_subs * 16 * 4, stream);

    if (bits == 4)
        hipLaunchKernelGGL((gemv_vN<2>), dim3((n_subs + 32 * 2 - 1) / (32 * 2), splits), dim3(256), 0, stream,
                           p_tr, p_x, p_y, n_subs, k_tiles, splits, row_stride);
    else
        hipLaunchKernelGGL(gemv_lut<6>, dim3((n_subs + 31) / 32, splits), dim3(256), 0, stream,
                           p_tr, p_x, p_y, n_subs, k_tiles, splits, row_stride);
}

// sum (n_rows x N) fp32 partials into y (bench shim path; production fuses this
// into m1_had_out_f32_to_h16)
__global__ void m1_reduce_rows(const float* __restrict__ partials, float* __restrict__ y,
                               int64_t n4, int row_stride4, int n_rows)
{
    // float4-vectorized: n4/row_stride4 are in float4 units
    int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n4) return;
    float4 acc = ((const float4*) partials)[i];
    for (int r = 1; r < n_rows; ++r)
    {
        const float4 v = *((const float4*) (partials + (int64_t) r * row_stride4) + i);
        acc.x += v.x; acc.y += v.y; acc.z += v.z; acc.w += v.w;
    }
    ((float4*) y)[i] = acc;
}


// split-K row stride the launch() policy will use for this shape (0 = atomic mode);
// callers use it to size the (splits x N) partials scratch
inline int m1_row_stride(int bits, int n_subs, int k_tiles)
{
    int splits = m1_pick_splits(bits, n_subs, k_tiles);
    return (splits == 4) ? n_subs * 16 : 0;
}

// shared raw entry for the bench shim; DEFINED in exl3_gemm.cu (the single guarded TU)
int m1_entry(const at::Tensor& B, const at::Tensor& x, at::Tensor& y);

// Fused output epilogue for the M=1 path: 128-hadamard with svh post-scale on the fp32
// GEMV result, stored DIRECTLY as half — identical arithmetic to had_ff_r_128_inner
// followed by an RN cast (the old path: in-place fp32 had kernel + separate copy_), one
// kernel and one less full-N pass instead of two.

__global__ __launch_bounds__(32) void m1_had_out_f32_to_h16
(
    const float* __restrict__ input_ptr,
    half* __restrict__ output_ptr,
    const half* __restrict__ scale,
    const float r_scale,
    const int row_stride,    // > 0: input is (n_rows x N) partials to sum first
    const int n_rows
)
{
    const int t = threadIdx.x & 31;
    const size_t off = (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128 + t * 4;
    output_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;

    float4 v;
    if (row_stride > 0)
    {
        const float4* p0 = (const float4*) (input_ptr + off);
        const float4* p1 = (const float4*) (input_ptr + off + row_stride);
        const float4 a = p0[0];
        const float4 b = p1[0];
        v.x = a.x + b.x; v.y = a.y + b.y; v.z = a.z + b.z; v.w = a.w + b.w;
        if (n_rows == 4)  // two more partial rows
        {
            const float4* p2 = (const float4*) (input_ptr + off + 2 * (size_t) row_stride);
            const float4* p3 = (const float4*) (input_ptr + off + 3 * (size_t) row_stride);
            const float4 c = p2[0];
            const float4 d = p3[0];
            v.x += c.x + d.x; v.y += c.y + d.y; v.z += c.z + d.z; v.w += c.w + d.w;
        }
    }
    else
    {
        input_ptr += (size_t) gridDim.y * 128 * blockIdx.x + blockIdx.y * 128;
        v = ((const float4*) input_ptr)[t];
    }

    float v0 = v.x, v1 = v.y, v2 = v.z, v3 = v.w;
    float s0 = v0 + v1, d0 = v0 - v1;
    float s1 = v2 + v3, d1 = v2 - v3;
    v.x = s0 + s1;
    v.y = d0 + d1;
    v.z = s0 - s1;
    v.w = d0 - d1;

    shuffle_had_f2x32(v.x, v.y, t);
    shuffle_had_f2x32(v.z, v.w, t);
    v.x *= r_scale;
    v.y *= r_scale;
    v.z *= r_scale;
    v.w *= r_scale;

    // post scale (svh), same order as had_ff_r_128_inner<false, true>
    {
        const int i = blockIdx.y * 32 + t;
        half4 scales = ((const half4*) scale)[i];
        v.x *= __low2float(scales.x);
        v.y *= __high2float(scales.x);
        v.z *= __low2float(scales.y);
        v.w *= __high2float(scales.y);
    }

    ((half2*) output_ptr)[t * 2]     = __floats2half2_rn(v.x, v.y);
    ((half2*) output_ptr)[t * 2 + 1] = __floats2half2_rn(v.z, v.w);
}

// ---------------- full M=1 wrapper + default-on dispatch decision ----------------
// The tensor-level gr wrapper (had_in + GEMV + in-place fp32 had_out + cast copy with
// persistent scratch) lives in exl3_gemm.cu: torch's hipify drops non-mapped
// ATen includes from .cuh twins, so factories cannot live here. The shim installs the
// implementation into this pointer at static init.
int m1_gr_default(const at::Tensor&, const at::Tensor&, at::Tensor&,
                  const c10::optional<at::Tensor>&,
                  const c10::optional<at::Tensor>&,
                  const c10::optional<at::Tensor>&);

inline int (*gemv_m1_gr)(const at::Tensor&, const at::Tensor&, at::Tensor&,
                          const c10::optional<at::Tensor>&,
                          const c10::optional<at::Tensor>&,
                          const c10::optional<at::Tensor>&) = &m1_gr_default;

// returns true if the M=1 fast path was taken (and the GEMM is complete)
static inline bool try_m1
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int size_m,
    int size_k,
    int size_n,
    bool mul1,
    bool mcg,
    bool graph
)
{
    constexpr bool enabled = true;  // unconditional: the M=1 GEMV is the ROCm path
    const int bits = (int)(B.size(2) / 16);
    if (!enabled || size_m != 1 || !mul1 || mcg ||
        !(bits == 3 || bits == 4 || bits == 6) ||
        (size_k % 16) || (size_n % 16))
        return false;
    if (graph) return false;  // ext-graph capture mode keeps the graph-parameterized GEMM
    // Warm the lock-cascade GEMM's autotuner ONCE per shape while eager: at capture time
    // the GEMM path (graph mode) must not hit a cold autotune, which syncs mid-capture
    static std::set<std::tuple<int, int64_t, int64_t, int>> warmed;
    auto key = std::make_tuple(A.device().index(), size_k, size_n, bits);
    if (warmed.insert(key).second)
        return false;  // first eager call of this shape: fall through to the GEMM
    gemv_m1_gr(A, B, C, suh, A_had, svh);
    return true;
}
}  // namespace exl3_rocm_gemv

#endif  // EXL3_ROCM_GEMV_M1
