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

// vN: configurable amortized GEMV. SUBS = subtiles per thread (multiple of
// 2); each thread computes a 16 x (16*SUBS) output tile: SUBS*16 fp32 chains,
// SUBS*16 fdot2 per k-tile, x pairs and decode constants amortized SUBS/2x
// more than v11. Cross-pair emission order per LLVM #186179; explicit stores.
__device__ __forceinline__ uint32_t dp4a_u(unsigned x, unsigned y, unsigned c)
{
    return __builtin_amdgcn_udot4(x, y, c, false);
}

__device__ __forceinline__ uint32_t fshift_u64(uint32_t b, uint32_t a, uint32_t s)
{
    return static_cast<uint32_t>((static_cast<uint64_t>(a) << 32 | b) >> s);
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
__device__ __forceinline__ __half dec_half(uint32_t v);

__device__ __forceinline__ __half dec_half(uint32_t w)
{
    uint32_t s = dp4a_u(w * 0x83DCD12Du, 0x01010101u, 0x6400u);
    __half h = __ushort_as_half(static_cast<uint16_t>(s));
    return __hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}

#if defined(__HIPCC__)
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

namespace exl3_rocm_gemv
{
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

    // the fragment holds the RAW window bits as f16 VALUES (h = 1024 +
    // sum-of-nibbles); dec_half's hfma already applies the affine
    // (h * k_inv + k_bias) per element at decode time

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

}   // (P36) anonymous namespace: unfused bench kernels deleted

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

template <int bits, int cb>
__device__ __forceinline__ void phase1_dq(SegCtx& c)
{
    if constexpr (cb == 2 && (bits == 3 || bits == 4 || bits == 6))
    {
        // P43: tensor-core mma core for the ROCm inner (mul1 semantics — the
        // b4 fragment holds raw windows; the affine correction is applied at
        // the store, linearly outside the k-loop). Same SegCtx contract, same
        // callers, same lock-cascade/epilogue as the scalar path below.
        using namespace exl3_rocm_gemv;
        using Dec = wmma_decoder<bits>;
        constexpr int W = Dec::words_per_row;
        constexpr int W1 = exl3_rocm_gemv::BLOCKS_N;

        const int lane = threadIdx.x & 31;
        const int warp = threadIdx.x >> 5;
        const int warps = blockDim.x >> 5;
        const int xset = (lane < 8) ? 0 : (lane < 16) ? 2 : (lane < 24) ? 1 : 3;
        const int arow = lane & 15;

        uint32_t* dec_lut = nullptr;   // b3 decode LUT (b4/b6 need none)
        if constexpr (bits == 3)
        {
            __shared__ uint32_t dec_lut_sh[1024];
            dec_lut = dec_lut_sh;
            Dec::init_lut(dec_lut);
            __syncthreads();
        }

        // P46: runtime warp tile from the instance's actual width: 2 subtiles
        // per warp when the tile has >= 2 subtiles per warp (SHAPE_5's 16
        // subtiles / 8 warps), else 1. phase1_dq is not a template on the
        // tile sizes, so the width arrives via SegCtx (subs_tile).
        const int wt = 1;
        constexpr int WMAX = 2;
        for (int idx0 = warp * wt; idx0 < c.subs_tile; idx0 += warps * wt)
        {
            fragment<matrix_a, 16, 16, 16, InputT, row_major> aFrag;
            fragment<matrix_b, 16, 16, 16, InputT, col_major> bFrag[WMAX];
            MmaFragAcc acc[WMAX];
            #pragma unroll
            for (int b = 0; b < wt; ++b) fill_fragment(acc[b], 0.0f);

            uint32_t* fa = static_cast<uint32_t*>(static_cast<void*>(&aFrag.x));
            if (arow >= c.size_m) { fa[0] = 0u; fa[1] = 0u; fa[2] = 0u; fa[3] = 0u; }

            const uint32_t* gx = reinterpret_cast<const uint32_t*>(c.A);
            const uint32_t* arow_base = gx + (int64_t)arow * (c.size_k / 16) * 8;

            // P47: 2-kt software pipeline - both k-tiles' trellis words issue
            // back-to-back (2x bytes in flight per lane) before either decode
            // chain consumes them; targets the measured 128B/warp/kt memory
            // parallelism gap vs tier A's 512B.
            constexpr int KSTEP = 2;   // P47: 2 trellis words in flight per lane (best measured)
            int t = c.kt0;
            for (; t + KSTEP - 1 < c.kt1; t += KSTEP)
            {
                fragment<matrix_a, 16, 16, 16, InputT, row_major> aFrags[KSTEP];
                #pragma unroll
                for (int kk = 0; kk < KSTEP; ++kk)
                {
                    uint32_t* fak = static_cast<uint32_t*>(static_cast<void*>(&aFrags[kk].x));
                    if (arow < c.size_m)
                    {
                        const uint32_t* xpk = arow_base + (t + kk) * 8 + 4 * (lane >> 4);
                        fak[0] = xpk[0]; fak[1] = xpk[1]; fak[2] = xpk[2]; fak[3] = xpk[3];
                    }
                    else { fak[0] = 0u; fak[1] = 0u; fak[2] = 0u; fak[3] = 0u; }
                }
                #pragma unroll
                for (int kk = 0; kk < KSTEP; ++kk)
                {
                    #pragma unroll
                    for (int b = 0; b < WMAX; ++b)
                    {
                        if (b >= wt) break;
                        uint32_t fbv[4] = {0u, 0u, 0u, 0u};
                        const int s = idx0 + b;
                        if (s < c.subs_tile)
                        {
                            const int n = c.col * c.subs_tile + s;
                            const uint32_t* sub = c.B + (int64_t)(t + kk) * (c.nsub_total * W) + (int64_t)n * W;
                            Dec::decode_lane(sub, lane, xset, fbv, dec_lut);
                        }
                        uint32_t* fb = static_cast<uint32_t*>(static_cast<void*>(&bFrag[b].x));
                        fb[0] = fbv[0]; fb[1] = fbv[1]; fb[2] = fbv[2]; fb[3] = fbv[3];
                        mma_sync(acc[b], aFrags[kk], bFrag[b], acc[b]);
                    }
                }
            }
            for (; t < c.kt1; ++t)
            {
                if (arow < c.size_m)
                {
                    const uint32_t* xp = arow_base + t * 8 + 4 * (lane >> 4);
                    fa[0] = xp[0]; fa[1] = xp[1]; fa[2] = xp[2]; fa[3] = xp[3];
                }
                else { fa[0] = 0u; fa[1] = 0u; fa[2] = 0u; fa[3] = 0u; }
                #pragma unroll
                for (int b = 0; b < WMAX; ++b)
                {
                    if (b >= wt) break;
                    uint32_t fbv[4] = {0u, 0u, 0u, 0u};
                    const int s = idx0 + b;
                    if (s < c.subs_tile)
                    {
                        const int n = c.col * c.subs_tile + s;
                        const uint32_t* sub = c.B + (int64_t)t * (c.nsub_total * W) + (int64_t)n * W;
                        Dec::decode_lane(sub, lane, xset, fbv, dec_lut);
                    }
                    uint32_t* fb = static_cast<uint32_t*>(static_cast<void*>(&bFrag[b].x));
                    fb[0] = fbv[0]; fb[1] = fbv[1]; fb[2] = fbv[2]; fb[3] = fbv[3];
                    mma_sync(acc[b], aFrag, bFrag[b], acc[b]);
                }
            }

            #pragma unroll
            for (int b = 0; b < WMAX; ++b)
            {
                if (b >= wt) break;
                const int s = idx0 + b;
                if (s >= c.subs_tile) continue;
                const float* fr = static_cast<const float*>(static_cast<const void*>(&acc[b].x));
                const int ncol = lane & 15;
                const int m0 = lane >> 4;
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                {
                    const int m = 2 * i + m0;
                    if (m >= c.size_m) break;
                    c.sh_c[(size_t) m * (c.subs_tile * 16) + (size_t) s * 16 + ncol] = fr[i];
                }
            }
        }
    }
    else
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

