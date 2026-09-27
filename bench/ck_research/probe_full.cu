#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

// EXACT transplant of the v9 kernel stanza shape: V9PK + V9_RH macros copied
// from the header (duals removed, packs consumed via fdot2 like the passing
// discriminator build), 4 stanzas around a k-loop, 32 pinned accumulators,
// uint4-style word arrays + wrap words, xlo/xhi pairs. Compares final chains
// against a pure-C reference chain accumulation.

#define V9PK(PK, C, R, SHH, SHL) \
    "v_alignbit_b32 %[ta], %[" #R "], %[" #C "], " #SHH "\n" \
    "v_and_b32_e32 %[ta], 0xffff, %[ta]\n" \
    "v_mul_lo_u32 %[ta], %[ta], %[kmul]\n" \
    "v_dot4_u32_u8 %[ta], %[ta], 0x01010101, 0\n" \
    "v_alignbit_b32 %[tb], %[" #R "], %[" #C "], " #SHL "\n" \
    "v_and_b32_e32 %[tb], 0xffff, %[tb]\n" \
    "v_mul_lo_u32 %[tb], %[tb], %[kmul]\n" \
    "v_dot4_u32_u8 %[tb], %[tb], 0x01010101, 0\n" \
    "v_perm_b32 %[" #PK "], %[tb], %[ta], 0x05040100\n" \
    "v_add_u32 %[" #PK "], %[" #PK "], 0x64006400\n" \
    "v_pk_fma_f16 %[" #PK "], %[" #PK "], %[kinv], %[kbias]\n"

__device__ __forceinline__ __half dec_half_ref(uint32_t w)
{
    uint32_t prod = w * 0x83DCD12Du;
    uint32_t s = 0x6400u + (prod & 0xffu) + ((prod >> 8) & 0xffu)
               + ((prod >> 16) & 0xffu) + ((prod >> 24) & 0xffu);
    __half h = __ushort_as_half((uint16_t)s);
    return __hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}

// stanza identical to V9_RH but consuming via fdot2 (the build that PASSED
// when packs were C-computed and FAILED with asm packs, in the real kernel)
#define STZ(RH, WA0, WA1, WA2, WA3, WB0, WB1, WB2, WB3) \
    { \
        const uint32_t wc = w0v[RH], wr = w1v[RH]; \
        const uint32_t vc = u0v[RH], vr = u1v[RH]; \
        const __half2 xlo = xh2[RH]; \
        const __half2 xhi = xh2[RH + 4]; \
        uint32_t p0A0, p0A1, p0B0, p0B1, p1A0, p1A1, p1B0, p1B1, ta, tb; \
        __asm volatile \
        ( \
            V9PK(p0A0, wc, wr, 28, 24) \
            V9PK(p0A1, wc, wr, 20, 16) \
            V9PK(p0B0, wc, wr, 12, 8) \
            V9PK(p0B1, wc, wr, 4, 0) \
            V9PK(p1A0, vc, vr, 28, 24) \
            V9PK(p1A1, vc, vr, 20, 16) \
            V9PK(p1B0, vc, vr, 12, 8) \
            V9PK(p1B1, vc, vr, 4, 0) \
            : [p0A0] "=&v"(p0A0), [p0A1] "=&v"(p0A1), \
              [p0B0] "=&v"(p0B0), [p0B1] "=&v"(p0B1), \
              [p1A0] "=&v"(p1A0), [p1A1] "=&v"(p1A1), \
              [p1B0] "=&v"(p1B0), [p1B1] "=&v"(p1B1), \
              [ta] "=&v"(ta), [tb] "=&v"(tb) \
            : [wc] "r"(wc), [wr] "r"(wr), [vc] "r"(vc), [vr] "r"(vr), \
              [xlo] "v"(xlo), [xhi] "v"(xhi), \
              [kmul] "r"(kmul), [kinv] "v"(kinv), [kbias] "v"(kbias) \
        ); \
        const __half2 hA0 = *reinterpret_cast<__half2*>(&p0A0); \
        const __half2 hA1 = *reinterpret_cast<__half2*>(&p0A1); \
        const __half2 hB0 = *reinterpret_cast<__half2*>(&p0B0); \
        const __half2 hB1 = *reinterpret_cast<__half2*>(&p0B1); \
        const __half2 gA0 = *reinterpret_cast<__half2*>(&p1A0); \
        const __half2 gA1 = *reinterpret_cast<__half2*>(&p1A1); \
        const __half2 gB0 = *reinterpret_cast<__half2*>(&p1B0); \
        const __half2 gB1 = *reinterpret_cast<__half2*>(&p1B1); \
        WA0 = __builtin_amdgcn_fdot2(hA0, xlo, WA0, false); \
        WA1 = __builtin_amdgcn_fdot2(gA0, xlo, WA1, false); \
        WA2 = __builtin_amdgcn_fdot2(hA1, xhi, WA2, false); \
        WA3 = __builtin_amdgcn_fdot2(gA1, xhi, WA3, false); \
        WB0 = __builtin_amdgcn_fdot2(hB0, xlo, WB0, false); \
        WB1 = __builtin_amdgcn_fdot2(gB0, xlo, WB1, false); \
        WB2 = __builtin_amdgcn_fdot2(hB1, xhi, WB2, false); \
        WB3 = __builtin_amdgcn_fdot2(gB1, xhi, WB3, false); \
    }

__global__ void probe
(
    const uint32_t* __restrict__ subwords,   // [kt][2][32]: sub0,sub1 words
    const uint32_t* __restrict__ wraps,      // [kt][2]
    const __half2* __restrict__ xtiles,      // [kt][8]
    float* __restrict__ y, float* __restrict__ ref, int n_kt
)
{
    const uint32_t kmul = 0x83DCD12Du;
    const __half2 kinv = __half2half2(__ushort_as_half(0x1eee));
    const __half2 kbias = __half2half2(__ushort_as_half(0xc931));

    register float a0_0 __asm__("v0") = 0.0f;   register float a1_0 __asm__("v1") = 0.0f;
    register float a0_1 __asm__("v2") = 0.0f;   register float a1_1 __asm__("v3") = 0.0f;
    register float a0_2 __asm__("v4") = 0.0f;   register float a1_2 __asm__("v5") = 0.0f;
    register float a0_3 __asm__("v6") = 0.0f;   register float a1_3 __asm__("v7") = 0.0f;
    register float a0_4 __asm__("v8") = 0.0f;   register float a1_4 __asm__("v9") = 0.0f;
    register float a0_5 __asm__("v10") = 0.0f;  register float a1_5 __asm__("v11") = 0.0f;
    register float a0_6 __asm__("v12") = 0.0f;  register float a1_6 __asm__("v13") = 0.0f;
    register float a0_7 __asm__("v14") = 0.0f;  register float a1_7 __asm__("v15") = 0.0f;
    register float a0_8 __asm__("v16") = 0.0f;  register float a1_8 __asm__("v17") = 0.0f;
    register float a0_9 __asm__("v18") = 0.0f;  register float a1_9 __asm__("v19") = 0.0f;
    register float a0_10 __asm__("v20") = 0.0f; register float a1_10 __asm__("v21") = 0.0f;
    register float a0_11 __asm__("v22") = 0.0f; register float a1_11 __asm__("v23") = 0.0f;
    register float a0_12 __asm__("v24") = 0.0f; register float a1_12 __asm__("v25") = 0.0f;
    register float a0_13 __asm__("v26") = 0.0f; register float a1_13 __asm__("v27") = 0.0f;
    register float a0_14 __asm__("v28") = 0.0f; register float a1_14 __asm__("v29") = 0.0f;
    register float a0_15 __asm__("v30") = 0.0f; register float a1_15 __asm__("v31") = 0.0f;

    // C reference accumulators (identical chain structure, pure C packs)
    float r0[16], r1[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) { r0[i] = 0.0f; r1[i] = 0.0f; }

    for (int kt = 0; kt < n_kt; ++kt)
    {
        const uint32_t* s0w = subwords + kt * 64;
        const uint32_t* s1w = s0w + 32;
        const uint32_t w0v[4] = { s0w[0], s0w[1], s0w[2], s0w[3] };
        const uint32_t w1v[4] = { wraps[kt * 2 + 0], s0w[0], s0w[1], s0w[2] };
        const uint32_t u0v[4] = { s1w[0], s1w[1], s1w[2], s1w[3] };
        const uint32_t u1v[4] = { wraps[kt * 2 + 1], s1w[0], s1w[1], s1w[2] };
        const __half2* xh2 = xtiles + kt * 8;

        STZ(0, a0_0, a1_0, a0_1, a1_1, a0_8,  a1_8,  a0_9,  a1_9)
        STZ(1, a0_2, a1_2, a0_3, a1_3, a0_10, a1_10, a0_11, a1_11)
        STZ(2, a0_4, a1_4, a0_5, a1_5, a0_12, a1_12, a0_13, a1_13)
        STZ(3, a0_6, a1_6, a0_7, a1_7, a0_14, a1_14, a0_15, a1_15)

        // C reference on the same words/x
        #pragma unroll
        for (int rh = 0; rh < 4; ++rh)
        {
            const uint32_t wc = w0v[rh], wr = w1v[rh];
            const uint32_t vc = u0v[rh], vr = u1v[rh];
            const __half2 xlo = xh2[rh], xhi = xh2[rh + 4];
            #pragma unroll
            for (int col = 0; col < 2; ++col)
            {
                #pragma unroll
                for (int j = 0; j < 2; ++j)
                {
                    const int shh = 28 - 16 * col - 4 * j * 2;
                    const __half2 e0 = __halves2half2
                    (
                        dec_half_ref(__funnelshift_r(wc, wr, shh) & 0xffffu),
                        dec_half_ref(__funnelshift_r(wc, wr, shh - 4) & 0xffffu)
                    );
                    const __half2 e1 = __halves2half2
                    (
                        dec_half_ref(__funnelshift_r(vc, vr, shh) & 0xffffu),
                        dec_half_ref(__funnelshift_r(vc, vr, shh - 4) & 0xffffu)
                    );
                    const __half2 xx = j ? xhi : xlo;
                    const int c0 = col * 8 + rh * 2 + j;
                    r0[c0] = __builtin_amdgcn_fdot2(e0, xx, r0[c0], false);
                    r1[c0] = __builtin_amdgcn_fdot2(e1, xx, r1[c0], false);
                }
            }
        }
    }

    if (threadIdx.x == 0 && blockIdx.x == 0)
    {
        float out0[16] = { a0_0, a0_1, a0_2, a0_3, a0_4, a0_5, a0_6, a0_7,
                           a0_8, a0_9, a0_10, a0_11, a0_12, a0_13, a0_14, a0_15 };
        float out1[16] = { a1_0, a1_1, a1_2, a1_3, a1_4, a1_5, a1_6, a1_7,
                           a1_8, a1_9, a1_10, a1_11, a1_12, a1_13, a1_14, a1_15 };
        int nbad = 0;
        #pragma unroll
        for (int i = 0; i < 16; ++i)
        {
            y[i] = out0[i]; y[16 + i] = out1[i];
            ref[i] = r0[i]; ref[16 + i] = r1[i];
            if (fabsf(out0[i] - r0[i]) > 1e-3 || fabsf(out1[i] - r1[i]) > 1e-3) nbad++;
        }
        y[32] = (float) nbad;
    }
}

int main()
{
    const int n_kt = 8;
    uint32_t* subwords; uint32_t* wraps; __half2* xtiles; float *y, *ref;
    hipMalloc(&subwords, n_kt * 64 * 4);
    hipMalloc(&wraps, n_kt * 2 * 4);
    hipMalloc(&xtiles, n_kt * 8 * 4);
    hipMalloc(&y, 256); hipMalloc(&ref, 256);
    uint32_t* hw = new uint32_t[n_kt * 64];
    uint32_t* hwr = new uint32_t[n_kt * 2];
    __half2* hx = new __half2[n_kt * 8];
    for (int i = 0; i < n_kt * 64; ++i) hw[i] = 0x9e3779b9u * (i + 1) ^ (i * 0x85ebca6bu);
    for (int i = 0; i < n_kt * 2; ++i) hwr[i] = 0x1234567u + i * 7919u;
    for (int i = 0; i < n_kt * 8; ++i)
        hx[i] = __halves2half2(__float2half(0.01f * (i % 17) - 0.08f),
                               __float2half(0.013f * ((i * 7) % 11) - 0.06f));
    hipMemcpy(subwords, hw, n_kt * 64 * 4, hipMemcpyHostToDevice);
    hipMemcpy(wraps, hwr, n_kt * 2 * 4, hipMemcpyHostToDevice);
    hipMemcpy(xtiles, hx, n_kt * 8 * 4, hipMemcpyHostToDevice);
    probe<<<1, 32>>>(subwords, wraps, xtiles, y, ref, n_kt);
    hipDeviceSynchronize();
    float hy[33], hr[33];
    hipMemcpy(hy, y, 132, hipMemcpyDeviceToHost);
    hipMemcpy(hr, ref, 132, hipMemcpyDeviceToHost);
    printf("nbad chains = %d / 32\n", (int) hy[32]);
    for (int i = 0; i < 8 && i < 32; ++i)
        if (fabsf(hy[i] - hr[i]) > 1e-3)
            printf("chain %d: asm=%f ref=%f\n", i, hy[i], hr[i]);
    return 0;
}
