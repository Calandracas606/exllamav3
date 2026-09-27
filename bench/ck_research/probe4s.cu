#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

// Kernel-context repro: 4 stanzas + 32 live floats across a k-loop. Dumps the
// asm pack p0A0 per stanza (kt==0) and the C-computed expected pack per stanza.
__device__ __forceinline__ __half dec_half_ref(uint32_t w)
{
    uint32_t prod = w * 0x83DCD12Du;
    uint32_t s = 0x6400u + (prod & 0xffu) + ((prod >> 8) & 0xffu)
               + ((prod >> 16) & 0xffu) + ((prod >> 24) & 0xffu);
    __half h = __ushort_as_half((uint16_t)s);
    return __hfma(h, __ushort_as_half(0x1eee), __ushort_as_half(0xc931));
}

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

#define STZ(RH, OUT) \
    { \
        const uint32_t wc = w0v[RH], wr = w1v[RH]; \
        uint32_t p0A0, ta, tb; \
        __asm volatile \
        ( \
            V9PK(p0A0, wc, wr, 28, 24) \
            : [p0A0] "=&v"(p0A0), [ta] "=&v"(ta), [tb] "=&v"(tb) \
            : [wc] "r"(wc), [wr] "r"(wr), \
              [kmul] "r"(kmul), [kinv] "v"(kinv), [kbias] "v"(kbias) \
        ); \
        OUT = p0A0; \
    }

__global__ void probe(const uint32_t* words, uint32_t* out, float* accs, int n)
{
    const uint32_t kmul = 0x83DCD12Du;
    const __half2 kinv = __half2half2(__ushort_as_half(0x1eee));
    const __half2 kbias = __half2half2(__ushort_as_half(0xc931));
    float a[32];
    #pragma unroll
    for (int i = 0; i < 32; ++i) a[i] = 0.0f;

    for (int kt = 0; kt < n; ++kt)
    {
        const uint32_t w0v[4] = { words[0], words[1], words[2], words[3] };
        const uint32_t w1v[4] = { words[4], words[5], words[6], words[7] };
        uint32_t o0, o1, o2, o3;
        STZ(0, o0)
        STZ(1, o1)
        STZ(2, o2)
        STZ(3, o3)
        if (kt == 0 && threadIdx.x == 0)
        {
            out[0] = o0; out[1] = o1; out[2] = o2; out[3] = o3;
            #pragma unroll
            for (int rh = 0; rh < 4; ++rh)
            {
                const __half2 e = __halves2half2
                (
                    dec_half_ref(__funnelshift_r(w0v[rh], w1v[rh], 28) & 0xffffu),
                    dec_half_ref(__funnelshift_r(w0v[rh], w1v[rh], 24) & 0xffffu)
                );
                out[4 + rh] = *reinterpret_cast<const uint32_t*>(&e);
                out[8 + rh] = w0v[rh]; out[12 + rh] = w1v[rh];
            }
        }
        #pragma unroll
        for (int i = 0; i < 32; ++i) a[i] += (float)(o0 + o1 + o2 + o3);
    }
    #pragma unroll
    for (int i = 0; i < 32; ++i) accs[i] = a[i];
}

int main()
{
    uint32_t* words; uint32_t* out; float* accs;
    hipMalloc(&words, 64); hipMalloc(&out, 128); hipMalloc(&accs, 256);
    uint32_t hw[8];
    for (int i = 0; i < 8; ++i) hw[i] = 0x9e3779b9u * (i + 1) ^ (i * 0x85ebca6bu);
    hipMemcpy(words, hw, 32, hipMemcpyHostToDevice);
    probe<<<1, 32>>>(words, out, accs, 8);
    hipDeviceSynchronize();
    uint32_t h[16];
    hipMemcpy(h, out, 64, hipMemcpyDeviceToHost);
    for (int rh = 0; rh < 4; ++rh)
    {
        const bool ok = h[rh] == h[4 + rh];
        printf("rh=%d: asm=%08x expected=%08x %s  (wc=%08x wr=%08x)\n",
               rh, h[rh], h[4 + rh], ok ? "OK" : "MISMATCH", h[8 + rh], h[12 + rh]);
    }
    return 0;
}
