#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

// Compare the EXACT V9PK macro text (copied from the kernel header) against
// dec_half for all 8 (col, j) window pairs of one rh word-pair.
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

__global__ void probe(uint32_t wc, uint32_t wr, float* out)
{
    const uint32_t kmul = 0x83DCD12Du;
    const __half2 kinv = __half2half2(__ushort_as_half(0x1eee));
    const __half2 kbias = __half2half2(__ushort_as_half(0xc931));
    uint32_t p0A0, p0A1, p0B0, p0B1, ta, tb;
    uint32_t p1A0, p1A1, p1B0, p1B1;
    __asm volatile
    (
        V9PK(p0A0, wc, wr, 28, 24)
        V9PK(p0A1, wc, wr, 20, 16)
        V9PK(p0B0, wc, wr, 12, 8)
        V9PK(p0B1, wc, wr, 4, 0)
        V9PK(p1A0, vc, vr, 28, 24)
        V9PK(p1A1, vc, vr, 20, 16)
        V9PK(p1B0, vc, vr, 12, 8)
        V9PK(p1B1, vc, vr, 4, 0)
        : [p0A0] "=&v"(p0A0), [p0A1] "=&v"(p0A1),
          [p0B0] "=&v"(p0B0), [p0B1] "=&v"(p0B1),
          [p1A0] "=&v"(p1A0), [p1A1] "=&v"(p1A1),
          [p1B0] "=&v"(p1B0), [p1B1] "=&v"(p1B1),
          [ta] "=&v"(ta), [tb] "=&v"(tb)
        : [wc] "r"(wc), [wr] "r"(wr), [vc] "r"(wc), [vr] "r"(wr),
          [kmul] "r"(kmul), [kinv] "v"(kinv), [kbias] "v"(kbias)
    );
    // reference: v8 window extraction. window li -> funnelshift_r(wc, wr, 28-4li)
    // colA j0 = li 0,1 (shifts 28,24); j1 = li 2,3 (20,16); colB j0 = li 4,5
    // (12,8); j1 = li 6,7 (4,0). Pack = (lo-shift first window, second).
    // x pair (r0, r1) must pair (w@sh0, w@sh1) — print halves per pack.
    const __half2 q0 = *reinterpret_cast<__half2*>(&p0A0);
    const __half2 q1 = *reinterpret_cast<__half2*>(&p0A1);
    const __half2 q2 = *reinterpret_cast<__half2*>(&p0B0);
    const __half2 q3 = *reinterpret_cast<__half2*>(&p0B1);
    printf("1A0 lo=%04x hi=%04x  1B1 lo=%04x hi=%04x\n",
        (unsigned)__half_as_ushort(__low2half(*reinterpret_cast<__half2*>(&p1A0))),
        (unsigned)__half_as_ushort(__high2half(*reinterpret_cast<__half2*>(&p1A0))),
        (unsigned)__half_as_ushort(__low2half(*reinterpret_cast<__half2*>(&p1B1))),
        (unsigned)__half_as_ushort(__high2half(*reinterpret_cast<__half2*>(&p1B1))));
    printf("A0: lo=%04x (want %04x) hi=%04x (want %04x)\n",
        (unsigned)__half_as_ushort(__low2half(q0)),
        (unsigned)__half_as_ushort(dec_half_ref(__funnelshift_r(wc, wr, 28) & 0xffffu)),
        (unsigned)__half_as_ushort(__high2half(q0)),
        (unsigned)__half_as_ushort(dec_half_ref(__funnelshift_r(wc, wr, 24) & 0xffffu)));
    printf("A1: lo=%04x (want %04x) hi=%04x (want %04x)\n",
        (unsigned)__half_as_ushort(__low2half(q1)),
        (unsigned)__half_as_ushort(dec_half_ref(__funnelshift_r(wc, wr, 20) & 0xffffu)),
        (unsigned)__half_as_ushort(__high2half(q1)),
        (unsigned)__half_as_ushort(dec_half_ref(__funnelshift_r(wc, wr, 16) & 0xffffu)));
    printf("B0: lo=%04x (want %04x) hi=%04x (want %04x)\n",
        (unsigned)__half_as_ushort(__low2half(q2)),
        (unsigned)__half_as_ushort(dec_half_ref(__funnelshift_r(wc, wr, 12) & 0xffffu)),
        (unsigned)__half_as_ushort(__high2half(q2)),
        (unsigned)__half_as_ushort(dec_half_ref(__funnelshift_r(wc, wr, 8) & 0xffffu)));
    printf("B1: lo=%04x (want %04x) hi=%04x (want %04x)\n",
        (unsigned)__half_as_ushort(__low2half(q3)),
        (unsigned)__half_as_ushort(dec_half_ref(__funnelshift_r(wc, wr, 4) & 0xffffu)),
        (unsigned)__half_as_ushort(__high2half(q3)),
        (unsigned)__half_as_ushort(dec_half_ref(__funnelshift_r(wc, wr, 0) & 0xffffu)));
    if (threadIdx.x == 0)
        out[0] = 1.0f;
}

int main()
{
    float* out; hipMalloc(&out, 16);
    probe<<<1, 1>>>(0xA5C3F07Bu, 0x12345678u, out);
    hipDeviceSynchronize();
    return 0;
}
