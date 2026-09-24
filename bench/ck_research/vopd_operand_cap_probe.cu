#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// Does clang's GCC-style inline asm cap operand count before the 16x32-tile
// variant (which needs ~38 operands per fused stanza) becomes unbuildable?
#define DUAL(A, W0, X0, B, W1, X1) \
    "v_dual_dot2acc_f32_f16 %[" #A "], %[" #W0 "], %[" #X0 "] :: " \
    "v_dual_dot2acc_f32_f16 %[" #B "], %[" #W1 "], %[" #X1 "]\n"

__global__ void probe(const __half2* a, const __half2* b, float* out, int n)
{
    float c0 = 0.f, c1 = 0.f, c2 = 0.f, c3 = 0.f, c4 = 0.f, c5 = 0.f, c6 = 0.f, c7 = 0.f;
    float d0 = 0.f, d1 = 0.f, d2 = 0.f, d3 = 0.f, d4 = 0.f, d5 = 0.f, d6 = 0.f, d7 = 0.f;
    float e0 = 0.f, e1 = 0.f, e2 = 0.f, e3 = 0.f, e4 = 0.f, e5 = 0.f, e6 = 0.f, e7 = 0.f;
    float f0 = 0.f, f1 = 0.f, f2 = 0.f, f3 = 0.f, f4 = 0.f, f5 = 0.f, f6 = 0.f, f7 = 0.f;
    uint32_t p0, p1, p2, p3, p4, p5, p6, p7;
    uint32_t q0, q1, q2, q3, q4, q5, q6, q7;
    const __half2 w0 = a[threadIdx.x], w1 = a[threadIdx.x + 1];
    const __half2 w2 = a[threadIdx.x + 2], w3 = a[threadIdx.x + 3];
    const __half2 x0 = b[threadIdx.x], x1 = b[threadIdx.x + 1];
    for (int i = 0; i < n; ++i)
    {
        __asm volatile
        (
            "v_perm_b32 %[p0], %[w0], %[w1], 0x05040100\n"
            "v_perm_b32 %[p1], %[w1], %[w2], 0x05040100\n"
            "v_perm_b32 %[p2], %[w2], %[w3], 0x05040100\n"
            "v_perm_b32 %[p3], %[w3], %[w0], 0x05040100\n"
            "v_perm_b32 %[p4], %[w1], %[w0], 0x05040100\n"
            "v_perm_b32 %[p5], %[w2], %[w1], 0x05040100\n"
            "v_perm_b32 %[p6], %[w3], %[w2], 0x05040100\n"
            "v_perm_b32 %[p7], %[w0], %[w3], 0x05040100\n"
            "v_perm_b32 %[q0], %[x0], %[x1], 0x05040100\n"
            "v_perm_b32 %[q1], %[x1], %[x0], 0x05040100\n"
            "v_perm_b32 %[q2], %[x0], %[w0], 0x05040100\n"
            "v_perm_b32 %[q3], %[x1], %[w1], 0x05040100\n"
            "v_perm_b32 %[q4], %[x0], %[w2], 0x05040100\n"
            "v_perm_b32 %[q5], %[x1], %[w3], 0x05040100\n"
            "v_perm_b32 %[q6], %[x0], %[w0], 0x05040100\n"
            "v_perm_b32 %[q7], %[x1], %[w1], 0x05040100\n"
            DUAL(c0, p0, q0, d0, p1, q1)
            DUAL(c1, p1, q0, d1, p2, q1)
            DUAL(c2, p2, q0, d2, p3, q1)
            DUAL(c3, p3, q0, d3, p0, q1)
            DUAL(c4, p4, q0, d4, p5, q1)
            DUAL(c5, p5, q0, d5, p6, q1)
            DUAL(c6, p6, q0, d6, p7, q1)
            DUAL(c7, p7, q0, d7, p0, q1)
            DUAL(e0, q2, p0, f0, q3, p1)
            DUAL(e1, q3, p0, f1, q4, p1)
            DUAL(e2, q4, p0, f2, q5, p1)
            DUAL(e3, q5, p0, f3, q6, p1)
            DUAL(e4, q6, p0, f4, q7, p1)
            DUAL(e5, q7, p0, f5, q2, p1)
            DUAL(e6, q2, p0, f6, q3, p1)
            DUAL(e7, q3, p0, f7, q4, p1)
            : [c0] "+v"(c0), [c1] "+v"(c1), [c2] "+v"(c2), [c3] "+v"(c3),
              [c4] "+v"(c4), [c5] "+v"(c5), [c6] "+v"(c6), [c7] "+v"(c7),
              [d0] "+v"(d0), [d1] "+v"(d1), [d2] "+v"(d2), [d3] "+v"(d3),
              [d4] "+v"(d4), [d5] "+v"(d5), [d6] "+v"(d6), [d7] "+v"(d7),
              [e0] "+v"(e0), [e1] "+v"(e1), [e2] "+v"(e2), [e3] "+v"(e3),
              [e4] "+v"(e4), [e5] "+v"(e5), [e6] "+v"(e6), [e7] "+v"(e7),
              [f0] "+v"(f0), [f1] "+v"(f1), [f2] "+v"(f2), [f3] "+v"(f3),
              [f4] "+v"(f4), [f5] "+v"(f5), [f6] "+v"(f6), [f7] "+v"(f7),
              [p0] "=&v"(p0), [p1] "=&v"(p1), [p2] "=&v"(p2), [p3] "=&v"(p3),
              [p4] "=&v"(p4), [p5] "=&v"(p5), [p6] "=&v"(p6), [p7] "=&v"(p7),
              [q0] "=&v"(q0), [q1] "=&v"(q1), [q2] "=&v"(q2), [q3] "=&v"(q3),
              [q4] "=&v"(q4), [q5] "=&v"(q5), [q6] "=&v"(q6), [q7] "=&v"(q7)
            : [w0] "v"(w0), [w1] "v"(w1), [w2] "v"(w2), [w3] "v"(w3),
              [x0] "v"(x0), [x1] "v"(x1)
        );
    }
    out[threadIdx.x] = c0 + c1 + c2 + c3 + c4 + c5 + c6 + c7
                     + d0 + d1 + d2 + d3 + d4 + d5 + d6 + d7
                     + e0 + e1 + e2 + e3 + e4 + e5 + e6 + e7
                     + f0 + f1 + f2 + f3 + f4 + f5 + f6 + f7;
}
