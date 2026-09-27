#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdio>

// Deterministic VOPD pairing: pin operands to explicit regs (banks = aligned
// pairs: v2/v3 bank1, v4/v5 bank2, v6/v7 bank3; dsts v0/v1 opposite parity).
__global__ void probe(const __half2* a, const __half2* b, float* out)
{
    const __half2 xh = a[threadIdx.x];
    const __half2 wh = b[threadIdx.x];
    const __half2 vh = b[threadIdx.x + 32];
    const float acc_in = out[threadIdx.x];
    const float acc2_in = out[threadIdx.x + 32];

    register float acc  __asm__("v0") = acc_in;
    register float acc2 __asm__("v3") = acc2_in;
    register __half2 w  __asm__("v2") = wh;
    register __half2 x  __asm__("v4") = xh;
    register __half2 v  __asm__("v12") = vh;
    register __half2 x2c __asm__("v14") = xh;
    __asm volatile
    (
        "v_dual_dot2acc_f32_f16 %0, %2, %3 :: v_dual_dot2acc_f32_f16 %1, %4, %5"
        : "+v"(acc), "+v"(acc2)
        : "v"(w), "v"(x), "v"(v), "v"(x2c)
    );
    out[threadIdx.x] = acc;
    out[threadIdx.x + 32] = acc2;
}

int main()
{
    __half2* a; __half2* b; float* out;
    hipMalloc(&a, 256); hipMalloc(&b, 256); hipMalloc(&out, 512);
    const __half2 ha = __halves2half2(__float2half(1.0f), __float2half(2.0f));
    const __half2 hw = __halves2half2(__float2half(3.0f), __float2half(4.0f));
    const __half2 hv = __halves2half2(__float2half(5.0f), __float2half(6.0f));
    hipMemset(a, 0, 256);
    hipMemset(b, 0, 256);
    hipMemcpy(a, &ha, 4, hipMemcpyHostToDevice);
    hipMemcpy(b, &hw, 4, hipMemcpyHostToDevice);
    hipMemcpy(b + 32, &hv, 4, hipMemcpyHostToDevice);
    hipMemset(out, 0, 512);
    probe<<<1, 32>>>(a, b, out);
    hipDeviceSynchronize();
    float h_out[64];
    hipMemcpy(h_out, out, 256, hipMemcpyDeviceToHost);
    printf("acc=%f (want 11)   acc2=%f (want 17)\n", h_out[0], h_out[32]);
    return 0;
}
