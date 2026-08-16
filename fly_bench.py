"""Interleaved A/B benchmark: FlyDSL GEMV vs Triton GEMV kernel.

Methodology per AGENTS.md: 50+ warmup iterations to ramp GPU clocks, then
alternating rounds of Triton / Fly calls on the same tensors (in-process A/B
neutralizes clock drift), sync-bracketed wall time. Effective bandwidth =
trellis bytes read per call / time.
"""
import os, sys, time, torch
sys.path.insert(0, ".")

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_triton import exl3_gemm_triton  # registers the op
from exllamav3.modules.quant.exl3_triton import _decode_lut, _get_perm
import exllamav3.exl3_gemm_fly as fly
import triton

dev = "cuda:0"

def make_random_trellis(in_features, out_features, K, dev):
    rows = in_features // 16
    cols = out_features // 16
    packed_size = 256 * K // 16
    encoded = torch.randint(-32768, 32767, (rows, cols, 256), dtype=torch.int16, device=dev)
    packed = torch.zeros((rows, cols, packed_size), dtype=torch.int16, device=dev)
    ext.pack_trellis(packed, encoded.contiguous(), K)
    return packed

def bench(fn, warmup=60, iters=40):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    # timed
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    return (t1 - t0) / iters * 1e3  # ms

SHAPES = [
    (5120, 17408, 4),   # 45.2 MB trellis
    (2048, 5120, 4),    #  6.5 MB
    (5120, 248320, 6),  # 953 MB lm_head-like (spans many MB, exceeds L2)
]

# long clock ramp: several seconds of large matmuls (AGENTS.md discipline)
print("ramping clocks...")
a = torch.randn(4096, 4096, dtype=torch.half, device=dev)
b = torch.randn(4096, 4096, dtype=torch.half, device=dev)
t0 = time.perf_counter()
while time.perf_counter() - t0 < 5.0:
    for _ in range(200):
        a @ b
    torch.cuda.synchronize()
print("clocks ramped")

for in_f, out_f, K in SHAPES:
    torch.manual_seed(42)
    trellis = make_random_trellis(in_f, out_f, K, dev)
    suh = (torch.randint(0, 2, (in_f,), device=dev).to(torch.half) * 2 - 1)
    svh = (torch.randint(0, 2, (out_f,), device=dev).to(torch.half) * 2 - 1)
    x = torch.randn(1, in_f, dtype=torch.half, device=dev)

    tbytes = trellis.numel() * 2
    gb = tbytes / 1e9

    # Triton full linear (had -> gemm -> had)
    from exllamav3.modules.quant.exl3_triton import exl3_gemm as exl3_gemm_triton_fn
    # Fly full linear
    y_t = exl3_gemm_triton_fn(x, trellis, suh, svh, K, False, False, in_f, out_f, torch.device(dev))
    y_f = fly.exl3_gemm_fly(x, trellis, suh, svh, K, False, False, in_f, out_f, torch.device(dev))
    torch.testing.assert_close(y_f, y_t, rtol=2e-2, atol=0.5)

    # Alternate A/B in rounds so clock state is shared
    def run_triton():
        exl3_gemm_triton_fn(x, trellis, suh, svh, K, False, False, in_f, out_f, torch.device(dev))
    def run_fly():
        fly.exl3_gemm_fly(x, trellis, suh, svh, K, False, False, in_f, out_f, torch.device(dev))

    for _ in range(60):
        run_triton(); run_fly()
    torch.cuda.synchronize()

    ROUNDS = 3
    t_tr = t_fl = 0.0
    for r in range(ROUNDS):
        for _ in range(20): run_triton()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20): run_triton()
        torch.cuda.synchronize(); t_tr += time.perf_counter() - t0
        for _ in range(20): run_fly()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(20): run_fly()
        torch.cuda.synchronize(); t_fl += time.perf_counter() - t0
    ms_tr = t_tr / ROUNDS / 20 * 1e3
    ms_fl = t_fl / ROUNDS / 20 * 1e3
    print(f"[{K}b {in_f:5d}x{out_f:6d}] {gb*1000:7.1f} MB  "
          f"triton {ms_tr:7.3f} ms ({gb/ms_tr*1000:6.0f} GB/s)  "
          f"fly {ms_fl:7.3f} ms ({gb/ms_fl*1000:6.0f} GB/s)  "
          f"ratio {ms_fl/ms_tr:.2f}x")
