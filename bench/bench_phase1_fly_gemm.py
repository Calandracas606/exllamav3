"""Phase 1: standalone dense GEMM benchmark — FlyDSL RDNA3 WMMA vs rocBLAS.

27B shapes (K x N) at M in {128, 512, 2048}. Clock-ramp discipline: 5 s of
big matmuls before timing (AGENTS.md).
"""
import os
import sys
import json
import itertools
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3._flydsl_kernels.rdna3_f16_gemm import create_wmma_gemm_module

DEV = "cuda:0"
SHAPES = [
    (5120, 17408),
    (17408, 5120),
    (5120, 248320),
]
MS = [128, 512, 2048]

# tile candidates: (reg_m, reg_n, reg_k, waves_m, waves_n)
# BLOCK_M = 16*rm*wm, BLOCK_N = 16*rn*wn, BLOCK_K = 16*rk
# LDS budget 64 KB/WG: 2*(BM+BN)*(BK+8)*2 bytes
def lds_bytes(rm, rn, rk, wm, wn, akp=8, bkp=8):
    bm, bn, bk = 16 * rm * wm, 16 * rn * wn, 16 * rk
    return 2 * (bm * (bk + akp) + bn * (bk + bkp)) * 2

TILES = []
for rm, rn, rk, wm, wn in itertools.product((2, 4, 8), (2, 4, 8), (2, 4), (1, 2), (1, 2)):
    bm, bn, bk = 16 * rm * wm, 16 * rn * wn, 16 * rk
    threads = 32 * wm * wn
    thrs_k = bk // 8
    thrs_m = threads // thrs_k
    if thrs_k * thrs_m != threads:
        continue
    if bm % thrs_m or bn % thrs_m:
        continue
    if lds_bytes(rm, rn, rk, wm, wn) > 64 * 1024:
        continue
    if rm * rn * 8 > 128:  # acc VGPR budget per thread
        continue
    TILES.append((rm, rn, rk, wm, wn))

_seen = set()
_t = []
for t in TILES:
    bm, bn, bk = 16 * t[0] * t[3], 16 * t[1] * t[4], 16 * t[2]
    key = (bm, bn, bk, 32 * t[3] * t[4])
    if key not in _seen:
        _seen.add(key)
        _t.append(t)
TILES = _t
TILES.sort(key=lambda t: -(16 * t[0] * t[3]) * (16 * t[1] * t[4]))


def ramp_clocks(seconds=5.0):
    a = torch.randn(4096, 4096, dtype=torch.half, device=DEV)
    b = torch.randn(4096, 4096, dtype=torch.half, device=DEV)
    ev = torch.cuda.Event(enable_timing=True)
    ev2 = torch.cuda.Event(enable_timing=True)
    ev.record()
    while True:
        a @ b
        ev2.record()
        torch.cuda.synchronize()
        if ev.elapsed_time(ev2) / 1e3 >= seconds:
            break
    del a, b
    torch.cuda.empty_cache()


def bench(fn, iters=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters  # ms


def main():
    results = []
    ramp_clocks()
    for K, N in SHAPES:
        for M in MS:
            A = (torch.randn(M, K, dtype=torch.half, device=DEV) * 0.1)
            BT = (torch.randn(N, K, dtype=torch.half, device=DEV) * 0.1)
            flops = 2.0 * M * N * K

            C_r = torch.empty(M, N, dtype=torch.half, device=DEV)
            t_blas = bench(lambda: torch.matmul(A, BT.t(), out=C_r))
            blas_tf = flops / (t_blas * 1e-3) / 1e12

            best = None
            tried = 0
            for (rm, rn, rk, wm, wn) in TILES:
                bm, bn, bk = 16 * rm * wm, 16 * rn * wn, 16 * rk
                if M % bm or N % bn or K % bk:
                    continue
                try:
                    launch, _, _, _ = create_wmma_gemm_module(
                        M, N, K, in_dtype="f16", out_dtype="f16",
                        reg_m=rm, reg_n=rn, reg_k=rk, waves_m=wm, waves_n=wn, group_m=8,
                    )
                except Exception as ex:
                    print(f"    skip {bm}x{bn}x{bk}: {type(ex).__name__}: {ex}")
                    continue
                C = torch.zeros(M, N, dtype=torch.half, device=DEV)
                launch(C, A, BT, torch.cuda.current_stream())
                torch.cuda.synchronize()
                ref = A.float() @ BT.float().t()
                err = (C.float() - ref).abs().max().item()
                ok = err < 0.5
                t = bench(lambda: launch(C, A, BT, torch.cuda.current_stream()))
                tried += 1
                tf = flops / (t * 1e-3) / 1e12
                tag = f"{bm}x{bn}x{bk}w{32*wm*wn}"
                print(f"    fly {tag:16s} {t:8.3f} ms  {tf:7.1f} TF/s  err={err:.4f} {'OK' if ok else 'FAIL'}")
                if ok and (best is None or t < best[1]):
                    best = (tag, t, tf)
                del C
            # f32 out with the best tile
            t_f32 = None
            if best is not None:
                for (rm, rn, rk, wm, wn) in TILES:
                    if f"{16*rm*wm}x{16*rn*wn}x{16*rk}w{32*wm*wn}" == best[0]:
                        try:
                            launch32, _, _, _ = create_wmma_gemm_module(
                                M, N, K, in_dtype="f16", out_dtype="f32",
                                reg_m=rm, reg_n=rn, reg_k=rk, waves_m=wm, waves_n=wn, group_m=8,
                            )
                            C32 = torch.zeros(M, N, dtype=torch.float, device=DEV)
                            launch32(C32, A, BT, torch.cuda.current_stream())
                            torch.cuda.synchronize()
                            t_f32 = bench(lambda: launch32(C32, A, BT, torch.cuda.current_stream()))
                        except Exception as ex:
                            print(f"    f32-out failed: {ex}")
                        break

            results.append(dict(K=K, N=N, M=M, blas_ms=t_blas, blas_tf=blas_tf,
                                fly_tag=best[0] if best else None,
                                fly_ms=best[1] if best else None,
                                fly_tf=best[2] if best else None,
                                fly_f32_ms=t_f32, tried=tried))
            print(f"  [{K}x{N}] M={M}: rocBLAS {t_blas:.3f} ms ({blas_tf:.1f} TF/s) | "
                  f"fly {best[1]:.3f} ms ({best[2]:.1f} TF/s, {best[0]})"
                  f"{' | f32-out ' + format(t_f32, '.3f') + ' ms' if t_f32 else ''}")
            del A, BT, C_r
            torch.cuda.empty_cache()

    print("\n=== Phase 1 summary ===")
    print(f"{'K':>6s} {'N':>7s} {'M':>5s} {'rocBLAS ms':>10s} {'TF/s':>6s} {'fly ms':>8s} {'TF/s':>6s} {'tile':>14s} {'f32out ms':>9s}")
    for r in results:
        print(f"{r['K']:6d} {r['N']:7d} {r['M']:5d} {r['blas_ms']:10.3f} {r['blas_tf']:6.1f} "
              f"{(r['fly_ms'] or 0):8.3f} {(r['fly_tf'] or 0):6.1f} {str(r['fly_tag']):>14s} "
              f"{(format(r['fly_f32_ms'], '.3f') if r['fly_f32_ms'] else '-'):>9s}")
    with open(os.path.join(os.path.dirname(__file__), "phase1_results.json"), "w") as f:
        json.dump(results, f, indent=1)


if __name__ == "__main__":
    main()
