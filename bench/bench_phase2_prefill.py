"""Phase 2: EXL3 prefill paths at 27B shapes, M in {128, 512, 2048}.

Paths:
  1. incumbent fused Triton (exl3_gemm_triton.exl3_gemm, M>1 branch)
  2. reference reconstruct -> hgemm (what the model runs today for M > 144)
  3. candidate exl3_prefill_fly (Triton dequant-to-dense [N,K] + FlyDSL WMMA)

End-to-end had->gemm->had, dequant cost INCLUDED (weights cleared between
iters for the candidate; steady-state numbers also reported since chunked
prefill reuses weights across chunks when the cache holds).
"""
import os, sys, json
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.exl3_gemm_triton import exl3_gemm, _decode_lut, _get_perm  # noqa
import exllamav3.exl3_prefill_fly as pf

DEV = os.environ.get("EXL_TEST_DEVICE", "cuda:0")
SHAPES = [  # (in_f, out_f, bits)
    (5120, 17408, 4),
    (17408, 5120, 4),
    (5120, 248320, 6),
]
MS = [128, 512, 2048]


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


def make_random_trellis(in_features, out_features, K, dev):
    rows = in_features // 16
    cols = out_features // 16
    packed_size = 256 * K // 16
    encoded = torch.randint(-32768, 32767, (rows, cols, 256), dtype=torch.int16, device=dev)
    packed = torch.zeros((rows, cols, packed_size), dtype=torch.int16, device=dev)
    ext.pack_trellis(packed, encoded.contiguous(), K)
    return packed


def bench(fn, iters=20, warmup=5):
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
    return s.elapsed_time(e) / iters


def main():
    results = []
    for in_f, out_f, bits in SHAPES:
        torch.manual_seed(42)
        trellis = make_random_trellis(in_f, out_f, bits, DEV)
        suh = (torch.randint(0, 2, (in_f,), device=DEV).to(torch.half) * 2 - 1)
        svh = (torch.randint(0, 2, (out_f,), device=DEV).to(torch.half) * 2 - 1)

        # dequant-only timings
        t_recon = bench(lambda: ext.reconstruct(
            _w_buf(in_f, out_f), trellis, bits, False, False))
        t_dq = bench(lambda: pf.dequant_dense_fly(trellis, bits, 0))

        for M in MS:
            x = torch.randn(M, in_f, dtype=torch.half, device=DEV)

            def path_triton():
                return exl3_gemm(x, trellis, suh, svh, bits, False, False,
                                 in_f, out_f, torch.device(DEV))

            w_kn = torch.empty((in_f, out_f), dtype=torch.half, device=DEV)
            xh = torch.empty_like(x)
            y2 = torch.empty((M, out_f), dtype=torch.half, device=DEV)

            def path_recon():
                ext.had_r_128(x, xh, suh, None, 1.0)
                ext.reconstruct(w_kn, trellis, bits, False, False)
                ext.hgemm(xh, w_kn, y2)
                ext.had_r_128(y2, y2, None, svh, 1.0)
                return y2

            def path_fly_cold():
                pf._W_CACHE.clear()
                return pf.exl3_prefill_fly(
                    x, trellis, suh, svh, bits, False, False, in_f, out_f,
                    torch.device(DEV), cache_weights=True)

            def path_fly_hot():
                return pf.exl3_prefill_fly(
                    x, trellis, suh, svh, bits, False, False, in_f, out_f,
                    torch.device(DEV), cache_weights=True)

            t1 = bench(path_triton)
            t2 = bench(path_recon)
            # candidate cold: drop the W cache before every call
            def cold():
                pf._W_CACHE.clear()
                return path_fly_hot()
            t3 = bench(cold, iters=10, warmup=2)
            t4 = bench(path_fly_hot)  # steady state (W in DRAM cache dict)

            # correctness spot-check vs reconstruct path
            y_ref = path_recon().float()
            y_f = path_fly_hot().float()
            dmax = (y_f - y_ref).abs().max().item()

            row = dict(in_f=in_f, out_f=out_f, bits=bits, M=M,
                       triton_ms=t1, recon_ms=t2, fly_cold_ms=t3, fly_hot_ms=t4,
                       recon_dequant_ms=t_recon, fly_dequant_ms=t_dq, dmax=dmax)
            results.append(row)
            print(f"[{bits}b {in_f:5d}x{out_f:6d} M={M:4d}] "
                  f"triton {t1:8.2f} | recon+hgemm {t2:8.2f} | "
                  f"fly cold {t3:8.2f} hot {t4:8.2f} | dq: recon {t_recon:7.2f} fly {t_dq:7.2f} | dmax {dmax:.3f}")
            del x, xh, y2, w_kn
            torch.cuda.empty_cache()
        del trellis
        torch.cuda.empty_cache()

    print("\n=== Phase 2 summary (ms) ===")
    print(f"{'shape':>22s} {'M':>5s} {'triton':>8s} {'recon':>8s} {'flycold':>8s} {'flyhot':>8s} {'dmax':>6s}")
    for r in results:
        print(f"{str(r['bits'])+'b '+str(r['in_f'])+'x'+str(r['out_f']):>22s} "
              f"{r['M']:5d} {r['triton_ms']:8.2f} {r['recon_ms']:8.2f} {r['fly_cold_ms']:8.2f} {r['fly_hot_ms']:8.2f} {r['dmax']:6.3f}")
    with open(os.path.join(os.path.dirname(__file__), "phase2_results.json"), "w") as f:
        json.dump(results, f, indent=1)


def _w_buf(in_f, out_f):
    global _W
    try:
        if _W.shape != (in_f, out_f):
            raise AttributeError
    except AttributeError:
        _W = torch.empty((in_f, out_f), dtype=torch.half, device=DEV)
    return _W


_W = None

if __name__ == "__main__":
    ramp_clocks()
    main()
