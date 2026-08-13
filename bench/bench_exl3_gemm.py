"""Benchmark exl3_gemm implementations using torch.utils.benchmark.

Compares the Triton op against the reference reconstruct_hgemm path across
various shapes. Uses torch.utils.benchmark.Timer for wall-clock measurement
(not manual timing).

Usage:
    python bench/bench_exl3_gemm.py [--device cuda:0]
"""
import argparse
import os

import torch
from torch.utils.benchmark import Timer

from exllamav3.ext import exllamav3_ext as ext
from exllamav3.exl3_gemm_triton import exl3_gemm as exl3_gemm_triton


def make_random_trellis(in_features, out_features, K, dev):
    rows = in_features // 16
    cols = out_features // 16
    packed_size = 256 * K // 16
    encoded = torch.randint(-32768, 32767, (rows, cols, 256), dtype=torch.int16, device=dev)
    packed = torch.zeros((rows, cols, packed_size), dtype=torch.int16, device=dev)
    ext.pack_trellis(packed, encoded.contiguous(), K)
    return packed


def reference_hgemm(x, trellis, suh, svh, K, mcg, mul1, in_f, out_f, dev):
    original_shape = x.shape
    x = x.view(-1, in_f)
    rows = x.shape[0]
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    w = torch.empty((in_f, out_f), dtype=torch.half, device=dev)
    ext.reconstruct(w, trellis, K, mcg, mul1)
    y = torch.empty((rows, out_f), dtype=torch.half, device=dev)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y.view(original_shape[:-1] + (out_f,))


SHAPES = [
    # (in_features, out_features, K, label)
    (512, 512, 4, "512x512 K4"),
    (1024, 1024, 4, "1024x1024 K4"),
    (2048, 2048, 4, "2048x2048 K4"),
    (4096, 4096, 4, "4096x4096 K4"),
    (2048, 4096, 4, "2048x4096 K4"),
    (4096, 2048, 4, "4096x2048 K4"),
    (1024, 1024, 6, "1024x1024 K6"),
]

BATCH_SIZES = [1, 4, 16]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default=os.environ.get("EXL_TEST_DEVICE", "cuda:0"))
    args = parser.parse_args()
    dev = args.device

    print(f"{'Shape':<25} {'Bsz':>4} {'Reference (ms)':>15} {'Triton (ms)':>12} {'Ratio':>8}")
    print(f"{'-'*25} {'-'*4} {'-'*15} {'-'*12} {'-'*8}")

    for in_f, out_f, K, label in SHAPES:
        torch.manual_seed(42)
        trellis = make_random_trellis(in_f, out_f, K, dev)
        suh = torch.ones(in_f, dtype=torch.half, device=dev)
        svh = torch.ones(out_f, dtype=torch.half, device=dev)
        mcg, mul1 = True, False

        for bsz in BATCH_SIZES:
            x = torch.randn(bsz, in_f, dtype=torch.half, device=dev)

            # Reference
            t_ref = Timer(
                stmt="fn(x)",
                globals={"fn": lambda x: reference_hgemm(x, trellis, suh, svh, K, mcg, mul1, in_f, out_f, dev), "x": x},
                num_threads=1,
            )

            # Triton
            t_tri = Timer(
                stmt="fn(x)",
                globals={"fn": lambda x: exl3_gemm_triton(x, trellis, suh, svh, K, mcg, mul1, in_f, out_f, dev, torch.half), "x": x},
                num_threads=1,
            )

            m_ref = t_ref.adaptive_num_calls(min_measured_time=0.1).mean * 1000
            m_tri = t_tri.adaptive_num_calls(min_measured_time=0.1).mean * 1000
            ratio = m_tri / m_ref if m_ref > 0 else float('inf')

            print(f"{label:<25} {bsz:>4} {m_ref:>15.2f} {m_tri:>12.2f} {ratio:>7.2f}x")


if __name__ == "__main__":
    main()
