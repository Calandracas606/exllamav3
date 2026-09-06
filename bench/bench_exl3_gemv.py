"""EXL3 Triton GEMV decode bandwidth benchmark (M == 1).

Measures the production decode path (``_linear_exl3_triton``: input Hadamard
-> fused dequant+GEMV [split-K where eligible] -> fused reduce + output
Hadamard) for every trellis bit width K = 1..8 on three shape classes, and
reports both achieved DRAM bandwidth (GB/s over trellis bytes + scales) and
decode throughput (weights/s = K*N / time).

Shapes (representative of the 27B/9B decode streams):
  down    : K=12288 -> N=4096   (MLP down_proj class, CTA-starved N, split-K)
  wide    : K=5120  -> N=17408  (gate/up + qkv class, large N, split-K)
  stream  : K=4096  -> N=65536  (lm_head-class huge-N stream)

Synthetic random trellises (decode cost is value-independent); suh/svh random
signs. Hot clocks: a dummy-matmul ramp precedes every timed window; timings
are best-of-N passes of R calls each with CUDA events.

Usage (main-tree venv):
  cd <repo>
  MAIN=<main repo>
  ROCM_HOME=$MAIN/.venv/lib/python3.11/site-packages/_rocm_sdk_devel \
  LD_LIBRARY_PATH=$MAIN/.venv/lib/python3.11/site-packages/torch/lib \
  PYTHONPATH=$PWD $MAIN/.venv/bin/python bench/bench_exl3_gemv.py [--json out.json]

Optional: EXL3_SPLITK=off forces the classic (non-split) path for comparison.
"""
import argparse
import json
import time

import torch

from exllamav3.modules.quant.exl3_triton import linear_exl3_triton

SHAPES = [
    ("down", 12288, 4096),
    ("wide", 5120, 17408),
    ("stream", 4096, 65536),
]
BITWIDTHS = [1, 2, 3, 4, 5, 6, 7, 8]
WARMUP_CALLS = 40
TIMED_CALLS = 60
PASSES = 4


def ramp_clocks(sec: float = 3.0):
    """Hold the GPU busy for `sec` so clocks boost. Each matmul is issued
    only after the previous one completes (bounded queue: an async-launch
    loop here would enqueue minutes of backlog the sync then grinds)."""
    a = torch.randn(8192, 8192, device="cuda:0", dtype=torch.bfloat16)
    b = torch.randn(8192, 8192, device="cuda:0", dtype=torch.bfloat16)
    t0 = time.time()
    while time.time() - t0 < sec:
        a = a @ b
        torch.cuda.synchronize()
    torch.cuda.synchronize()


def make_case(bits: int, k_dim: int, n_dim: int, dev):
    trellis = torch.randint(
        0, 65536, (k_dim // 16, n_dim // 16, 256 * bits // 16),
        dtype=torch.int32, device=dev,
    ).to(torch.short)
    suh = torch.sign(torch.randn(k_dim, device=dev)).half()
    svh = torch.sign(torch.randn(n_dim, device=dev)).half()
    x = torch.randn(1, k_dim, dtype=torch.half, device=dev) * 0.1
    return trellis, suh, svh, x


def make_cold_case(bits: int, k_dim: int, n_dim: int, dev, budget_bytes=1.5e9):
    """`copies` distinct trellises so consecutive timed calls never re-read an
    L2-resident weight (a single reused trellis <= 96 MB L2 overstates the
    bandwidth vs in-situ decode, which streams a cold 13 GB/token)."""
    one = make_case(bits, k_dim, n_dim, dev)
    nbytes = one[0].numel() * one[0].element_size()
    copies = max(2, min(16, int(budget_bytes // nbytes)))
    trellises = [one[0]] + [make_case(bits, k_dim, n_dim, dev)[0]
                            for _ in range(copies - 1)]
    return trellises, one[1], one[2], one[3], copies


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default=None, help="optional output JSON path")
    ap.add_argument("--bits", default="1,2,3,4,5,6,7,8",
                    help="comma-separated bit widths to benchmark")
    ap.add_argument("--shapes", default="down,wide,stream",
                    help="comma-separated shape names to benchmark")
    ap.add_argument("--shape", action="append", default=[], metavar="K:N",
                    help="extra custom shape as K:N (repeatable); e.g. the "
                         "real 27B down_proj is --shape 17408:5120")


    args = ap.parse_args()

    dev = torch.device("cuda:0")
    sel_bits = [int(b) for b in args.bits.split(",")]
    sel_shapes = [s for s in SHAPES if s[0] in set(args.shapes.split(","))]
    for spec in args.shape:
        k, n = (int(v) for v in spec.split(":"))
        sel_shapes.append((f"K{k}N{n}", k, n))
    print(torch.cuda.get_device_name(0), flush=True)
    print(f"{'bits':>4} {'shape':>7} {'K':>6} {'N':>6} {'us/call':>9} "
          f"{'GB/s':>7} {'Gweights/s':>11}", flush=True)
    results = []

    for bits in sel_bits:
        for name, k_dim, n_dim in sel_shapes:
            trellises, suh, svh, x, copies = make_cold_case(bits, k_dim, n_dim, dev)
            trellis = trellises[0]
            nbytes = (trellis.numel() * trellis.element_size()
                      + suh.numel() * suh.element_size()
                      + svh.numel() * svh.element_size())
            nweights = k_dim * n_dim

            def call(i=[0]):
                linear_exl3_triton(x, trellises[i[0] % copies], suh, svh, bits,
                                   False, False, k_dim, n_dim, dev, torch.half)
                i[0] += 1

            # long ramp BEFORE the first call: the autotuner benches on the
            # first call and mis-ranks configs at idle/ramping clocks
            ramp_clocks(3.0)
            for _ in range(WARMUP_CALLS):
                call()
            torch.cuda.synchronize()
            ramp_clocks(2.0)

            ev0 = torch.cuda.Event(enable_timing=True)
            ev1 = torch.cuda.Event(enable_timing=True)
            best = float("inf")
            for _ in range(PASSES):
                ev0.record()
                for _ in range(TIMED_CALLS):
                    call()
                ev1.record()
                torch.cuda.synchronize()
                best = min(best, ev0.elapsed_time(ev1) / TIMED_CALLS)

            gbps = nbytes / best / 1e6
            gwps = nweights / best / 1e6
            print(f"{bits:>4} {name:>7} {k_dim:>6} {n_dim:>6} "
                  f"{best * 1000:>9.1f} {gbps:>7.0f} {gwps:>11.0f}", flush=True)
            results.append({
                "bits": bits, "shape": name, "K": k_dim, "N": n_dim,
                "us": round(best * 1000, 1), "copies": copies,
                "GBps": round(gbps, 1),
                "Gweights_per_s": round(gwps, 1),
            })
            del trellises, trellis, suh, svh, x
            torch.cuda.empty_cache()

    if args.json:
        with open(args.json, "w") as f:
            json.dump(results, f, indent=1)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
