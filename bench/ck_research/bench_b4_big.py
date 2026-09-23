#!/usr/bin/env python3
"""b4 large-N GEMV bench: K=5120, N configurable (17408 = up_proj 45MB L2-resident;
278528 = lm_head-class 713MB DRAM regime). args: n_subs splits"""
import os, sys, time
import torch

sys.path.insert(0, "/tmp/conversation-worktrees/24b04d07-f055-440e-8f42-cb4f9d94c353/exllamav3")
from exllamav3.ext import exllamav3_ext as ext
torch.set_grad_enabled(False)

n_subs = int(sys.argv[1])
splits = int(sys.argv[2])
K = 5120
k_tiles = K // 16
dev = "cuda:0"

torch.manual_seed(0)
tr = torch.randint(-32768, 32767, (k_tiles, n_subs, 64), dtype = torch.int16, device = dev)
xh = (torch.randn(K, device = dev) / (K ** 0.5)).to(torch.float16).contiguous()
N = n_subs * 16
y = torch.zeros(N, dtype = torch.float32, device = dev)

# correctness on a small slice
NS = min(n_subs, 64)
w = torch.empty((K, NS * 16), dtype = torch.half, device = dev)
ext.reconstruct_slice(w, tr[:, :NS].contiguous(), 4, False, True, 0)
ref = (xh.view(1, -1) @ w).view(-1)
err = 0.0
for _ in range(2):
    y.zero_()
    ext.exl3_gemv_bench(tr, xh.view(K), y)
    torch.cuda.synchronize()
    err = max(err, (y[:ref.numel()] - ref).abs().max().item())

def launch():
    ext.exl3_gemv_bench(tr, xh.view(K), y)

# clock ramp (perf run only — the interleaved loop below keeps clocks honest within-run)
a = torch.randn(2048, 2048, dtype = torch.half, device = dev)
b = torch.randn(2048, 2048, dtype = torch.half, device = dev)
t0 = time.time()
while time.time() - t0 < 5.0:
    c = a @ b
torch.cuda.synchronize()

for _ in range(20):
    launch()
torch.cuda.synchronize()
total = 0.0
iters = 60
for _ in range(iters):
    e0 = torch.cuda.Event(enable_timing = True)
    e1 = torch.cuda.Event(enable_timing = True)
    e0.record()
    launch()
    e1.record()
    torch.cuda.synchronize()
    total += e0.elapsed_time(e1)
us = total * 1000 / iters
mb = tr.numel() * 2 / 1e6
print(f"v6 b4 N={N:7d} trellis {mb:6.0f} MB splits={splits:2d}: {us:9.1f} us  {mb/1e3/(us*1e-6):7.1f} GB/s  maxerr={err:.5f}", flush = True)
