#!/usr/bin/env python3
"""b4 large-N GEMV, GRAPH-REPLAY regime (production decode condition): capture N
iterations of the GEMV into one CUDA graph, replay, report per-iteration time. This
removes per-call host launch overhead from the measurement — matching how the model
actually runs (whole-step graphs). args: n_subs splits"""
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
ITER = 20

torch.manual_seed(0)
tr = torch.randint(-32768, 32767, (k_tiles, n_subs, 64), dtype = torch.int16, device = dev)
xh = (torch.randn(K, device = dev) / (K ** 0.5)).to(torch.float16).contiguous()
N = n_subs * 16
y = torch.zeros(N, dtype = torch.float32, device = dev)

NS = min(n_subs, 64)
w = torch.empty((K, NS * 16), dtype = torch.half, device = dev)
ext.reconstruct_slice(w, tr[:, :NS].contiguous(), 4, False, True, 0)
ref = (xh.view(1, -1) @ w).view(-1)
y.zero_()
ext.exl3_gemv_bench(tr, xh.view(K), y)
torch.cuda.synchronize()
err = (y[:ref.numel()] - ref).abs().max().item()

def launch():
    ext.exl3_gemv_bench(tr, xh.view(K), y)

a = torch.randn(2048, 2048, dtype = torch.half, device = dev)
b = torch.randn(2048, 2048, dtype = torch.half, device = dev)
t0 = time.time()
while time.time() - t0 < 6.0:
    c = a @ b
torch.cuda.synchronize()

for _ in range(20):
    launch()
torch.cuda.synchronize()

g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    for _ in range(ITER):
        launch()
torch.cuda.synchronize()

best = 1e9
for _ in range(3):
    e0 = torch.cuda.Event(enable_timing = True)
    e1 = torch.cuda.Event(enable_timing = True)
    g.replay()
    torch.cuda.synchronize()
    e0.record()
    g.replay()
    e1.record()
    torch.cuda.synchronize()
    best = min(best, e0.elapsed_time(e1) * 1000 / ITER)

mb = tr.numel() * 2 / 1e6
print(f"b4 {mb:4.0f}MB splits={splits} GRAPH-REPLAY: {best:7.1f}us  {mb/1e3/(best*1e-6):6.1f} GB/s  maxerr={err:.5f}")
