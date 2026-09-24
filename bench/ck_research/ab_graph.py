#!/usr/bin/env python3
"""Graph-mode A/B: capture lin.forward (ck vs native) in a private CUDA graph and time
REPLAYS (production decode regime); eager via per-call events. One mode per process."""
import os, sys, time
import torch

sys.path.insert(0, "/tmp/conversation-worktrees/24b04d07-f055-440e-8f42-cb4f9d94c353/exllamav3")
from exllamav3 import Config, Model

torch.set_grad_enabled(False)
MODEL, KEY, MODE = sys.argv[1], sys.argv[2], sys.argv[3]
cfg = Config.from_directory(MODEL)
model = Model.from_config(cfg)
lin = model.find_module(KEY)
lin.load(device = "cuda:0")
inner = lin.inner
K, N = inner.in_features, inner.out_features
tr_bytes = inner.trellis.numel() * 2

x = torch.randn((1, 1, K), dtype = torch.float16, device = "cuda:0")

os.environ["EXL3_CK_GEMV"] = "1" if MODE == "ck" else "0"

def bench_mode():
    for _ in range(10):
        lin.forward(x, {"reconstruct": False})
    torch.cuda.synchronize()

    iters = 120
    total = 0.0
    for _ in range(iters):
        e0 = torch.cuda.Event(enable_timing = True)
        e1 = torch.cuda.Event(enable_timing = True)
        e0.record()
        lin.forward(x, {"reconstruct": False})
        e1.record()
        torch.cuda.synchronize()
        total += e0.elapsed_time(e1)
    eager_us = total * 1000.0 / iters

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for _ in range(20):
            lin.forward(x, {"reconstruct": False})
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    e0 = torch.cuda.Event(enable_timing = True)
    e1 = torch.cuda.Event(enable_timing = True)
    e0.record()
    for _ in range(20):
        g.replay()
    e1.record()
    torch.cuda.synchronize()
    graph_us = e0.elapsed_time(e1) * 1000 / (20 * 20)
    return eager_us, graph_us

# clock ramp
a = torch.randn(2048, 2048, dtype = torch.half, device = "cuda:0")
b = torch.randn(2048, 2048, dtype = torch.half, device = "cuda:0")
t0 = time.time()
while time.time() - t0 < 5.0:
    c = a @ b
torch.cuda.synchronize()

e, gph = bench_mode()
print(f"RESULT {MODE} {KEY}: eager {e:8.1f} us ({tr_bytes/(e*1e3):6.1f} GB/s)   graph-replay {gph:8.1f} us ({tr_bytes/(gph*1e3):6.1f} GB/s)", flush = True)
