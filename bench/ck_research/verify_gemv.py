#!/usr/bin/env python3
"""GEMV correctness vs reconstruct_slice reference (b3/b4/b6, several shapes)."""
import sys
import torch

sys.path.insert(0, "/tmp/conversation-worktrees/24b04d07-f055-440e-8f42-cb4f9d94c353/exllamav3")
from exllamav3.ext import exllamav3_ext as ext
torch.set_grad_enabled(False)
torch.manual_seed(0)

CASES = [
    (3, 128, 384), (3, 256, 1024),
    (4, 128, 384), (4, 5120, 17408),
    (6, 128, 384), (6, 1024, 1024), (6, 5120, 248320),
]

allok = True
for bits, K, N in CASES:
    kt, ns = K // 16, N // 16
    tr = torch.randint(-32768, 32767, (kt, ns, 16 * bits), dtype = torch.int16, device = "cuda:0")
    x = (torch.randn(K, device = "cuda:0") / K ** 0.5).to(torch.float16)
    y = torch.zeros(N, dtype = torch.float32, device = "cuda:0")
    ext.exl3_gemv_bench(tr, x, y)
    w = torch.empty((K, N), dtype = torch.half, device = "cuda:0")
    ext.reconstruct_slice(w, tr, bits, False, True, 0)
    ref = (x.view(1, -1) @ w).view(-1)
    d = (y - ref).abs()
    ok = (d <= 0.05).all().item()
    allok = allok and ok
    print(f"bits={bits} K={K} N={N}: maxerr={d.max().item():.6f} nbad(>0.05)={(d > 0.05).sum().item()}/{N}", flush = True)
print("ALL GEMV VERIFICATIONS PASS" if allok else "FAILURES PRESENT")
