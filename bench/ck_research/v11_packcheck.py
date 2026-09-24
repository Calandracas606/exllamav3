#!/usr/bin/env python3
"""v11 a2 packcheck: dump u-packs for thread 0 (nt0=0) and compare against
reconstruct_slice weights of subtile 2 (cols 0 and 8, rows 0,1 / 8,9)."""
import sys
import struct
import torch

sys.path.insert(0, "/tmp/conversation-worktrees/24b04d07-f055-440e-8f42-cb4f9d94c353/exllamav3")
from exllamav3.ext import exllamav3_ext as ext
torch.set_grad_enabled(False)
torch.manual_seed(0)

bits, K, N = 4, 128, 512
kt_, ns = K // 16, N // 16
tr = torch.randint(-32768, 32767, (kt_, ns, 16 * bits), dtype = torch.int16, device = "cuda:0")
x = (torch.randn(K, device = "cuda:0") / K ** 0.5).to(torch.float16)
y = torch.zeros(8192, dtype = torch.float32, device = "cuda:0")
ext.exl3_gemv_bench(tr, x, y)
u = struct.unpack('<11I', y[4096:4107].cpu().numpy().tobytes())

w = torch.empty((K, N), dtype = torch.half, device = "cuda:0")
ext.reconstruct_slice(w, tr, bits, False, True, 0)

def f16_bits(h):
    return int(torch.tensor(h, dtype = torch.half).view(torch.uint16).item())

# thread 0: cA=0, quad idx 0 -> nt0 = 0, second pair = subtiles 2 and 3.
# u packs decode subtile nt0+2 = 2, cols 0 (A) and 8 (B), rows (0,1),(8,9).
checks = [
    ("uA0 (col0 rows0,1)", u[0], 2, 0, 0, 1),
    ("uA1 (col0 rows8,9)", u[1], 2, 0, 8, 9),
    ("uB0 (col8 rows0,1)", u[2], 2, 8, 0, 1),
    ("uB1 (col8 rows8,9)", u[3], 2, 8, 8, 9),
]
for name, pk, sub, col, r0, r1 in checks:
    want = f16_bits(w[r0, sub * 16 + col]) | (f16_bits(w[r1, sub * 16 + col]) << 16)
    print(f"{name}: got={pk:08x} want={want:08x} {'OK' if pk == want else 'MISMATCH'}")
print("words: uc=%08x ur=%08x u0=%08x | s2=%08x %08x %08x %08x" % tuple(u[4:11]))
