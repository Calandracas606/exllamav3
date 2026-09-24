#!/usr/bin/env python3
"""Dump all 32 asm packs (4 rh x 8) + words and validate against the
reconstructed weight matrix: pack (lo,hi) must equal
(w[row0, col], w[row1, col]) as f16 bits, per the v9 mapping."""
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
raw = y[4096:4128].cpu().numpy().tobytes()
packs = struct.unpack('<32I', raw)
words = struct.unpack('<16I', y[4128:4144].cpu().numpy().tobytes())

w = torch.empty((K, N), dtype = torch.half, device = "cuda:0")
ext.reconstruct_slice(w, tr, bits, False, True, 0)

def f16_bits(h):
    return int(torch.tensor(h, dtype = torch.half).view(torch.uint16).item())

# thread t=0: cA=0, pair=0, nt0=0, subtile pair (0, 1), kt = kt0 = 0.
# pack slots per rh: [p0A0(p0A0 shifts 28,24), p0A1(20,16), p0B0(12,8), p0B1(4,0), p1A0.., p1A1.., p1B0.., p1B1..]
# chain mapping: colA c=0/8 (out cols 0 and 8), rows for j0 (2rh,2rh+1), j1 (2rh+8,2rh+9)
ok_all = True
for rh in range(4):
    p = packs[rh * 8: rh * 8 + 8]
    checks = [
        (p[0], 0, 0, 2 * rh,     2 * rh + 1),      # p0A0: sub0 colA j0
        (p[1], 0, 0, 2 * rh + 8, 2 * rh + 9),      # p0A1: sub0 colA j1
        (p[2], 0, 8, 2 * rh,     2 * rh + 1),      # p0B0: sub0 colB j0
        (p[3], 0, 8, 2 * rh + 8, 2 * rh + 9),      # p0B1: sub0 colB j1
        (p[4], 1, 0, 2 * rh,     2 * rh + 1),      # p1A0: sub1 colA j0
        (p[5], 1, 0, 2 * rh + 8, 2 * rh + 9),      # p1A1
        (p[6], 1, 8, 2 * rh,     2 * rh + 1),      # p1B0
        (p[7], 1, 8, 2 * rh + 8, 2 * rh + 9),      # p1B1
    ]
    for pk, sub, col, r0, r1 in checks:
        want = f16_bits(w[r0, sub * 16 + col]) | (f16_bits(w[r1, sub * 16 + col]) << 16)
        if pk != want:
            ok_all = False
            print(f"rh={rh} sub={sub} col={col} rows=({r0},{r1}): pack={pk:08x} want={want:08x}"
                  f"  lo_want={f16_bits(w[r0, sub*16+col]):04x} hi_want={f16_bits(w[r1, sub*16+col]):04x}")
print("PACKS ALL MATCH RECONSTRUCTION" if ok_all else "PACK MISMATCHES FOUND")
