"""Correctness test for exl3_prefill_fly (M>1) vs the reconstruct reference."""
import os, sys, torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.exl3_gemm_triton import _decode_lut, _get_perm  # noqa (register ops)
import exllamav3.exl3_prefill_fly as pf

dev = os.environ.get("EXL_TEST_DEVICE", "cuda:0")


def make_random_trellis(in_features, out_features, K, dev):
    rows = in_features // 16
    cols = out_features // 16
    packed_size = 256 * K // 16
    encoded = torch.randint(-32768, 32767, (rows, cols, 256), dtype=torch.int16, device=dev)
    packed = torch.zeros((rows, cols, packed_size), dtype=torch.int16, device=dev)
    ext.pack_trellis(packed, encoded.contiguous(), K)
    return packed


def reference(x, trellis, suh, svh, K, mcg, mul1, in_features, out_features):
    x = x.view(-1, in_features)
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    w = torch.empty((in_features, out_features), dtype=torch.half, device=x.device)
    ext.reconstruct(w, trellis, K, mcg, mul1)
    y = torch.empty((x.shape[0], out_features), dtype=torch.half, device=x.device)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y


SHAPES = [
    # (in_f, out_f, K(bits), mcg, mul1)
    (5120, 17408, 4, False, False),
    (5120, 17408, 4, True, False),
    (5120, 17408, 4, False, True),
    (5120, 17408, 6, False, False),
    (5120, 17408, 6, True, False),
    (5120, 17408, 6, False, True),
    (17408, 5120, 4, False, False),
    (17408, 5120, 6, False, False),
    (5120, 248320, 4, False, False),
    (5120, 248320, 6, False, False),
    (2048, 5120, 4, False, False),
]
MS = [2, 128, 513]  # 513 exercises the M-padding fallback

all_ok = True
for in_f, out_f, K, mcg, mul1 in SHAPES:
    torch.manual_seed(42)
    trellis = make_random_trellis(in_f, out_f, K, dev)
    suh = (torch.randint(0, 2, (in_f,), device=dev).to(torch.half) * 2 - 1)
    svh = (torch.randint(0, 2, (out_f,), device=dev).to(torch.half) * 2 - 1)
    for M in MS:
        x = torch.randn(M, in_f, dtype=torch.half, device=dev)
        y_ref = reference(x, trellis, suh, svh, K, mcg, mul1, in_f, out_f)
        try:
            pf._W_CACHE.clear()
            y_fly = pf.exl3_prefill_fly(
                x, trellis, suh, svh, K, mcg, mul1, in_f, out_f, torch.device(dev),
                cache_weights=False,
            )
        except Exception as e:
            import traceback; traceback.print_exc()
            print(f"[{K}b mcg={int(mcg)} mul1={int(mul1)} {in_f}x{out_f} M={M}] EXCEPTION: {type(e).__name__}: {e}")
            all_ok = False
            continue
        try:
            torch.testing.assert_close(y_fly, y_ref, rtol=2e-2, atol=0.5)
            status = "PASS"
        except AssertionError as e:
            status = f"FAIL ({str(e).splitlines()[1] if len(str(e).splitlines())>1 else e})"
            all_ok = False
        diff = (y_fly.float() - y_ref.float()).abs()
        print(f"[{K}b mcg={int(mcg)} mul1={int(mul1)} {in_f:5d}x{out_f:6d} M={M:4d}] {status}  "
              f"max|d|={diff.max().item():.4f} mean|d|={diff.mean().item():.6f}")

print("ALL PASS" if all_ok else "FAILURES PRESENT")
sys.exit(0 if all_ok else 1)

