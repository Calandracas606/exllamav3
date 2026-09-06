"""exl3_mgemm and the fused exl3_moe kernel on the lock-cascade inner.

mgemm: one cooperative launch computes a batch of EXL3 GEMMs against a list of
trellises (blockIdx.z slices the matrices); each column tile is assembled
through the column locks, so z slices never contend. Covers the uniform-width
no-index form, the per-token expert selection with weighted reduction, and
m > 1.

The fused MoE kernel test drives ext.exl3_moe directly with the same argument
shapes the module path passes (block_sparse_mlp.py): expert-sorted token ids,
per-expert counts, silu-gated experts and the weighted scatter-add into a
float32 output.
"""
import os
import pytest
import torch

torch.manual_seed(1234)

from exllamav3.ext import exllamav3_ext as ext

pytestmark = pytest.mark.skipif(
    not torch.version.hip,
    reason = "ROCm exl3_gemm stack",
)

def device():
    return torch.device(os.environ.get("EXL_TEST_DEVICE", "cuda:0"))


def make_trellis(in_f, out_f, K, dev):
    return torch.randint(
        0, 65536,
        (in_f // 16, out_f // 16, 256 * K // 16),
        dtype = torch.int32,
        device = dev,
    ).to(torch.short)


def expert_product(x, t, suh, svh, K, dev):
    """Reference for one expert's projection: input had, hgemm, output had."""
    xh = torch.empty_like(x)
    ext.had_r_128(x, xh, suh, None, 1.0)
    w = torch.empty((x.shape[1], t.shape[1] * 16), dtype = torch.half, device = dev)
    ext.reconstruct(w, t, K, False, True)
    y = torch.empty((x.shape[0], w.shape[1]), dtype = torch.half, device = dev)
    ext.hgemm(xh, w, y)
    ext.had_r_128(y, y, None, svh, 1.0)
    return y


def run_mgemm(x, trellises, suhs, svhs, K, dev, indices = None, weights = None,
              num_tokens = 1):
    bszm = len(trellises)
    m = x.shape[0]
    in_f = x.shape[1]
    out_f = trellises[0].shape[1] * 16
    # With indices the slot count is indices.size(1); C's leading dim must cover
    # the slots (production passes the concurrency-sized fused scratch here)
    slots = indices.numel() if indices is not None else bszm
    y = torch.empty((slots, m, out_f), dtype = torch.half, device = dev)
    ptr_t = torch.tensor([t.data_ptr() for t in trellises], dtype = torch.long, device = dev)
    ptr_suh = torch.tensor([s.data_ptr() for s in suhs], dtype = torch.long, device = dev)
    ptr_svh = torch.tensor([s.data_ptr() for s in svhs], dtype = torch.long, device = dev)
    xh = torch.empty((max(slots, bszm), m, in_f), dtype = torch.half, device = dev)
    ext.exl3_mgemm(
        x.view(1, m, in_f), ptr_t, y, ptr_suh, xh, ptr_svh,
        indices.view(1, -1) if indices is not None else None,
        weights,
        K, -1, False, True,
        -1, -1, 0,
        num_tokens,
        None, None)
    return y


def test_mgemm_uniform():
    dev = device()
    in_f, out_f, mats = 512, 1024, 6
    K = 4
    x = torch.randn(1, in_f, dtype = torch.half, device = dev) * 0.1
    trellises = [make_trellis(in_f, out_f, K, dev) for _ in range(mats)]
    suhs = [torch.randn(1, in_f, dtype = torch.half, device = dev) for _ in range(mats)]
    svhs = [torch.randn(1, out_f, dtype = torch.half, device = dev) for _ in range(mats)]

    y = run_mgemm(x, trellises, suhs, svhs, K, dev)
    assert y.shape == (mats, 1, out_f)
    worst = 0.0
    for j in range(mats):
        ref = expert_product(x, trellises[j], suhs[j], svhs[j], K, dev)
        e = (y[j, 0].float() - ref[0].float()).abs().max().item()
        r = ref.float().abs().max().item()
        worst = max(worst, e / max(r, 1e-6))
    assert worst < 0.03, worst


def test_mgemm_weighted_reduce():
    """Per-token expert selection: slot groups collapse into token rows with weights"""
    dev = device()
    in_f, out_f = 512, 1024
    K = 4
    num_tokens, topk = 2, 3
    bszm = num_tokens * topk
    x = torch.randn(1, in_f, dtype = torch.half, device = dev) * 0.1
    pool = 4
    trellises = [make_trellis(in_f, out_f, K, dev) for _ in range(pool)]
    suhs = [torch.randn(1, in_f, dtype = torch.half, device = dev) for _ in range(pool)]
    svhs = [torch.randn(1, out_f, dtype = torch.half, device = dev) for _ in range(pool)]

    idx = torch.tensor([[0, 1, 2], [1, 3, 0]], dtype = torch.long, device = dev)
    wgt = (torch.rand(bszm, dtype = torch.half, device = dev) * 0.5 + 0.25).view(1, bszm)

    y = run_mgemm(x, trellises, suhs, svhs, K, dev,
                  indices = idx, weights = wgt,
                  num_tokens = num_tokens)
    y = y[:num_tokens]
    assert y.shape == (num_tokens, 1, out_f)

    worst = 0.0
    for tk in range(num_tokens):
        acc = torch.zeros(out_f, dtype = torch.float, device = dev)
        for k in range(topk):
            j = tk * topk + k
            ei = idx[tk, k].item()
            single = expert_product(x, trellises[ei], suhs[ei], svhs[ei], K, dev)
            acc += single[0].float() * float(wgt.view(-1)[j])
        e = (y[tk, 0].float() - acc.half().float()).abs().max().item()
        worst = max(worst, e / max(acc.abs().max().item(), 1e-6))
    assert worst < 0.05, worst


def test_mgemm_m_gt1():
    dev = device()
    in_f, out_f = 512, 768
    K = 4
    x = torch.randn(7, in_f, dtype = torch.half, device = dev) * 0.1
    trellises = [make_trellis(in_f, out_f, K, dev) for _ in range(3)]
    suhs = [torch.randn(1, in_f, dtype = torch.half, device = dev) for _ in range(3)]
    svhs = [torch.randn(1, out_f, dtype = torch.half, device = dev) for _ in range(3)]

    y = run_mgemm(x, trellises, suhs, svhs, K, dev)
    assert y.shape == (3, 7, out_f)
    worst = 0.0
    for j in range(3):
        ref = expert_product(x, trellises[j], suhs[j], svhs[j], K, dev)
        e = (y[j].float() - ref.float()).abs().max().item()
        worst = max(worst, e / max(ref.float().abs().max().item(), 1e-6))
    assert worst < 0.03, worst


# ---------------------------------------------------------------------------
# Fused MoE kernel (ext.exl3_moe), same argument shapes the module path passes
# ---------------------------------------------------------------------------

def test_exl3_moe_fused():
    dev = device()
    H, I = 256, 512          # dims must be 128-multiples for the tile kernels
    K = 4
    E, topk, bsz = 6, 2, 5
    act_limit = 0.0
    MOE_ACT_SILU = 0

    hidden = torch.randn(bsz, H, dtype = torch.half, device = dev) * 0.1
    gates = [make_trellis(H, I, K, dev) for _ in range(E)]
    ups = [make_trellis(H, I, K, dev) for _ in range(E)]
    downs = [make_trellis(I, H, K, dev) for _ in range(E)]
    suh_g = [torch.randn(1, H, dtype = torch.half, device = dev) for _ in range(E)]
    svh_g = [torch.randn(1, I, dtype = torch.half, device = dev) for _ in range(E)]
    suh_u = [torch.randn(1, H, dtype = torch.half, device = dev) for _ in range(E)]
    svh_u = [torch.randn(1, I, dtype = torch.half, device = dev) for _ in range(E)]
    suh_d = [torch.randn(1, I, dtype = torch.half, device = dev) for _ in range(E)]
    svh_d = [torch.randn(1, H, dtype = torch.half, device = dev) for _ in range(E)]

    # Routing: top-k experts per token (as the module path computes them)
    sel = torch.stack([torch.randperm(E, device = dev)[:topk] for _ in range(bsz)]).long()
    wgt = (torch.rand(bsz, topk, device = dev) * 0.5 + 0.25).half()

    flat_e = sel.reshape(-1)
    flat_t = torch.arange(bsz, device = dev).repeat_interleave(topk)
    flat_w = wgt.reshape(-1).half()
    order = flat_e.argsort(stable = True)
    token_sorted = flat_t[order].long()
    weight_sorted = flat_w[order]
    expert_count = torch.bincount(flat_e, minlength = E + 1).long()
    num_active = int((expert_count[:E] > 0).sum())

    C = ext.exl3_moe_max_concurrency(dev.index)
    temp_state_g = torch.zeros(C, 128, H, dtype = torch.half, device = dev)
    temp_state_u = torch.zeros(C, 128, H, dtype = torch.half, device = dev)
    temp_inter_g = torch.zeros(C, 128, I, dtype = torch.half, device = dev)
    temp_inter_u = torch.zeros(C, 128, I, dtype = torch.half, device = dev)

    def ptrs(ts):
        return torch.tensor([t.data_ptr() for t in ts], dtype = torch.long, device = dev)

    output = torch.zeros(bsz, H, dtype = torch.float, device = dev)
    ext.exl3_moe(
        hidden, output, expert_count, token_sorted, weight_sorted,
        temp_state_g, temp_state_u, temp_inter_g, temp_inter_u,
        MOE_ACT_SILU, K, K, K,
        ptrs(gates), ptrs(suh_g), ptrs(svh_g),
        ptrs(ups), ptrs(suh_u), ptrs(svh_u),
        ptrs(downs), ptrs(suh_d), ptrs(svh_d),
        False, True, False, True, False, True,
        act_limit, num_active,
    )
    torch.cuda.synchronize()

    # Reference: per-expert silu(gate(x)) * up(x), down-projected, weighted per token
    ref = torch.zeros(bsz, H, dtype = torch.float, device = dev)
    for tk in range(bsz):
        for k in range(topk):
            e = int(sel[tk, k])
            g = expert_product(hidden[tk:tk + 1], gates[e], suh_g[e], svh_g[e], K, dev)
            u = expert_product(hidden[tk:tk + 1], ups[e], suh_u[e], svh_u[e], K, dev)
            a = (torch.nn.functional.silu(g.float()) * u.float()).half()
            d = expert_product(a, downs[e], suh_d[e], svh_d[e], K, dev)
            ref[tk] += d[0].float() * float(wgt[tk, k])

    rel = (output - ref).abs().max().item() / max(ref.abs().max().item(), 1e-6)
    assert rel < 0.05, rel
