#!/usr/bin/env python3
"""Whole-forward equality gate on the REAL 27B model: linear.forward(reconstruct=True) vs
linear.forward(ck) through the BC path — the same tolerance test_qgemm applies (rtol/atol 0.05)."""
import os, sys
import torch

sys.path.insert(0, "/tmp/conversation-worktrees/24b04d07-f055-440e-8f42-cb4f9d94c353/exllamav3")
from exllamav3 import Config, Model
torch.set_grad_enabled(False)

MODEL = "/models/Qwen3.8-27B-exl3-4.00bpw"
KEYS = [
    ("model.language_model.layers.3.self_attn.q_proj", "model.language_model.layers.1.input_layernorm"),
    ("model.language_model.layers.3.self_attn.k_proj", "model.language_model.layers.1.input_layernorm"),
    ("model.language_model.layers.3.self_attn.v_proj", "model.language_model.layers.1.input_layernorm"),
    ("model.language_model.layers.3.self_attn.o_proj", "model.language_model.layers.1.post_attention_layernorm"),
    ("model.language_model.layers.1.mlp.up_proj", "model.language_model.layers.1.post_attention_layernorm"),
    ("model.language_model.layers.1.mlp.gate_proj", "model.language_model.layers.1.post_attention_layernorm"),
    ("model.language_model.layers.1.mlp.down_proj", None),
    ("lm_head", None),
]

cfg = Config.from_directory(MODEL)
model = Model.from_config(cfg)

ok_all = True
for bs in (1, 2, 33):
    for key, _ in KEYS:
        lin = model.find_module(key)
        lin.load(device = "cuda:0")
        torch.manual_seed(0)
        x = torch.randn((1, bs, lin.in_features), dtype = torch.half, device = "cuda:0")
        os.environ["EXL3_CK_GEMV"] = "0"
        ref = lin.forward(x, {"reconstruct": True}).float()
        os.environ["EXL3_CK_GEMV"] = "1"
        ck = lin.forward(x, {"reconstruct": False}).float()
        try:
            torch.testing.assert_close(ck, ref, rtol = 0.05, atol = 0.05)
            res = "OK"
        except AssertionError as e:
            res = "FAIL " + str(e).splitlines()[0][:80]
            ok_all = False
        print(f"bs={bs:3d} {key.split('.')[-1]:10s}: {res}", flush = True)
        lin.unload()

print("REAL-MODEL FORWARD EQUALITY: ALL PASS" if ok_all else "FAILURES PRESENT")
