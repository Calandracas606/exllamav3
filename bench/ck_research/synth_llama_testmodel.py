#!/usr/bin/env python3
"""Synthesize a tiny 1-layer Llama EXL3 model for tests/test_qgemm.py."""
import sys, os, json, torch
sys.path.insert(0, "/tmp/conversation-worktrees/24b04d07-f055-440e-8f42-cb4f9d94c353/exllamav3")
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

torch.set_grad_enabled(False)
torch.manual_seed(7)
dev = "cuda:0"
H = 512
cfg = {
    "architectures": ["LlamaForCausalLM"],
    "hidden_size": H, "num_attention_heads": 8, "num_key_value_heads": 4,
    "head_dim": 64, "intermediate_size": 1536, "rms_norm_eps": 1e-5,
    "num_hidden_layers": 1, "vocab_size": 512, "tie_word_embeddings": False,
    "rope_theta": 10000.0, "max_position_embeddings": 128, "hidden_act": "silu",
}

out_dir = "/tmp/synth_llama"
os.makedirs(out_dir, exist_ok = True)

linears = {
    "model.layers.0.self_attn.q_proj": (H, 512),
    "model.layers.0.self_attn.k_proj": (H, 256),
    "model.layers.0.self_attn.v_proj": (H, 256),
    "model.layers.0.self_attn.o_proj": (512, H),
    "model.layers.0.mlp.up_proj": (H, 1536),
    "model.layers.0.mlp.gate_proj": (H, 1536),
    "model.layers.0.mlp.down_proj": (1536, H),
    "lm_head": (H, 512),
}

tensors = {}
# plain fp16 tensors
tensors["model.embed_tokens.weight"] = (torch.randn(512, H) * 0.02).half()
tensors["model.layers.0.input_layernorm.weight"] = (torch.randn(H) * 0.02 + 1).half()
tensors["model.layers.0.post_attention_layernorm.weight"] = (torch.randn(H) * 0.02 + 1).half()
tensors["model.norm.weight"] = (torch.randn(H) * 0.02 + 1).half()

for i, (key, (k, n)) in enumerate(linears.items()):
    w = (torch.randn(k, n) * 0.02).float()
    quant_args = {
        "K": 4, "seed": i, "devices": [dev], "device_ratios": None,
        "apply_out_scales": False, "debug_dir": None, "mul1": True,
    }
    H_data = {
        "H": torch.empty(k, k, device = "meta"),
        "L": None,
        "device": dev,
    }
    try:
        res = quantize_exl3(w, H_data, quant_args, return_weight_q = False)
    except Exception as e:
        print(f"FAIL {key}: {type(e).__name__}: {e}")
        raise
    out_tensors = res[-1]
    for tname, t in out_tensors.items():
        tt = t
        if tt.dtype == torch.uint16: tt = tt.view(torch.int16)
        if tt.dtype == torch.uint32: tt = tt.view(torch.int32)
        tensors[f"{key}.{tname}"] = tt.cpu().contiguous()
    print(f"quantized {key}: " + ", ".join(f"{n2}:{tuple(t.shape)}" for n2, t in out_tensors.items()), flush = True)

json.dump(cfg, open(f"{out_dir}/config.json", "w"), indent = 2)
from safetensors.torch import save_file
save_file(tensors, f"{out_dir}/model.safetensors")
print("MODEL WRITTEN to", out_dir)
