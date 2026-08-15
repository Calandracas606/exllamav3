"""Needle-in-a-haystack test for exllamav3 on ROCm with AITER bridge.

Generates a long context prompt with a hidden reference value, asks the model
to retrieve it, and checks correctness. Tests multiple context lengths.
"""

import pytest

if __name__ != "__main__":
    pytest.skip(
        "script-style test; run directly: python tests/test_niah.py [model_dir]",
        allow_module_level=True,
    )

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("AITER_TRITON_ONLY", "1")

import torch
from exllamav3 import Model, Config, Cache, Tokenizer
from exllamav3.generator import Generator
from exllamav3.generator.sampler.presets import DefaultSampler
from exllamav3.aiter_kernels import is_aiter_available

# Import the haystack prompt builder from the examples
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "examples"))
from chat_util import make_haystack_prompt

model_dir = sys.argv[1] if len(sys.argv) > 1 else \
    "/tmp/hf_cache2/models--turboderp--Qwen3.6-27B-exl3/snapshots/04075440a2269126a40e164dbb718700784c39dd"

# Context lengths to test (in tokens)
target_lengths = [int(x) for x in sys.argv[2:]] if len(sys.argv) > 2 else [500, 1000, 2000, 4000]

print(f"Model dir: {model_dir}")
print(f"AITER available: {is_aiter_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Load model
config = Config.from_directory(model_dir)
max_cache = (max(target_lengths) + 512 + 255) // 256 * 256
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=max_cache)
model.load()
tokenizer = Tokenizer.from_config(config)

print(f"Model loaded: {type(config).__name__}, {config.num_hidden_layers} layers")
print(f"Cache size: {max_cache} tokens")
print()

generator = Generator(model, cache, tokenizer)

# Run needle-in-haystack at each context length
print("=" * 70)
print("NEEDLE-IN-A-HAYSTACK TEST")
print("=" * 70)

num_pass = 0
num_total = 0

for target_tokens in target_lengths:
    user_prompt, ref_value, actual_tokens = make_haystack_prompt(target_tokens, tokenizer)
    print(f"\nTarget: {target_tokens} tokens | Actual: {actual_tokens} tokens | Ref value: {ref_value}")
    print("-" * 50)

    output = generator.generate(
        prompt=user_prompt,
        sampler=DefaultSampler(),
        max_new_tokens=64,
        add_bos=True,
    )

    # Check if the reference value appears in the output
    found = ref_value in output
    num_total += 1
    status = "✅ PASS" if found else "❌ FAIL"
    if found:
        num_pass += 1

    # Show a trimmed version of the output
    output_trimmed = output[:200].replace("\n", " ")
    print(f"  Output: {output_trimmed}...")
    print(f"  Result: {status}")

print()
print("=" * 70)
print(f"SUMMARY: {num_pass}/{num_total} passed")
print("=" * 70)
