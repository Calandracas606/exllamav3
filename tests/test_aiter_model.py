"""Quick end-to-end model inference test with AITER bridge on ROCm gfx1100."""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("AITER_TRITON_ONLY", "1")

import torch
from exllamav3 import Model, Config, Cache, Tokenizer
from exllamav3.generator import Generator
from exllamav3.generator.sampler.presets import DefaultSampler
from exllamav3.aiter_kernels import is_aiter_available

model_dir = sys.argv[1] if len(sys.argv) > 1 else \
    os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca")

print(f"Model dir: {model_dir}")
print(f"AITER available: {is_aiter_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# Load model
config = Config.from_directory(model_dir)
model = Model.from_config(config)
cache = Cache(model, max_num_tokens=4096)
model.load()
tokenizer = Tokenizer.from_config(config)

print(f"Model loaded: {type(config).__name__}, {config.num_hidden_layers} layers")
print(f"Hidden size: {config.hidden_size}")
print()

# Generate
generator = Generator(model, cache, tokenizer)

prompt = "The capital of France is"
print(f"Prompt: {prompt}")
output = generator.generate(
    prompt=prompt,
    sampler=DefaultSampler(),
    max_new_tokens=30,
    add_bos=True,
)
print(f"Output: {output}")
print()

# A slightly longer generation
prompt2 = "Explain what a transformer neural network is in simple terms:"
print(f"Prompt: {prompt2}")
output2 = generator.generate(
    prompt=prompt2,
    sampler=DefaultSampler(),
    max_new_tokens=80,
    add_bos=True,
)
print(f"Output: {output2}")
print()
print("SUCCESS: Model inference works with AITER bridge on ROCm gfx1100")
