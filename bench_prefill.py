#!/usr/bin/env python3
"""Prefill benchmark + stage profiler: time the prefill gen.iterate() call for
a long prompt (bench_decode-style), then break down per-stage GPU time with
CUDA events around each linear/attention call via module hooks.

Usage:
    python bench_prefill.py [--model 9b|27b] [--prompt-tokens 2048] [--stages]
"""
import argparse
import os
import sys
import time
from collections import defaultdict

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler

MODELS = {
    "9b": "/tmp/hf_cache/models--turboderp--Qwen3.5-9B-exl3/snapshots/01192d8c6d9cbd94f9cf99c0ddb85e9e217ccc01",
    "27b": "/tmp/hf_cache2/models--turboderp--Qwen3.6-27B-exl3/snapshots/04075440a2269126a40e164dbb718700784c39dd",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="27b", choices=list(MODELS))
    parser.add_argument("--prompt-tokens", type=int, default=2048)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=2048)
    parser.add_argument("--stages", action="store_true", help="per-stage event profile")
    args = parser.parse_args()

    model_path = MODELS[args.model]
    print(f"Loading {args.model} from {model_path}...")
    config = Config.from_directory(model_path)
    model = Model.from_config(config)
    # Cache MUST be constructed before model.load() so cache recurrent-layer
    # states (created on meta) are allocated on device during module load.
    cache = Cache(model, max_num_tokens=args.prompt_tokens + 256)
    tokenizer = Tokenizer(config)
    model.load(progressbar=False, device=os.environ.get("DEVICE", "cuda:0"))
    torch.cuda.synchronize()
    print(f"Loaded. VRAM: {torch.cuda.memory_allocated() / 1e9:.1f} GB")

    gen = Generator(model, cache, tokenizer, max_chunk_size=args.chunk)

    # long synthetic prompt: random token ids in normal range
    torch.manual_seed(1234)
    vocab = tokenizer.config.vocab_size

    # warmup (fills autotuners, ramps clocks ~ a full prefill pass)
    g2 = Generator(model, cache=cache, tokenizer=tokenizer, max_chunk_size=args.chunk)
    ids = torch.randint(10, vocab - 100, (1, args.prompt_tokens), dtype=torch.long)
    g2.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=ArgmaxSampler()))
    while g2.num_remaining_jobs():
        g2.iterate()
    torch.cuda.synchronize()

    # timed prefill: same cache, fresh prompt per repeat so every page is cold
    times = []
    for r in range(args.repeats):
        ids = torch.randint(10, vocab - 100, (1, args.prompt_tokens), dtype=torch.long)
        g = Generator(model, cache=cache, tokenizer=tokenizer, max_chunk_size=args.chunk)
        g.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=ArgmaxSampler()))
        t0 = time.perf_counter()
        while g.num_remaining_jobs():
            g.iterate()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        times.append(dt)
        print(f"repeat {r}: prefill {args.prompt_tokens} tokens in {dt:.3f} s "
              f"-> {args.prompt_tokens / dt:.1f} tok/s")
    med = sorted(times)[len(times) // 2]
    print(f"\nMEDIAN prefill: {med * 1000:.1f} ms -> {args.prompt_tokens / med:.1f} tok/s")

    if args.stages:
        print("\n=== per-stage GPU-time profile (one prefill pass, CUDA events) ===")
        stage_times = defaultdict(float)
        stage_counts = defaultdict(int)
        events = {}

        def wrap(name, fn):
            evs = []
            def hooked(*a, **kw):
                s = torch.cuda.Event(enable_timing=True)
                e = torch.cuda.Event(enable_timing=True)
                s.record()
                r = fn(*a, **kw)
                e.record()
                evs.append((s, e, name))
                return r
            return hooked, evs

        # hook every module, aggregate by class; unique key per instance
        n_hooked = 0
        all_events = []
        for mod in model:
            cls = type(mod).__name__
            if cls in ("TransformerModel", "TransformerBlock", "MoeTransformerBlock"):
                continue  # containers add nothing new; children are hooked
            if not hasattr(mod, "forward"):
                continue
            name = cls
            inner = getattr(mod, "inner", None)
            if inner is not None and type(inner).__name__ in ("LinearEXL3", "LinearFP16"):
                name = f"L:{type(inner).__name__[6:].lower()}"
            hooked, evs = wrap(name, mod.forward)
            mod.forward = hooked
            all_events.append((name, evs))
            n_hooked += 1

        c = cache
        ids = torch.randint(10, vocab - 100, (1, args.prompt_tokens), dtype=torch.long)
        g = Generator(model, cache=c, tokenizer=tokenizer, max_chunk_size=args.chunk)
        g.enqueue(Job(input_ids=ids, max_new_tokens=1, sampler=ArgmaxSampler()))
        while g.num_remaining_jobs():
            g.iterate()
        torch.cuda.synchronize()

        for name, evs in all_events:
            for (s, e, n) in evs:
                stage_times[name] += s.elapsed_time(e)
                stage_counts[name] += 1

        total = sum(v for k, v in stage_times.items() if not k.startswith("fp16") or True)
        other = None
        for k in sorted(stage_times, key=stage_times.get, reverse=True):
            print(f"  {k:32s} {stage_times[k]:9.1f} ms  x{stage_counts[k]:4d}")
        print(f"  {'(hooked modules total)':32s} {total:9.1f} ms")
        # compare against full pass time
        print(f"  (hooked {n_hooked} modules)")


if __name__ == "__main__":
    main()
