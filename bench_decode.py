#!/usr/bin/env python3
"""Benchmark decode speed on ROCm: total time per token and dispatch time (no sync).

Usage:
    python bench_decode.py [--model 9b|27b] [--num-tokens 1024] [--warmup 64]
"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exllamav3 import Config, Model, Cache, Tokenizer, Generator, Job, ArgmaxSampler

MODELS = {
    "9b": "/tmp/hf_cache/models--turboderp--Qwen3.5-9B-exl3/snapshots/01192d8c6d9cbd94f9cf99c0ddb85e9e217ccc01",
    "27b": "/tmp/hf_cache2/models--turboderp--Qwen3.6-27B-exl3/snapshots/04075440a2269126a40e164dbb718700784c39dd",
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="9b", choices=list(MODELS))
    parser.add_argument("--num-tokens", type=int, default=1024)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--cache", type=int, default=4096)
    parser.add_argument("--prompt", default="The capital of France is")
    parser.add_argument("--max-new", type=int, default=0, help="if set, stop after this many tokens and print text")
    parser.add_argument("--print-text", action="store_true")
    args = parser.parse_args()

    model_path = MODELS[args.model]
    print(f"Loading {args.model} from {model_path}...")
    config = Config.from_directory(model_path)
    model = Model.from_config(config)
    cache = Cache(model, max_num_tokens=args.cache)
    tokenizer = Tokenizer(config)
    model.load(progressbar=False, device="cuda:0")
    torch.cuda.synchronize()
    vram = torch.cuda.memory_allocated() / 1e9
    print(f"Loaded. VRAM: {vram:.1f} GB")

    gen = Generator(model, cache, tokenizer)
    ids = tokenizer.encode(args.prompt, add_bos=True)
    print(f"Prompt tokens: {ids.shape[-1]}")

    # Coherence check + warmup
    gen.enqueue(Job(input_ids=ids, max_new_tokens=args.warmup, sampler=ArgmaxSampler()))
    t0 = time.perf_counter()
    while gen.num_remaining_jobs():
        gen.iterate()
    torch.cuda.synchronize()
    print(f"Warmup ({args.warmup} tokens): {time.perf_counter() - t0:.2f} s")

    # Timed decode: measure both total (with sync) and dispatch-only (no sync) per token
    gen2 = Generator(model, cache, tokenizer)
    gen2.enqueue(Job(input_ids=ids, max_new_tokens=args.num_tokens, sampler=ArgmaxSampler()))
    gen2.iterate()  # prefill
    torch.cuda.synchronize()

    dispatch_times = []
    t_all = time.perf_counter()
    last_sync = time.perf_counter()
    sync_times = []
    n = 0
    while gen2.num_remaining_jobs() and n < args.num_tokens:
        t0 = time.perf_counter()
        gen2.iterate()
        dispatch_times.append(time.perf_counter() - t0)
        n += 1
        if n % 128 == 0:
            torch.cuda.synchronize()
            now = time.perf_counter()
            sync_times.append((n, (now - last_sync) / 128))
            last_sync = now
    torch.cuda.synchronize()
    total = time.perf_counter() - t_all

    # Dispatch time is lower bound of CPU cost; windowed sync time gives steady-state tok/s
    steady = [t for _, t in sync_times]
    print(f"\n{'=' * 60}")
    print(f"Model: {args.model}, tokens: {n}")
    print(f"total wall: {total:.2f}s  -> {n / total:.2f} tok/s overall (incl. python)")
    if steady:
        med = sorted(steady)[len(steady) // 2]
        print(f"steady-state windowed: {min(steady) * 1000:.2f} - {max(steady) * 1000:.2f} ms/token (median {med * 1000:.2f})")
        print(f"  -> steady tok/s (median window): {1 / med:.2f}")
    print(f"dispatch-only avg: {sum(dispatch_times) / len(dispatch_times) * 1000:.2f} ms/token")
    print(f"dispatch-only median: {sorted(dispatch_times)[len(dispatch_times) // 2] * 1000:.2f} ms/token")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

