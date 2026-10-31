# Debug branch: paged-prefill shared-memory overrun

Instrumentation only — no behaviour change. Every wrapper calls through and re-raises, and the
tracer is a no-op unless `EXL3_SMEM_TRACE` is set.

## The case to explain

On a 3x7900XTX server (TP=3, Qwen3.8-27B, fp16 cache), prefill fails with:

    out of resource: shared memory, Required: 131072, Hardware limit: 65536

but the ladder reports the tile it picked as fitting:

    -- smem: paged_attn_prefill (256, 256, 0, 0, 0, True, ...) config (128, 32, 8, 2):
       65536 B (fits 65536 B)

So the compile the ladder measured reported 65536 B, and the compile the launch performed
required 131072 B. The two paths disagree.

## What this records

* every compile of the prefill kernel, tagged `warmup` (the ladder's probe) or `launch`
* the footprint at compile, and after `_init_handles()` loads the module
* Triton's own specialization key, so two compiles of one tile can be told apart
* the call's variant flags (`q_len`, `num_splits`, `resume`, staging, window, heads)
* on a launch failure: a retry of every ladder candidate using **real launches** rather than
  warmups, i.e. which tiles actually load

## Run

    EXL3_SMEM_TRACE=1 ... python -m tabbyAPI
    EXL3_SMEM_TRACE=1 EXL3_SMEM_TRACE_VERBOSE=1 ...   # also print each event live

A report prints at process exit. In a TP worker, the report comes from the child process.

## What to look for

The `probe vs launch, per Triton specialization` table. A row marked `MISMATCH` — or a
`launch=` footprint of `-` where the warmup had a value — is the bug. Also read the
`real-launch verdict per candidate` line: it names the tiles that actually load.

If instead the failing launch shows a *different specialization* than the probed one, then the
ladder picked a config for one variant and the launch ran another.
