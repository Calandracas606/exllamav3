"""
Temporary instrumentation for the paged-prefill shared-memory overrun.

The case to explain: pick_config() measured a prefill tile at 65536 B and accepted it, then the
same tile failed to launch with OutOfResources(131072). The compile-only warmup that
smem.shared_bytes() uses therefore disagrees with the compile the launch performs. This module
records what actually happens instead of inferring it:

  * every compile, tagged warmup (the ladder's probe) or launch, with the footprint Triton
    reports for it
  * Triton's specialization key, so two compiles of one tile config can be told apart
  * the tile and the variant flags the call ran with (q_len, num_splits, resume, split/window/
    staging flags), which is what differs between a probe and a launch
  * on a launch failure, a retry of the ladder using real launches, which answers "which tile
    would actually have worked" rather than "which tile measured small enough"

Off unless EXL3_SMEM_TRACE=1. EXL3_SMEM_TRACE_VERBOSE=1 also prints each event live. The
wrappers call through and re-raise, so behaviour with tracing on is identical to tracing off.
"""

import atexit
import inspect
import os

import torch

_ENABLED = os.environ.get("EXL3_SMEM_TRACE", "0") not in ("0", "")
_VERBOSE = os.environ.get("EXL3_SMEM_TRACE_VERBOSE", "0") not in ("0", "")
_MAX_EVENTS = 4000

_events = []
_picks = []
_seen_specs = {}
_failed = False
_installed = False


def enabled() -> bool:
    return _ENABLED


def _names(kernel):
    try:
        return list(inspect.signature(kernel.fn).parameters)
    except Exception:
        return []


def _describe(kernel, args) -> str:
    """The call's tile and its variant flags: what a probe and a launch can differ by"""
    names = _names(kernel)
    vals = dict(zip(names, args))
    parts = []
    for k in ("q_len", "kv_append_len", "num_splits", "resume", "IS_SPLIT", "NEW_KV",
              "QCK", "QCV", "kv_offset"):
        if k in vals and not isinstance(vals[k], torch.Tensor):
            parts.append(f"{k}={vals[k]}")
    parts.append(f"BLOCK_M={vals.get('BLOCK_M')}")
    parts.append(f"BLOCK_N={vals.get('BLOCK_N')}")
    for k in ("HAS_WINDOW_LEFT", "HAS_WINDOW_RIGHT", "HAS_SINKS", "SOFTCAP", "WIDE_INDEX",
              "CAUSAL", "n_q_heads", "n_kv_heads", "head_dim", "num_pages_per_seq"):
        if k in vals and not isinstance(vals[k], torch.Tensor):
            parts.append(f"{k}={vals[k]}")
    q = vals.get("q")
    if isinstance(q, torch.Tensor):
        parts.append(f"qnumel={q.numel()}%16={q.numel() % 16}")
    return " ".join(parts)


def _spec(kernel, args, kwargs):
    """Triton's own cache key for the call, plus the compile options that are part of it.
    num_warps/num_stages fork the compile, and a probe that uses different values than the
    launch measures a different kernel."""
    nw = kwargs.get("num_warps")
    ns = kwargs.get("num_stages")
    try:
        device = torch.cuda.current_device()
        binder = kernel.device_caches[device][4]
        _, spec, options = binder(*args, **kwargs)
        return (f"num_warps={nw} num_stages={ns} "
                f"opts(nw={getattr(options, 'num_warps', '?')},ns={getattr(options, 'num_stages', '?')}) "
                f"spec={spec}")
    except Exception as e:
        return f"num_warps={nw} num_stages={ns} spec=<unavailable: {type(e).__name__}>"


def _emit(kind, text):
    global _failed
    if len(_events) < _MAX_EVENTS:
        _events.append(f"{kind:12} {text}")
    if kind.endswith("FAIL"):
        _failed = True
        _report(reason = "a compile or launch failed")
    elif _VERBOSE:
        print(f" -- smem-trace: {kind:12} {text}", flush = True)


def instrument(kernel, label):
    """Wrap JITFunction.run: every compile goes through it, both the ladder's warmup=True probes
    and the warmup=False launches that K[grid](...) performs"""
    global _installed
    if not _ENABLED or getattr(kernel, "_exl3_traced", False):
        return
    original = kernel.run

    def traced(*args, **kwargs):
        warmup = bool(kwargs.get("warmup", False))
        desc = _describe(kernel, args)
        spec = _spec(kernel, args, kwargs)
        try:
            compiled = original(*args, **kwargs)
        except Exception as e:
            _emit("LAUNCH-FAIL" if not warmup else "PROBE-FAIL",
                  f"{label} shared=<none> {desc}\n{'':13}{spec}\n{'':13}error={e}")
            raise
        shared = getattr(getattr(compiled, "metadata", None), "shared", None)
        _seen_specs.setdefault(shared, spec)
        if not warmup:
            shared_after = shared
            try:
                if getattr(compiled, "_run", None) is None:
                    compiled._init_handles()
                shared_after = getattr(getattr(compiled, "metadata", None), "shared", None)
                mod = "loaded"
            except Exception as e:
                shared_after = f"<{type(e).__name__}>"
                mod = "load-failed"
            _emit("launch",
                  f"{label} shared_at_compile={shared} shared_{mod}={shared_after} {desc}\n"
                  f"{'':13}{spec}")
        else:
            _emit("warmup", f"{label} shared={shared} {desc}\n{'':13}{spec}")
        return compiled

    kernel.run = traced
    kernel._exl3_traced = True
    if not _installed:
        atexit.register(_report, "process exit")
        _installed = True


def log_pick(name, key, cfg, shared, limit, ladder):
    if not _ENABLED:
        return
    _picks.append(f"{name} key={key} -> cfg={cfg} shared={shared} limit={limit} "
                  f"ladder={ladder}")


def retry_ladder_with_launches(kernel, label, prepare, cfgs):
    """Test each candidate with a real launch. A compile-only warmup can under-report the
    footprint, so the only trustworthy answer to "which tile fits" is what the driver accepts.
    `prepare(cfg)` returns (grid, args, num_warps, num_stages, ...)."""
    rows = []
    for cfg in cfgs:
        try:
            grid, args, num_warps, num_stages = prepare(cfg)[:4]
        except Exception as e:
            rows.append(f"cfg={cfg} PREPARE-FAIL {type(e).__name__}")
            continue
        try:
            compiled = kernel.run(*args, grid = grid, warmup = False,
                                  num_warps = num_warps, num_stages = num_stages)
            shared = getattr(getattr(compiled, "metadata", None), "shared", None)
            rows.append(f"cfg={cfg} LAUNCHES shared={shared}")
        except Exception as e:
            msg = str(e).split(". Reducing")[0]
            rows.append(f"cfg={cfg} FAILS ({msg})")
    return rows


def _norm_spec(spec_text):
    """Strip the num_warps/num_stages prefix so warmups and launches can be matched on the
    Triton cache key alone, which is what decides whether two compiles are the same kernel"""
    marker = "spec="
    return spec_text.split(marker, 1)[1] if marker in spec_text else spec_text


def _comparison_table(events):
    """For each distinct Triton specialization, the footprint the warmup reported next to the
    footprint the launch actually loaded. A row where those differ is the bug."""
    warm = {}
    launch = {}
    for e in events:
        spec = _norm_spec(e.split("\n")[1]) if "\n" in e else ""
        if e.startswith("warmup") and "shared=" in e:
            warm.setdefault(spec, set()).add(e.split("shared=")[1].split()[0])
        elif e.startswith("launch") and "shared_loaded=" in e:
            after = e.split("shared_loaded=")[1].split()[0]
            launch.setdefault(spec, set()).add(after)
    rows = []
    for spec in set(warm) | set(launch):
        w = ",".join(sorted(warm.get(spec, {"-"})))
        s = ",".join(sorted(launch.get(spec, {"-"})))
        flag = "  <-- MISMATCH" if (w != "-" and s != "-" and w != s) else ""
        short = f"{spec[:76]}..." if len(spec) > 76 else spec
        rows.append(f"  warmup={w:12} launch={s:12} {short}{flag}")
    return rows


def _report(reason = ""):
    if not _ENABLED or not (_events or _picks):
        return
    global _failed
    out = ["", "=" * 108,
           f"EXL3_SMEM_TRACE  ({reason})",
           "=" * 108]
    if _picks:
        out.append("pick_config decisions:")
        out += [f"  {p}" for p in _picks]
        out.append("")
    out.append("compiles (warmup = the ladder's probe, launch = the real launch):")
    out += [f"  {e}" for e in _events if not _VERBOSE]
    out.append("")
    table = _comparison_table(_events)
    if table:
        out.append("probe vs launch, per Triton specialization:")
        out += table
        out.append("")
    warm = []
    launch = []
    for e in _events:
        if e.startswith("warmup") and "shared=" in e:
            warm.append(e.split("shared=")[1].split()[0])
        if e.startswith("launch") and "shared_at_compile=" in e:
            after = e.split("shared_loaded=")[1].split()[0]
            launch.append(after)
    out.append(f"warmup footprints:  {sorted(set(warm))}")
    out.append(f"launch footprints:  {sorted(set(launch))}")
    if warm and launch and not (set(warm) & set(launch)):
        out.append(">>> probe and launch footprints are DISJOINT: the ladder measures a compile "
                   "other than the one the launch performs")
    out.append("=" * 108)
    print("\n".join(out), flush = True)
    if _failed:
        _events.clear()
        _picks.clear()
