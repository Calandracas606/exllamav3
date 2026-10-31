"""
Shared-memory budget for the Triton attention kernels.

The tile configs in this package are sized for the ~100 KB of dynamic shared memory that Ampere
and later grant a block. Older parts grant less (Turing: 64 KB), and Triton does not shrink a
tile on its own: a kernel whose compiled footprint exceeds the limit fails at launch with
OutOfResources. Guessing per-architecture tiles is unreliable because the footprint of the same
config differs between architectures (no cp.async pipelining below sm_80, for one), so each
launch site lists a ladder of configs instead -- the stock config first, then progressively
smaller tiles -- and pick_config() walks it with compile-only warmups, taking the first whose
measured footprint fits the device. Picks are cached per launch key, so the walk is paid once
per kernel family, and on a full-budget device the first candidate always fits, which is exactly
the compile the launch would have done anyway.
"""

import os
import torch

_limit = {}
# Test override: pretend the device grants at most this many bytes, to exercise the ladders on
# any GPU (the picks then reflect this device's footprints, not the smaller device's)
_env_limit = int(os.environ.get("EXL3_TRITON_SMEM_LIMIT", "0") or "0")
# EXL3_TRITON_SMEM_DEBUG=1 prints every pick with its measured footprint
_debug = bool(os.environ.get("EXL3_TRITON_SMEM_DEBUG"))
_picks = {}
# Per pick key, every candidate the walk actually probed and what the probe said, kept so a launch
# that fails anyway can report the decision that admitted it instead of only the loader's complaint
_probe_log = {}


class NoFittingConfig(RuntimeError):
    """No candidate in a ladder fits the device's shared memory. Callers with an alternative
    formulation of the same computation catch this and take it; the miss is cached, so the
    probe cost is paid once per launch key."""


def _probe_signature(args, num_warps, num_stages) -> str:
    """The arguments one launch-shaped call is made with, in the form the compiler and loader size
    their structures from.

    Tensor dtype and shape decide how the kernel specialises and how much it needs; the scalars are
    passed through because the constexpr ones are part of the compile key. Recording this next to the
    measured footprint is what makes a report from another machine comparable: if the probe and the
    launch recorded different signatures for one config, they compiled different kernels and the
    footprints are not comparable at all.
    """
    tensors = []
    for a in args:
        if isinstance(a, torch.Tensor):
            tensors.append("x".join(str(s) for s in a.shape) + f":{str(a.dtype).replace('torch.', '')}")
        else:
            tensors.append(str(a))
    return f"nw={num_warps} ns={num_stages} " + "|".join(tensors)


NOTE_REASON = "smem-probe-miss"


def _get_probe_log(name: str, key):
    """The walk recorded under this launch key: (probed pairs, probe signatures, launch signature)."""
    return _probe_log.get((name, key))


def _record_walk(name: str, key, log, sigs) -> None:
    """Keep the walk for a key, preserving any launch signature already recorded for it."""
    prev = _probe_log.get((name, key))
    launch_sig = prev[2] if prev else None
    _probe_log[(name, key)] = (log, dict(sigs), launch_sig)


def _record_launch(name: str, key, args, num_warps, num_stages) -> None:
    """Record the arguments the launch is about to use, next to what the walk measured.

    The walk measures a config; the launch then launches it with some argument list. prepare() is
    written so those are the same list. When they are not -- a buffer of another dtype, a different
    head count -- the walk measured a different kernel, which is exactly the case where a probe
    reports a footprint the launch then exceeds.
    """
    sig = _probe_signature(args, num_warps, num_stages)
    prev = _probe_log.get((name, key))
    walk, sigs = (prev[0], prev[1]) if prev else (None, {})
    _probe_log[(name, key)] = (walk, sigs, sig)


def _note_probe_miss(err: Exception, name: str, key, device, cfg) -> bool:
    """Attach the walk that approved `cfg`, and the arguments the launch used, to a launch-time
    OutOfResources, in place.

    The loader reports only what it wanted. The walk reports what it measured per candidate, and the
    recorded signatures say whether the probe compiled the same kernel the launch did. Those three
    separate the cases a report from hardware we do not have cannot otherwise distinguish: the walk
    measured this config as fitting, or never measured it, or measured it with other arguments.

    Returns whether the note was attached; the exception is unchanged otherwise.
    """
    add_note = getattr(err, "add_note", None)
    if add_note is None:
        return False
    limit = smem_limit(device)
    walk, sigs, launch_sig = _probe_log.get((name, key), (None, {}, None))
    lines = [f"[{NOTE_REASON}] {name}: config {cfg} on a device granting {limit} B"]
    if not walk:
        lines.append("  probe recorded nothing for this launch key: the config came from the pick "
                     "cache, or the key differs between the pick and this launch")
    else:
        lines.append("  probe measured:")
        for cand, need in walk:
            lines.append(f"    {cand}: {need} B")
        if cfg not in [c for c, _ in walk]:
            lines.append(f"  {cfg} was never probed; the walk picked from {[c for c, _ in walk]}")
        elif all(need <= limit for _, need in walk):
            lines.append("  every probed candidate measured as fitting, so for at least one of them "
                         "the compiled footprint and the footprint the loader accepts disagree")
        else:
            picked = dict(walk).get(cfg)
            lines.append(f"  {cfg} measured {picked} B, which fits {limit} B, yet the load failed")
        probe_sig = sigs.get(cfg)
        if probe_sig is not None and launch_sig is not None:
            if probe_sig == launch_sig:
                lines.append("  probe and launch arguments are identical, so the two compiles of "
                             "this config should be the same kernel")
            else:
                lines.append("  probe and launch arguments DIFFER -- they compiled different kernels:")
                lines.append(f"    probe:  {probe_sig}")
                lines.append(f"    launch: {launch_sig}")
    add_note("\n".join(lines))
    return True


def _dev_index(device) -> int:
    idx = device.index if hasattr(device, "index") else device
    return torch.cuda.current_device() if idx is None else idx


def smem_limit(device) -> int:
    """Dynamic shared memory per block the device grants (the opt-in maximum), in bytes."""
    idx = _dev_index(device)
    lim = _limit.get(idx)
    if lim is None:
        props = torch.cuda.get_device_properties(idx)
        lim = getattr(props, "shared_memory_per_block_optin", 0)
        if not lim and torch.version.hip:
            # ROCm has no opt-in tier (the property is absent): the per-block LDS is the limit
            lim = props.shared_memory_per_block
        if not lim:
            lim = 64 * 1024 if props.major < 8 else 96 * 1024
        if _env_limit:
            lim = min(lim, _env_limit)
        _limit[idx] = lim
    return lim


def pick_config(device, name: str, key, candidates, probe):
    """First config in `candidates` whose compiled kernel fits the device's shared memory.

    probe(cfg) compiles the kernel for that config without launching it and returns the
    footprint in bytes (see shared_bytes). `key` must cover every launch constant the footprint
    depends on; the pick is cached per (device architecture, limit, name, key). Raises
    NoFittingConfig when no candidate fits (also cached).
    """
    idx = _dev_index(device)
    limit = smem_limit(idx)
    full_key = (torch.cuda.get_device_capability(idx), limit, name, key)
    cfg = _picks.get(full_key, ...)
    if cfg is None:
        raise NoFittingConfig(f"{name}: no tile configuration fits the device's {limit} B of dynamic shared memory")
    if cfg is not ...:
        return cfg
    need = None
    log = []
    sigs = {}
    for cand in candidates:
        measured = probe(cand)
        # A probe may return (footprint, signature); the signature is what the compile keys on
        need, sig = measured if isinstance(measured, tuple) else (measured, None)
        log.append((cand, need))
        if sig is not None:
            sigs[cand] = sig
        if _debug:
            print(f" -- smem: {name} {key} config {cand}: {need} B ({'fits' if need <= limit else 'over'} {limit} B)", flush = True)
        if need <= limit:
            _picks[full_key] = cand
            _record_walk(name, key, log, sigs)
            return cand
    _picks[full_key] = None
    _record_walk(name, key, log, sigs)
    raise NoFittingConfig(
        f"{name}: no tile configuration fits the device's {limit} B of dynamic shared memory "
        f"(the smallest candidate needs {need} B)"
    )


def shared_bytes(kernel, args, **kwargs) -> int:
    """Compile-only warmup of a JIT kernel with the launch's real arguments; the launch grid does
    not affect the footprint. The compile is cached by Triton, so the eventual launch reuses it."""
    return kernel.run(*args, grid = (1,), warmup = True, **kwargs).metadata.shared


def tile_ladder(block_m: int, block_n: int, num_warps: int, num_stages: int,
                min_m: int = 16, min_n: int = 16):
    """Stock (block_m, block_n, num_warps, num_stages) first, then halving q tiles at the same
    kv tile, then the same at a halved kv tile. Warps follow the q tile so each warp keeps whole
    16-row MMA tiles."""
    cands = [(block_m, block_n, num_warps, num_stages)]
    bn = block_n
    while bn >= min_n:
        bm = block_m
        while bm >= min_m:
            c = (bm, bn, max(1, min(num_warps, bm // 16)), num_stages)
            if c not in cands:
                cands.append(c)
            bm //= 2
        bn //= 2
    return cands


def halving_ladder(value: int, minimum: int = 16):
    """value, value / 2, ... down to `minimum` (single-parameter ladders such as a kv tile). The
    stock value always leads, even when it is already below the minimum."""
    cands = [value]
    while value // 2 >= minimum:
        value //= 2
        cands.append(value)
    return cands
