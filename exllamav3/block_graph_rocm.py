"""Whole-decode-step CUDA graph capture for ROCm.

On CUDA the C++ BC_* classes batch each module's kernels into per-module CUDA
graphs. On ROCm those sources are excluded from the build, so a decode step
pays full Python dispatch for every module of every layer: for a 64-layer
hybrid model that is ~77 ms of host work against ~5.5 ms of GPU work.

This module scales the pattern from ``bc_rocm.BC_LinearEXL3`` up to the whole
decode step: one ``torch.cuda.CUDAGraph`` per (cache, block-table-width,
recurrent-slots, last_tokens_only) key covering every module from the first
transformer block to the LM head. A replay costs one launch instead of
thousands.

State handling (everything the graph must read or write through stable
addresses):

- KV cache: the Triton paged-decode kernels write new K/V rows through the
  block table and cache_seqlens **tensors**, and the cache tensors themselves
  (``CacheLayer_fp16.k/v``) are allocated once per cache. Both are captured
  directly; per-token addressing is read from device memory at replay time.
- GDN recurrent state: ``GDNLayerState.conv_state``/``recurrent_state`` are
  persistent per-cache tensors updated in place by the C++ kernels. Captured
  directly.
- ``cache_seqlens`` / ``positions`` / ``block_table``: the generator supplies
  these as pinned host staging tensors that would otherwise be uploaded to a
  fresh device tensor on every forward. They are copied into persistent device
  buffers here, and the params dict is pointed at those buffers, so the
  captured kernels see fresh values at each replay without any recapture.
- Input IDs / embedding: the token embedding table lives on the CPU
  (``prefer_cpu``), so the embedding lookup runs outside the graph; its
  output is staged through a persistent device buffer that feeds the first
  block.

Everything is ROCm-only (the hook imported by ``model.py`` is a no-op on CUDA)
and can be disabled with ``EXL3_BLOCK_GRAPHS=0``. Prefill and any
non-(bsz=1, seqlen=1) forward always take the regular path, as do draft/
verification forwards (``recurrent_history``), non-causal spans and TP loads.
Capture failures disable the path permanently and fall back to regular decode.
"""
from __future__ import annotations

import os
import weakref

import torch

# Kill switch: EXL3_BLOCK_GRAPHS=0 disables this path entirely
enabled = bool(torch.version.hip) and os.environ.get("EXL3_BLOCK_GRAPHS", "1") != "0"

# Number of uncaptured decode steps before the graph is captured. Must be
# enough for all Triton autotuning (conv, GDN, attention, GEMV) to settle on
# the bsz=1/seqlen=1 shape and for the per-linear BC graphs to warm up.
WARMUP_STEPS = 8

# Extra warmup iterations on the side stream immediately before capture
CAPTURE_WARMUP_ITERS = 1

# Upper bound on simultaneously live graphs (per model); the key changes only
# when the block-table width bucket or recurrent slots change, so in practice
# one or two graphs exist at a time
MAX_GRAPHS = 4

_managers: "weakref.WeakKeyDictionary" = weakref.WeakKeyDictionary()


def _bc_capture_suppress(on: bool):
    """While capturing the whole-step graph, disable the per-linear BC graphs so
    their kernels are recorded directly as nodes instead of nested graph replays."""
    from . import bc_rocm
    bc_rocm.capture_suppress = on


def _eligible(model, input_ids: torch.Tensor, params: dict) -> bool:
    if not enabled:
        return False
    if input_ids.shape != (1, 1) or input_ids.dtype != torch.long:
        return False
    if params.get("attn_mode") != "flash_attn":
        return False
    if params.get("cache") is None or params.get("block_table") is None:
        return False
    if params.get("recurrent_history"):
        return False
    if params.get("prefill"):
        return False
    if params.get("non_causal_spans") is not None:
        return False
    if params.get("sim_kvq") is not None:
        return False
    if getattr(model, "loaded_tp", False):
        return False
    if getattr(model.config, "moe_cpu_hosts", None):
        return False
    if getattr(model, "first_block_idx", None) is None:
        return False
    if params.get("indexed_embeddings"):
        return False
    if getattr(model.config, "layer_map", None):
        return False
    return True


class _StepGraphManager:
    """Per-model static buffers and captured decode-step graphs.

    Graphs are keyed by everything that changes captured kernel parameters or
    tensor addresses: cache identity, block-table width, recurrent slots and
    last_tokens_only."""

    def __init__(self, model):
        self.model_ref = weakref.ref(model)
        self.device = model.modules[model.first_block_idx].device
        self.hidden_size = model.config.hidden_size

        self.static_hidden: torch.Tensor | None = None      # [1, 1, hidden], emb dtype
        self.static_cache_seqlens: torch.Tensor | None = None  # [1] int32
        self.static_positions: torch.Tensor | None = None   # [1] int32
        self.static_block_table: torch.Tensor | None = None  # [1, W] int32

        self.graphs: dict[tuple, tuple[torch.cuda.CUDAGraph, torch.Tensor, "weakref.ref"]] = {}
        self.graph_order: list[tuple] = []
        self.warmups: dict[tuple, int] = {}
        self.disabled = False

    # -- static buffer management ------------------------------------------

    def _ensure_hidden(self, dtype: torch.dtype):
        if self.static_hidden is None or self.static_hidden.dtype != dtype:
            self.static_hidden = torch.zeros(
                (1, 1, self.hidden_size), dtype=dtype, device=self.device
            )

    def refresh_inputs(self, params: dict):
        """Copy this step's dynamic values into the static device buffers and point
        params at them. Runs before every forward (captured or not) so the graph
        always reads fresh data. The copies are stream-ordered ahead of the replay."""
        dev = self.device

        bt = params.get("block_table")
        if bt is not None and bt is not self.static_block_table:
            width = bt.shape[-1]
            if self.static_block_table is None or self.static_block_table.shape[-1] != width:
                self.static_block_table = torch.zeros((1, width), dtype=torch.int32, device=dev)
            self.static_block_table.copy_(bt, non_blocking=bt.is_pinned())
            params["block_table"] = self.static_block_table

        cs = params.get("cache_seqlens")
        if cs is not None and cs is not self.static_cache_seqlens:
            if self.static_cache_seqlens is None:
                self.static_cache_seqlens = torch.zeros(
                    (cs.shape[0],), dtype=torch.int32, device=dev
                )
            self.static_cache_seqlens.copy_(cs, non_blocking=cs.is_pinned())
            params["cache_seqlens"] = self.static_cache_seqlens

        pos = params.get("positions")
        if pos is not None and pos is not self.static_positions:
            if pos is params.get("cache_seqlens"):
                # decode positions alias cache_seqlens (already staged above)
                params["positions"] = self.static_cache_seqlens
            else:
                if self.static_positions is None:
                    self.static_positions = torch.zeros(
                        (pos.shape[0],), dtype=pos.dtype, device=dev
                    )
                self.static_positions.copy_(pos, non_blocking=pos.is_pinned())
                params["positions"] = self.static_positions

    # -- forward pieces -----------------------------------------------------

    def _embed(self, model, input_ids: torch.Tensor, params: dict):
        """Run the pre-block modules (the CPU token embedding) and stage the hidden
        state into the static device buffer. Returns True on success."""
        x = input_ids
        for module, instance, idx in model.fwd_modules[:model.first_block_idx]:
            params["layer_instance"] = instance
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        if x.device.type != "cpu" or x.shape != (1, 1, self.hidden_size):
            return False
        if not x.is_contiguous():
            x = x.contiguous()
        self._ensure_hidden(x.dtype)
        self.static_hidden.copy_(x, non_blocking=x.is_pinned())
        return True

    def _run_blocks(self, model, params: dict) -> torch.Tensor:
        """The captured region: every fwd module from the first transformer block
        (i.e. after the CPU embedding) through the LM head."""
        x = self.static_hidden
        for module, instance, idx in model.fwd_modules[model.first_block_idx:]:
            params["layer_instance"] = instance
            if module.caps.get("logits_output") and (num := params.get("last_tokens_only")):
                x = x[..., -num:, :].contiguous()
            x = module.prepare_for_device(x, params)
            x = module.forward(x, params)
        return x

    def _key(self, params: dict) -> tuple | None:
        bt = params.get("block_table")
        if bt is None:
            return None
        rsg = params.get("recurrent_states")
        slots = tuple(r.slot for r in rsg) if rsg else ()
        return (
            id(params.get("cache")),
            bt.shape[-1],
            slots,
            params.get("last_tokens_only") or 0,
        )

    def _capture(self, model, input_ids: torch.Tensor, params: dict):
        """Capture the whole block chain. The pre-graph staging (embedding + input
        refresh) runs outside the capture region, exactly as it will at replay
        time.

        Unlike the per-linear capture in bc_rocm, there is NO side-stream warmup
        execution here: the block chain contains destructive state updates (GDN
        recurrent/conv state, KV cache writes), so re-running the step with the
        same input would double-advance them. The WARMUP_STEPS uncaptured decode
        steps have already loaded every kernel and settled every autotuner for
        these exact shapes, which is all the side warmup would provide. During
        the capture itself kernels are only recorded, never executed."""
        self._embed(model, input_ids, params)
        self.refresh_inputs(params)
        dev = self.device
        _bc_capture_suppress(True)
        try:
            torch.cuda.synchronize(dev)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                out = self._run_blocks(model, params)
            return graph, out
        finally:
            _bc_capture_suppress(False)

    def _store_graph(self, key: tuple, graph: torch.cuda.CUDAGraph, out: torch.Tensor, cache):
        if key in self.graph_order:
            self.graph_order.remove(key)
        # id(cache) alone could be recycled after a cache is detached and a new one
        # allocated at the same address; keep a weakref so lookups can verify identity
        self.graphs[key] = (graph, out, weakref.ref(cache))
        self.graph_order.append(key)
        while len(self.graph_order) > MAX_GRAPHS:
            old = self.graph_order.pop(0)
            self.graphs.pop(old, None)

    # -- entry point ---------------------------------------------------------

    def forward(self, model, input_ids: torch.Tensor, params: dict) -> torch.Tensor | None:
        """Returns logits for the decode step, or None if the caller must fall
        back to the regular forward path."""
        if self.disabled:
            return None
        key = self._key(params)
        if key is None:
            return None

        entry = self.graphs.get(key)
        if entry is not None:
            graph, static_out, cache_ref = entry
            if cache_ref() is not params.get("cache"):
                # cache object was replaced (id collision or detach); stale graph
                self.graphs.pop(key)
                self.graph_order.remove(key)
                return None
            if not self._embed(model, input_ids, params):
                return None
            self.refresh_inputs(params)
            graph.replay()
            # Clone: the sampler may hold the logits tensor past the next replay
            return static_out.clone()

        n = self.warmups.get(key, 0)
        if n < WARMUP_STEPS:
            # Regular uncaptured decode steps; Triton autotuning and the
            # per-linear BC graphs settle here. Params are pointed at the
            # static device buffers so the exercised path matches the later
            # capture exactly.
            self.refresh_inputs(params)
            self.warmups[key] = n + 1
            if not self._embed(model, input_ids, params):
                return None
            y = self._run_blocks(model, params)
            if n + 1 == WARMUP_STEPS:
                try:
                    graph, out = self._capture(model, input_ids, params)
                    self._store_graph(key, graph, out, params["cache"])
                except Exception as e:
                    if os.environ.get("EXL3_BLOCK_GRAPHS_DEBUG"):
                        import traceback
                        traceback.print_exc()
                    print(f" !! Block-graph capture failed ({type(e).__name__}: {e}); "
                          f"falling back to regular decode path")
                    self.disabled = True
            return y

        # Warmups exhausted but no graph (previous capture failed): fall back
        self.disabled = True
        return None


def maybe_bg_forward(model, input_ids: torch.Tensor, params: dict) -> torch.Tensor | None:
    """Hook called from Model.forward after prepare_inputs. Returns the decode-step
    logits from a captured graph, or None when the regular path must run.

    On CUDA (or with EXL3_BLOCK_GRAPHS=0) this is a no-op that always returns
    None."""
    if not _eligible(model, input_ids, params):
        return None
    mgr = _managers.get(model)
    if mgr is None:
        mgr = _managers[model] = _StepGraphManager(model)
    try:
        return mgr.forward(model, input_ids, params)
    except Exception as e:
        mgr.disabled = True
        if os.environ.get("EXL3_BLOCK_GRAPHS_DEBUG"):
            import traceback
            traceback.print_exc()
        print(f" !! Block-graph forward failed ({type(e).__name__}: {e}); "
              f"falling back to regular decode path")
        return None
