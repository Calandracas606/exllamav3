"""Python BC_* (batched-capture) implementations for ROCm.

On CUDA, the BC_* classes live in C++ (``exllamav3_ext/libtorch/``) and batch
kernel launches into a CUDA graph. Those sources are excluded from the ROCm
build, so ``ext.BC_LinearEXL3(...)`` would otherwise return ``None`` (see
``ext.py``'s ``_BCNone`` stub) and every EXL3 linear layer would fall back to
the Python dispatch path.

This module reimplements the hot-path classes in pure Python on top of
``torch.cuda.CUDAGraph``. The whole linear forward —
``exl3_ops::LinearEXL3_triton`` (hadamard -> fused-dequant-gemm -> hadamard,
+ optional bias) — is captured once into a graph and replayed, so a decoded
token pays one graph launch per layer instead of per-kernel dispatch.
"""
from __future__ import annotations

import torch

# Number of uncaptured calls before the graph is captured (mirrors the C++
# implementation's trigger point on the third invocation of a shape).
_WARMUP_ITERS = 3

# Extra warmup iterations on the side stream inside _capture. The uncaptured
# calls above have already settled the Triton autotuner; one side-stream run
# just ensures the kernels are loaded before the capture region begins.
_CAPTURE_WARMUP_ITERS = 1


class BC_LinearEXL3:
    """Graph-capturing EXL3 linear for the bsz == 1 decode path.

    Mirrors the C++ ``BC_LinearEXL3`` constructor and ``run_alloc`` method.
    The constructor receives a pre-allocated ``xh`` staging buffer of shape
    ``(1, in_features)`` (from ``g_tensor_cache``); it is stored for API
    compatibility. A separate half static buffer is used for the Hadamard-
    transformed input inside the captured graph, since ``had_r_128`` requires a
    half output while the passed-in buffer may be float32.
    """

    def __init__(
        self,
        trellis: torch.Tensor,
        suh: torch.Tensor | None,
        svh: torch.Tensor | None,
        K: int,
        bias: torch.Tensor | None,
        mcg: bool,
        mul1: bool,
        xh: torch.Tensor,
    ):
        self.trellis = trellis
        self.suh = suh
        self.svh = svh
        self.K = K
        self.bias = bias
        self.mcg = mcg
        self.mul1 = mul1
        self.xh = xh  # staging buffer [1, in_features], half

        self.device = trellis.device
        self.in_features = xh.shape[-1]

        # Lazily populated on the first bsz==1 call for each output dtype. The decode
        # hot path is half, but some projections (e.g. hyperconnection inputs that
        # feed norm / attention accumulation) request fp32 output; both are captured
        # independently.
        self._static_x: torch.Tensor | None = None
        self._static_xh: torch.Tensor | None = None
        self._static_ys: dict[torch.dtype, torch.Tensor] = {}
        self._graphs: dict[torch.dtype, torch.cuda.CUDAGraph] = {}
        self._warmups: dict[torch.dtype, int] = {}

    def _compute(self, static_y: torch.Tensor):
        """Run the complete EXL3 linear into ``static_y`` via the composition op.

        Imports at call time so importing bc_rocm (from ext.py) doesn't pull in
        triton, and so the op is registered before first use. Must only be
        called after the static buffers are populated.
        """
        torch.ops.exl3_ops.LinearEXL3_triton(
            self._static_x, static_y, self._static_xh,
            self.trellis, self.suh, self.svh,
            self.K, self.mcg, self.mul1, self.bias,
            self.in_features, static_y.shape[-1],
        )

    def _capture(self, x: torch.Tensor, out_features: int, out_dtype: torch.dtype):
        """Warm up once, then capture the graph for this shape/dtype."""
        # The caller's out_features must match the trellis, else the graph would
        # be captured for the wrong shape (trellis dim 1 is the out axis).
        assert out_features == self.trellis.shape[1] * 16, \
            f"BC_LinearEXL3: out_features {out_features} != trellis {self.trellis.shape[1] * 16}"

        self._static_x = torch.empty((1, self.in_features), dtype=torch.half, device=self.device)
        # Reuse the shared g_tensor_cache staging buffer when it is half (the common
        # case, matching the C++ implementation); only allocate a private buffer when
        # the constructor buffer is float32, since had_r_128 requires a half output.
        if self.xh.dtype == torch.half:
            self._static_xh = self.xh
        else:
            self._static_xh = torch.empty((1, self.in_features), dtype=torch.half, device=self.device)
        static_y = torch.empty((1, out_features), dtype=out_dtype, device=self.device)

        # Warm up on a side stream so capture runs cleanly: lazy allocations and
        # kernel loading must settle before the capture region begins.
        side = torch.cuda.Stream(device=self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(_CAPTURE_WARMUP_ITERS):
                self._static_x.copy_(x)
                self._compute(static_y)
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            self._compute(static_y)
        self._graphs[out_dtype] = graph
        self._static_ys[out_dtype] = static_y

    def run_alloc(self, x: torch.Tensor, out_features: int, output_fp32: bool) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        rows = x_flat.shape[0]

        out_dtype = torch.float if output_fp32 else torch.half
        out_shape = original_shape[:-1] + (out_features,)

        if out_features == 0:
            return torch.empty((rows, 0), dtype=out_dtype, device=self.device).view(out_shape)

        # The captured graph is only valid for the bsz == 1 shape. Any other shape
        # runs the regular (non-captured) compute path, exactly like the C++ version
        # which only graph-captures bsz == 1.
        if rows == 1 and x_flat.dtype == torch.half:
            graph = self._graphs.get(out_dtype)
            if graph is not None:
                # Replay path: copy input into static buffer, replay, clone the static
                # output into a fresh tensor (the static buffer is overwritten by the
                # next replay).
                self._static_x.copy_(x_flat)
                graph.replay()
                return self._static_ys[out_dtype].clone().view(out_shape)
            n = self._warmups.get(out_dtype, 0)
            if n < _WARMUP_ITERS:
                # First few calls of this dtype warm up the autotuner + capture. These
                # are real forward passes; compute directly and capture on the last one.
                self._warmups[out_dtype] = n + 1
                result = self._run_uncaptured(x_flat, out_features, out_dtype)
                if n + 1 == _WARMUP_ITERS:
                    self._capture(x_flat, out_features, out_dtype)
                return result.view(out_shape)

        # General / bsz > 1 path (no graph, no staging copies).
        result = self._run_uncaptured(x_flat, out_features, out_dtype)
        return result.view(out_shape)

    def _run_uncaptured(self, x_flat: torch.Tensor, out_features: int, out_dtype: torch.dtype) -> torch.Tensor:
        """Regular exl3_gemm compute (no graph capture), returning a fresh tensor."""
        from .exl3_gemm_triton import exl3_gemm
        return exl3_gemm(
            x_flat, self.trellis, self.suh, self.svh, self.K,
            self.mcg, self.mul1, self.in_features, out_features,
            self.device, out_dtype, self.bias,
        )


class BC_LinearFP16:
    """Minimal graph-capturing fp16 linear.

    Matches the C++ ``BC_LinearFP16`` constructor. ``weight`` is [in, out].
    Provided for completeness; the fp16 forward path currently runs inline.
    """

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor | None = None):
        self.weight = weight
        self.bias = bias

    def run_alloc(self, x: torch.Tensor, out_features: int, output_fp32: bool) -> torch.Tensor:
        original_shape = x.shape
        x_flat = x.reshape(-1, x.shape[-1])
        out_dtype = torch.float if output_fp32 else torch.half
        out_shape = original_shape[:-1] + (out_features,)
        if output_fp32:
            out = torch.matmul(x_flat, self.weight).to(torch.float)
        else:
            out = torch.matmul(x_flat, self.weight)
        if self.bias is not None:
            out = out + self.bias
        return out.view(out_shape)
