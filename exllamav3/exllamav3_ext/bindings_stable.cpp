/**
 * Custom operator bindings for exllamav3 (TORCH_LIBRARY).
 *
 * Registers existing C++ kernels as proper PyTorch custom operators so they
 * compose with torch.compile, CUDA graph capture, and the dispatcher.
 *
 * Compiled on both CUDA and ROCm alongside the pybind11 bindings.cpp.
 * Each op here is also available via the pybind11 path; the TORCH_LIBRARY
 * registration adds dispatcher/compile support.
 */
#include <torch/library.h>
#include <ATen/Tensor.h>
#include "quant/hadamard.cuh"

// exl3_ops::had_r_128 — 128-element row Hadamard transform with optional sign scaling.
// `output` is mutated in-place.

static void had_r_128_dispatch(
    const at::Tensor& input,
    at::Tensor& output,
    const c10::optional<at::Tensor>& pre_scale,
    const c10::optional<at::Tensor>& post_scale,
    double scale
) {
    had_r_128(input, output, pre_scale, post_scale, static_cast<float>(scale));
}

TORCH_LIBRARY(exl3_ops, m) {
    m.def(
        "had_r_128(Tensor input, Tensor(a!) output, Tensor? pre_scale, "
        "Tensor? post_scale, float scale) -> ()"
    );
}

TORCH_LIBRARY_IMPL(exl3_ops, CUDA, m) {
    m.impl("had_r_128", had_r_128_dispatch);
}
