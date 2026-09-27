#pragma once

#if defined(__HIPCC__)
// rocWMMA under torch's __HIP_NO_HALF_{CONVERSIONS,OPERATORS}__ flags:
// - the __half vector registrations static_cast float->__half (deleted ctor) -> break
// - HIP_NO_HALF skips them, but this SDK build still references hfloat16_t in
//   wmma_impl's gfx11 derivative regardless of the enable gate -> also breaks
// Bridge: staged includes so the alias exists (float16_t is bit-identical to
// __half) before the rest of rocwmma parses. Must run BEFORE any torch/ATen
// header (the float8 trait glue also breaks after the torch chain).
#ifndef EXL3_ROCWMMA_BRIDGE
#define EXL3_ROCWMMA_BRIDGE
#define HIP_NO_HALF 1
#include <rocwmma/internal/config.hpp>
#include <rocwmma/internal/types.hpp>
namespace rocwmma { using hfloat16_t = float16_t; }
#include <rocwmma/rocwmma.hpp>
#undef HIP_NO_HALF
#endif
#endif

#include <ATen/Tensor.h>
#include "../graph.cuh"

int exl3_gemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms,
    Graph* graph
);

int exl3_gemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const c10::optional<at::Tensor>& suh,
    const c10::optional<at::Tensor>& A_had,
    const c10::optional<at::Tensor>& svh,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int force_num_sms
);

int exl3_mgemm_gr
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K,
    int force_shape_idx,
    bool mcg,
    bool mul1,
    int min_index,
    int max_index,
    int force_num_sms,
    Graph* graph,
    int num_tokens = 1,
    const c10::optional<at::Tensor>& size_n_list = {},
    const c10::optional<at::Tensor>& c_ptrs = {},
    // Sliced mode (see exl3_mgemm_gr): per-entry full row width of the slice's matrix, per-entry
    // source matrix index (suh and A_had are then per source), and the number of sources
    const c10::optional<at::Tensor>& n_stride_list = {},
    const c10::optional<at::Tensor>& had_src_list = {},
    int num_had_src = 0
);

int exl3_mgemm
(
    const at::Tensor& A,
    const at::Tensor& B,
    at::Tensor& C,
    const at::Tensor& suh,
    const at::Tensor& A_had,
    const at::Tensor& svh,
    const c10::optional<at::Tensor>& indices,
    const c10::optional<at::Tensor>& weights,
    int K,
    int force_shape_idx,
    uint32_t mcg_mult,
    uint32_t mul1_mult,
    int min_index,
    int max_index,
    int force_num_sms,
    int num_tokens = 1,
    const c10::optional<at::Tensor>& size_n_list = {},
    const c10::optional<at::Tensor>& c_ptrs = {},
    const c10::optional<at::Tensor>& n_stride_list = {},
    const c10::optional<at::Tensor>& had_src_list = {},
    int num_had_src = 0
);
