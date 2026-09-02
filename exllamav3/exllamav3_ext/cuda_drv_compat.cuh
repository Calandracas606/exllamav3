#pragma once

// HIP compat for the CUDA driver API seam in cuda_drv.{h,cpp} and graph.cuh.
// HIP has no separate driver surface: the runtime hipModule*/hipGraph* functions
// take the same handle types. All seven entry points are remapped here as macros
// so the seam does not depend on torch's hipify renaming the cu* identifiers in
// the build copy (hipify does map the five module/func/launch names today, but
// does not map the two cuGraph* kernel-node functions; the macros make all of
// them explicit and hipify-independent). They work through DRV_STR's double
// indirection, which stringifies after macro expansion, so
// DRV_STR(cuGraphKernelNodeGetParams) resolves "hipGraphKernelNodeGetParams".

#if defined(USE_ROCM)

#include <hip/hip_runtime_api.h>

#define cuModuleLoadData                 hipModuleLoadData
#define cuModuleUnload                   hipModuleUnload
#define cuModuleGetFunction              hipModuleGetFunction
#define cuFuncSetAttribute              hipFuncSetAttribute
#define cuLaunchKernel                   hipModuleLaunchKernel
#define cuGraphKernelNodeGetParams       hipGraphKernelNodeGetParams
#define cuGraphExecKernelNodeSetParams   hipGraphExecKernelNodeSetParams

// Driver type names hipify does not map (there is no driver surface to map them to)
using CUgraphNode          = hipGraphNode_t;
using CUgraphExec          = hipGraphExec_t;
using CUmodule             = hipModule_t;
using CUfunction           = hipFunction_t;
using cudaKernelNodeParams = hipKernelNodeParams;
using CUDA_KERNEL_NODE_PARAMS = hipKernelNodeParams;

#endif
