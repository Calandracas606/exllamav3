#include <cstdio>
#include <c10/util/Exception.h>
#include "cuda_drv.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <dlfcn.h>
#include <link.h>
#include <cstring>
#endif

#define DRV_STR2(x) #x
#define DRV_STR(x) DRV_STR2(x)

static void* drv_sym(void* lib, const char* name)
{
    #ifdef _WIN32
        void* fp = (void*) GetProcAddress((HMODULE) lib, name);
    #else
        void* fp = dlsym(lib, name);
    #endif
    TORCH_CHECK(fp, "CUDA driver symbol not found: ", name);
    return fp;
}

const CudaDrv& CudaDrv::instance()
{
    static CudaDrv d = []
    {
        #ifdef _WIN32
            void* lib = (void*) LoadLibraryA("nvcuda.dll");
        #else
            // Prefer the HIP library the host process already has loaded (torch loads it on
            // ROCm; wheel installs keep it outside the default loader search path, and
            // dlopening a second copy would create two runtime instances whose module and
            // context tables know nothing about each other). Torch's copy is loaded into a
            // local scope (CPython dlopens RTLD_LOCAL), so dlsym(RTLD_DEFAULT) cannot see
            // it; dl_iterate_phdr finds the loaded object's path and dlopening that exact
            // path returns a handle to the same instance.
            void* lib = nullptr;
        #if defined(USE_ROCM)
            if (dlsym(RTLD_DEFAULT, "hipModuleLoadData")) lib = RTLD_DEFAULT;
            if (!lib)
            {
                struct PhdrCtx { const char* hit; } ctx = { nullptr };
                dl_iterate_phdr([](struct dl_phdr_info* info, size_t, void* data) -> int
                {
                    const char* base = std::strrchr(info->dlpi_name, '/');
                    base = base ? base + 1 : info->dlpi_name;
                    if (std::strncmp(base, "libamdhip64", 11) == 0)
                    {
                        static_cast<PhdrCtx*>(data)->hit = info->dlpi_name;
                        return 1;
                    }
                    return 0;
                }, &ctx);
                if (ctx.hit) lib = dlopen(ctx.hit, RTLD_NOW | RTLD_GLOBAL);
            }
        #endif
            if (!lib) lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_GLOBAL);
            if (!lib) lib = dlopen("libcuda.so", RTLD_NOW | RTLD_GLOBAL);
            // ROCm: the HIP runtime exports the same (hipified) entry points
            if (!lib) lib = dlopen("libamdhip64.so", RTLD_NOW | RTLD_GLOBAL);
        #endif
        TORCH_CHECK(lib, "Could not load the CUDA driver library");

        CudaDrv d{};
        d.module_load_data                  = (decltype(&cuModuleLoadData))               drv_sym(lib, DRV_STR(cuModuleLoadData));
        d.module_unload                     = (decltype(&cuModuleUnload))                 drv_sym(lib, DRV_STR(cuModuleUnload));
        d.module_get_function               = (decltype(&cuModuleGetFunction))            drv_sym(lib, DRV_STR(cuModuleGetFunction));
        d.func_set_attribute                = (decltype(&cuFuncSetAttribute))             drv_sym(lib, DRV_STR(cuFuncSetAttribute));
        d.launch_kernel                     = (decltype(&cuLaunchKernel))                 drv_sym(lib, DRV_STR(cuLaunchKernel));
        d.graph_kernel_node_get_params      = (decltype(&cuGraphKernelNodeGetParams))     drv_sym(lib, DRV_STR(cuGraphKernelNodeGetParams));
        d.graph_exec_kernel_node_set_params = (decltype(&cuGraphExecKernelNodeSetParams)) drv_sym(lib, DRV_STR(cuGraphExecKernelNodeSetParams));
        return d;
    }
    ();
    return d;
}
