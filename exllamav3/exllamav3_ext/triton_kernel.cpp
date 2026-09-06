#include <Python.h>
#include "triton_kernel.h"
#include <c10/util/Exception.h>
#include <cstdio>
#include "cuda_drv.h"

#if defined(USE_ROCM)
#include <cstring>
#include <cstdint>
#include <string>
#include <vector>

// Kernel-symbol discovery for the Triton code object (a standard ELF64 image on both
// vendors). Some Triton builds name the emitted kernel differently from metadata.name
// (suffixes, specialization tags), and module_get_function is exact-match; on a NotFound
// we scan the symtab and retry with the symbol that is actually there.

namespace
{
    struct Elf64Shdr { uint32_t name, type; uint64_t flags, addr, off, size; uint32_t link, info; uint64_t align, entsize; };
    struct Elf64Sym  { uint32_t name; uint8_t info, other; uint16_t shndx; uint64_t value, size; };

    bool find_kernel_symbol(const std::string& img, const std::string& requested, std::string& found)
    {
        if (img.size() < 64 || std::memcmp(img.data(), "\x7f" "ELF", 4) != 0) return false;
        auto rd = [&](uint64_t off, uint64_t len) -> const void* {
            return off + len <= img.size() ? img.data() + off : nullptr;
        };
        const uint8_t* e = (const uint8_t*) rd(0, 64);
        uint64_t shoff = 0; uint16_t shentsize = 0, shnum = 0;
        std::memcpy(&shoff, e + 0x28, 8);
        std::memcpy(&shentsize, e + 0x3a, 2);
        std::memcpy(&shnum, e + 0x3c, 2);
        std::vector<std::string> globals;
        for (uint16_t s = 0; s < shnum; ++s)
        {
            const Elf64Shdr* sh = (const Elf64Shdr*) rd(shoff + (uint64_t) s * shentsize, sizeof(Elf64Shdr));
            if (!sh || sh->type != 2) continue;                 // SHT_SYMTAB
            const Elf64Shdr* strs = (const Elf64Shdr*) rd(shoff + (uint64_t) sh->link * shentsize, sizeof(Elf64Shdr));
            if (!strs || strs->off + strs->size > img.size()) continue;
            const char* strtab = (const char*) (img.data() + strs->off);
            size_t strsz = strs->size;
            size_t count = sh->size / (sh->entsize ? sh->entsize : sizeof(Elf64Sym));
            for (size_t i = 0; i < count; ++i)
            {
                const Elf64Sym* sym = (const Elf64Sym*) rd(sh->off + i * (sh->entsize ? sh->entsize : sizeof(Elf64Sym)),
                                                           sizeof(Elf64Sym));
                if (!sym || sym->name >= strsz) continue;
                uint8_t bind = sym->info >> 4, type = sym->info & 0xf;
                if (bind != 1 /*STB_GLOBAL*/ || type != 2 /*STT_FUNC*/ || !sym->shndx || sym->shndx == 0xfff1) continue;
                const char* n = strtab + sym->name;
                size_t nlen = strnlen(n, strsz - sym->name);
                // Skip Triton's kernel-descriptor objects (.kd) and section/metadata symbols
                if (nlen > 3 && std::memcmp(n + nlen - 3, ".kd", 3) == 0) continue;
                globals.emplace_back(n, nlen);
            }
        }
        if (globals.empty()) return false;
        // Prefer a variant of the requested name (handles suffixed/annotated symbols),
        // otherwise a single kernel is definitive
        for (const std::string& g : globals)
            if (g.compare(0, requested.size(), requested) == 0) { found = g; return true; }
        if (globals.size() == 1) { found = globals[0]; return true; }
        return false;
    }
}
#endif

TritonKernel::TritonKernel(py::bytes cubin, std::string _name, int _num_warps, int _shared_bytes) :
    name(std::move(_name)),
    num_warps(_num_warps),
    shared_bytes(_shared_bytes)
{
    // Ensure the primary context of the current device is initialized and current for the
    // driver API before loading the module
    cudaFree(nullptr);

    std::string data = cubin;
    const CudaDrv& drv = CudaDrv::instance();
    cuda_check_drv(drv.module_load_data(&mod, data.data()));
#if defined(USE_ROCM)
    int err = (int) drv.module_get_function(&fn, mod, name.c_str());
    if (err != 0)
    {
        std::string resolved;
        if (err == 500 /*hipErrorNotFound / CUDA_ERROR_NOT_FOUND*/ &&
            find_kernel_symbol(data, name, resolved))
        {
            TORCH_WARN("TritonKernel: kernel symbol \'", resolved, "\' does not match the requested name \'",
                       name, "\'; using the symbol found in the image");
            name = resolved;
            err = (int) drv.module_get_function(&fn, mod, name.c_str());
        }
        if (err != 0)
        {
            fprintf(stderr, "CUDA driver error %d: %s %d\n", err, __FILE__, __LINE__);
            TORCH_CHECK(false, "TritonKernel: no kernel symbol \'", name, "\' in the module (err ", err, ")");
        }
    }
#else
    cuda_check_drv(drv.module_get_function(&fn, mod, name.c_str()));
#endif
    if (shared_bytes > 48 * 1024)
        cuda_check_drv(drv.func_set_attribute(fn, CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, shared_bytes));
}

TritonKernel::~TritonKernel()
{
    if (mod) CudaDrv::instance().module_unload(mod);
}

void TritonKernel::launch(int gx, int gy, int gz, std::vector<void*>& args, cudaStream_t stream) const
{
    // Two hidden trailing params: global scratch and profile scratch, both unused (size 0)
    args.push_back(nullptr);
    args.push_back(nullptr);
    std::vector<void*> arg_ptrs(args.size());
    for (size_t i = 0; i < args.size(); ++i)
        arg_ptrs[i] = &args[i];

    cuda_check_drv(
        CudaDrv::instance().launch_kernel(
            fn,
            gx, gy, gz,
            32 * num_warps, 1, 1,
            shared_bytes,
            stream,
            arg_ptrs.data(),
            nullptr
        )
    );
}
