"""
Every exllamav3_ext symbol referenced from the Python tree must either be bound on the
current platform or be referenced only from code that guards it (hasattr check or a
capability gate). This test would have caught the ROCm binding-parity holes: unguarded
MoE routing, H-capture's count_inf_nan, the moe_cpu_host import-time crash and the
fused-sampler AttributeError.
"""

import os
import re
import pytest
import torch
from exllamav3.ext import exllamav3_ext as ext

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# str methods that look like attribute access on a variable named ext
FALSE_POSITIVES = {"encode", "lower", "split", "endswith", "startswith", "join", "format"}

# BC_* classes the ROCm build registers natively (same pybind class registrations as
# the CUDA branch's *_bc.h headers): all libtorch BC host TUs build on HIP. Entries
# here are verified to be real classes.
BC_NATIVE = {
    "BC_Mamba2", "BC_GatedDeltaNet", "BC_GatedDeltaNetSplit",
    "BC_MLP", "BC_GatedMLP", "BC_BlockSparseMLP",
    "BC_DSV4Compressor", "BC_DSV4Attention", "BC_DSV4BatchAttention",
    "BC_MLAttention",
}


# Symbols not bound on ROCm whose call sites are gated structurally rather than by a
# literal hasattr in the same file: the gate location is recorded so the mapping is
# reviewed when symbols move. exl3_gemv_int8* is provided by exl3_rocm_stubs.cpp
# (always declining); use_mgemm's int8 path then never engages on its own.
GATED_ELSEWHERE = {
    "exl3_gemv_int8_max_k": "model/config.py use_mgemm (stub returns 0)",
}


def iter_ext_refs():
    pkg = os.path.join(ROOT, "exllamav3")
    for dirpath, _, files in os.walk(pkg):
        for f in files:
            if not f.endswith(".py"): continue
            path = os.path.join(dirpath, f)
            src = open(path, encoding = "utf-8").read()
            for name in re.findall(r"\bext\.([A-Za-z_][A-Za-z0-9_]*)", src):
                if name in FALSE_POSITIVES: continue
                yield os.path.relpath(path, ROOT), name, src


def test_every_referenced_symbol_is_bound_or_guarded():
    missing = []
    for path, name, src in sorted(set(iter_ext_refs())):
        if hasattr(ext, name): continue
        if name in GATED_ELSEWHERE: continue
        # Not bound on this platform: every file referencing it must guard the access
        guards = (
            f'hasattr(ext, "{name}")' in src or
            f"hasattr(ext, '{name}')" in src
        )
        if not guards:
            missing.append(f"{path}: ext.{name} not bound and no hasattr guard in file")
    assert not missing, "\n".join(missing)


@pytest.mark.skipif(torch.version.hip is None, reason = "ROCm-only: native BC class registration")
def test_bc_native_classes():
    for name in sorted(BC_NATIVE):
        assert hasattr(ext, name), f"{name} missing from the ROCm binding"
        # native pybind class, not the stub function
        assert isinstance(getattr(ext, name), type), f"{name} is not a class (stub leak?)"


def _branch_def_names(text, rocm: bool):
    """Names bound in the requested branch of bindings.cpp (function defs and classes)."""
    names = set(re.findall(r'm\.def\("(\w+)"', text))
    names |= set(re.findall(r'py::class_<\w+[^>]*>\(m,\s*"(\w+)"\)', text))
    if not rocm:
        # defs arriving via the *_bc.h headers pulled into the CUDA module body
        bc = ("linear_bc", "gated_delta_net_bc", "attention_bc", "mla_attention_bc",
              "gated_rmsnorm_bc", "mlp_bc", "blocksparse_mlp_bc", "dsv4_compressor_bc",
              "dsv4_attn_bc")
        for h in bc:
            p = os.path.join(ROOT, "exllamav3", "exllamav3_ext", "libtorch", h + ".h")
            if os.path.exists(p):
                names |= set(re.findall(r'py::class_<[\w ,<>:]+>\(m,\s*"(\w+)"\)', open(p).read()))
    return names


def test_binding_branches_match_module():
    """The m.def list in the active branch of bindings.cpp must match what the module
    actually exposes (parity between the binding file, build_config and the build)."""
    bpath = os.path.join(ROOT, "exllamav3", "exllamav3_ext", "bindings.cpp")
    src = open(bpath, encoding = "utf-8").read()
    m = re.search(r"#if !defined\(USE_ROCM\)(.*?)#else(.*?)#endif\n}", src, re.S)
    assert m, "two-branch module body not found"
    cuda_branch, rocm_branch = m.group(1), m.group(2)
    cuda_names = _branch_def_names(cuda_branch, rocm = False)
    rocm_names = _branch_def_names(rocm_branch, rocm = True)
    # the ROCm def list binds only what the CUDA build binds
    assert not (rocm_names - cuda_names), sorted(rocm_names - cuda_names)
    if torch.version.hip:
        for name in rocm_names:
            assert hasattr(ext, name), f"ROCm branch binds it, module lacks: {name}"
