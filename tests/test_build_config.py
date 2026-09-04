"""
Consistency of the ROCm source-exclusion list (exllamav3/exllamav3_ext/build_config.py) with the
build that is actually loaded, and the no-package-import property that setup.py relies on.
"""

import importlib.util
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
EXT_DIR = os.path.join(ROOT, "exllamav3", "exllamav3_ext")
BC_PATH = os.path.join(EXT_DIR, "build_config.py")


def load_build_config():
    spec = importlib.util.spec_from_file_location("exllamav3_build_config_test", BC_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_loads_without_exllamav3_package():
    """setup.py loads this file by path; it must not trigger the package __init__ (which
    would JIT-compile the extension in the middle of pip install)."""
    saved = sys.modules.pop("exllamav3", None)
    try:
        mod = load_build_config()
        assert callable(mod.get_sources)
    finally:
        if saved is not None:
            sys.modules["exllamav3"] = saved


def test_excludes_only_existing_sources():
    mod = load_build_config()
    for f in mod.ROCM_EXCLUDE_FILES:
        assert os.path.exists(os.path.join(EXT_DIR, f)), f"excluded source missing: {f}"


def test_get_sources_filters_and_skips_hipify_files():
    mod = load_build_config()
    cuda_sources = mod.get_sources(EXT_DIR, is_rocm = False)
    rel = {os.path.relpath(s, EXT_DIR) for s in cuda_sources}
    # CUDA builds everything
    assert "norm.cu" in rel and "quant/exl3_gemm.cu" in rel and "routing.cu" in rel
    assert not any("_hip" in r or r.endswith(".hip") for r in rel)

    rocm_sources = mod.get_sources(EXT_DIR, is_rocm = True)
    rel_rocm = {os.path.relpath(s, EXT_DIR) for s in rocm_sources}
    assert rel_rocm < rel
    for f in mod.ROCM_EXCLUDE_FILES:
        assert f not in rel_rocm
    # kernels the ROCm build must have (bindings and the loader depend on them)
    for f in ["norm.cu", "activation.cu", "routing.cu", "softcap.cu", "hc_mix.cu",
              "dsa_topk.cu", "quant/util.cu", "histogram.cu", "rope.cu"]:
        assert f in rel_rocm, f"{f} must build on ROCm"


def test_rocm_exclusions_match_loaded_module():
    """Every excluded .cu must not be needed by a symbol the module exposes; i.e. the
    extension as loaded must be importable and complete (smoke: a few key symbols)."""
    pytest.importorskip("exllamav3.ext")
    from exllamav3.ext import exllamav3_ext as ext
    mod = load_build_config()
    import torch
    if not torch.version.hip:
        pytest.skip("ROCm build layout check")
    rocm_sources = mod.get_sources(EXT_DIR, is_rocm = True)
    rel_rocm = {os.path.relpath(s, EXT_DIR) for s in rocm_sources}
    # sources excluded on ROCm provide symbols that must be absent; the two warp-mma
    # GEMV engines are the only remaining exclusions
    assert "quant/exl3_gemv.cu" not in rel_rocm
    assert not hasattr(ext, "exl3_gemv_int8")
    # formerly excluded, now built natively: the quantized KV cache and the MoE stack
    assert "cache/q_cache.cu" in rel_rocm
    assert hasattr(ext, "quant_cache_paged")
    assert "quant/exl3_moe.cu" in rel_rocm
    assert hasattr(ext, "exl3_moe")
    assert "routing.cu" in rel_rocm and hasattr(ext, "routing_std")
    # formerly excluded, now built (with the (const void*) cast): conversion works
    assert "quant/quantize.cu" in rel_rocm
    assert hasattr(ext, "quantize_tiles")
