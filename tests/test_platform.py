"""Tests for platform detection utility."""
import torch
from exllamav3.util.platform import IS_ROCM, IS_CUDA, has_ext


def test_is_rocm_matches_torch():
    assert IS_ROCM == (torch.version.hip is not None)

def test_is_cuda_matches_torch():
    assert IS_CUDA == (torch.version.cuda is not None)

def test_has_ext_returns_true_for_existing():
    # hgemm is always compiled (Category B)
    assert has_ext('hgemm') == True

def test_has_ext_for_platform_specific():
    # rms_norm is excluded from ROCm build
    if IS_ROCM:
        assert has_ext('rms_norm') == False
    else:
        assert has_ext('rms_norm') == True

def test_has_ext_false_for_nonexistent():
    assert has_ext('nonexistent_function_12345') == False
