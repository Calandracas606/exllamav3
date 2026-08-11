"""Platform detection and extension capability utilities.

Centralizes platform checks so call sites use IS_ROCM / has_ext() instead
of scattered torch.version.hip checks.
"""
import torch

IS_ROCM: bool = torch.version.hip is not None
IS_CUDA: bool = torch.version.cuda is not None


def has_ext(name: str) -> bool:
    """Check whether a C++ extension function or class is available.

    More robust than checking torch.version.hip: tests the actual compiled
    capability rather than inferring from the platform.
    """
    from ..ext import exllamav3_ext as ext
    return hasattr(ext, name)
