"""CUDA-view no-op gate as a pytest test: every file shared with upstream, preprocessed
for the CUDA branch (USE_ROCM undefined), must be byte-identical to upstream/master except
for the enumerated, justified deltas in tests/check_cuda_view.py. The checker needs a git
repository (it diffs against --base), so it skips outside a repo.
"""

import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))


def test_cuda_view_unchanged():
    r = subprocess.run(
        [sys.executable, os.path.join(HERE, "check_cuda_view.py"), "--base", "upstream/master"],
        cwd = os.path.dirname(HERE),
        capture_output = True, text = True,
    )
    if r.returncode == 128 and "Not a git repository" in r.stderr + r.stdout:
        pytest.skip("requires a git repository (base ref checkout)")
    assert r.returncode == 0, f"check_cuda_view failed:\n{r.stdout}\n{r.stderr}"
