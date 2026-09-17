import os
import subprocess
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")
pytest.importorskip("rocm_sdk")

if not torch.version.hip:
    pytest.skip("TheRock ROCm build required", allow_module_level = True)


# The subprocess reproduces the -tp load path's environment: torch imported with no
# ROCm dirs in LD_LIBRARY_PATH (the venv wheel resolves them via ctypes preloads, so
# the parent works), then file_system sharing execs torch_shm_manager which cannot
# resolve libtorch_cpu's ROCm dependencies unless the fix exported the lib dirs.
_SUBPROCESS_CODE = """
import os

import torch
import exllamav3.model.model_tp as model_tp

model_tp._ensure_shm_manager_rocm_env()
lp1 = os.environ.get("LD_LIBRARY_PATH", "")
model_tp._ensure_shm_manager_rocm_env()
lp2 = os.environ.get("LD_LIBRARY_PATH", "")

assert lp1 == lp2, f"not idempotent: {lp1!r} -> {lp2!r}"
assert "_rocm_sdk_core/lib" in lp1, f"ROCm lib dirs missing from LD_LIBRARY_PATH: {lp1!r}"

torch.multiprocessing.set_sharing_strategy("file_system")
torch.zeros(4).share_memory_()
print("SHM_OK")
"""


def test_file_system_sharing_works_without_preset_ld_library_path():

    env = {k: v for k, v in os.environ.items() if k != "LD_LIBRARY_PATH"}
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_CODE],
        capture_output = True,
        text = True,
        timeout = 600,
        env = env,
    )
    assert "SHM_OK" in result.stdout
    assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
