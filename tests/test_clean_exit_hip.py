import os
import subprocess
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

if not torch.version.hip:
    pytest.skip("HIP build required", allow_module_level = True)


# TheRock ROCm 7.14 gfx1100: after a TP session, HSA's C-level atexit handler
# (hsa_shut_down -> AqlQueue::~AqlQueue) segfaults during exit(), turning clean runs
# into exit 139. The workaround runs all registered Python-side cleanup hooks and then
# leaves via os._exit, skipping the faulting C-level atexit chain entirely.
_CHILD = """
import sys

sys.path.append({repo!r})

from exllamav3.util.misc import Cleanupper, bypass_rocm_atexit_crash

marker = {marker!r}
def hook():
    with open(marker, "w") as f:
        f.write("ran")

Cleanupper().register_atexit(hook)
bypass_rocm_atexit_crash(0)
# On HIP this line is never reached (os._exit above); reaching it means no-op path
with open(marker + ".reached", "w") as f:
    f.write("returned")
"""


def test_bypass_rocm_atexit_crash_runs_cleanup_and_exits_clean():

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        marker = os.path.join(td, "hook_ran")
        code = _CHILD.format(repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__))), marker = marker)
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output = True,
            text = True,
            timeout = 120,
        )
        # Exited through os._exit with the requested status (not a signal, not 1)
        assert result.returncode == 0, f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        # The registered cleanupper hook ran before the exit
        assert os.path.exists(marker) and open(marker).read() == "ran"
        # And execution did NOT continue past the call on HIP
        assert not os.path.exists(marker + ".reached")
