import os
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

torch = pytest.importorskip("torch")

if not torch.version.hip:
    pytest.skip("HIP build required", allow_module_level = True)

from exllamav3.model.config import InferParams


# The fused exl3_mgemm path (MultiLinear) is broken on gfx1100/RDN3: the HIP kernel
# flattens the per-matrix pointer arrays into a single matrix, producing scrambled
# staging buffers and out-of-bounds writes (see AGENTS.md). use_mgemm must keep every
# caller on the separate-projection path until the kernel is fixed upstream.
def test_use_mgemm_disabled_on_hip():

    ip = InferParams()

    assert ip.use_mgemm(6, 17408, mul1 = False, device = torch.device("cuda:0")) is False
    assert ip.use_mgemm(4, 1024, mul1 = True, device = torch.device("cuda:0")) is False
    # No device supplied (module-loading time decision) must also refuse
    assert ip.use_mgemm(6, 17408, mul1 = False) is False
