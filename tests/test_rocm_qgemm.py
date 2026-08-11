"""Test EXL3 reconstruct_hgemm on ROCm against pure reference.

Adapted from tests/test_qgemm.py. Since both qgemm and reconstruct paths
go through the same fallback on ROCm, we compare against a pure PyTorch
matmul using get_weight_tensor() as reference.

Requires a real EXL3 model on disk. Set EXL_TEST_MODEL to the model directory,
or it will be auto-skipped.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

torch.set_printoptions(precision=5, sci_mode=False, linewidth=200)

test_model = os.environ.get("EXL_TEST_MODEL", "")
device = os.environ.get("EXL_TEST_DEVICE", "cuda:0")
batch_sizes = [1, 2, 8, 16, 17, 32, 33]

_skip = not test_model or not os.path.isdir(test_model)
_skip_reason = (
    "Set EXL_TEST_MODEL to a directory containing an EXL3-quantized model"
    if not test_model else f"Model directory not found: {test_model}"
)

pytestmark = pytest.mark.skipif(_skip, reason=_skip_reason)


@pytest.fixture(scope="module")
def model_linears():
    """Load model once per session, find LinearEXL3 layers."""
    from exllamav3 import Config, Model, Cache
    from exllamav3.modules.quant.exl3 import LinearEXL3

    config = Config.from_directory(test_model)
    model = Model.from_config(config)
    model.load()

    all_linears = []
    for m in model.modules:
        for attr_name in vars(m):
            attr = getattr(m, attr_name)
            if isinstance(attr, LinearEXL3):
                all_linears.append((m.key, attr_name, attr))

    # Pick a representative subset: lm_head, one attention proj, one mlp proj
    test_subset = []
    for mkey, aname, lin in all_linears:
        if aname == "inner" and mkey == "lm_head":
            test_subset.append((mkey, aname))
        elif "q_proj" in mkey or "gate_proj" in mkey or "down_proj" in mkey:
            if len([t for t in test_subset if t[0] == mkey]) == 0:
                test_subset.append((mkey, aname))
        if len(test_subset) >= 4:
            break
    return all_linears, test_subset


def _get_test_params(model_linears):
    _, test_subset = model_linears
    return test_subset


@pytest.mark.parametrize("batch_size", batch_sizes)
@torch.inference_mode()
def test_reconstruct_hgemm(model_linears, batch_size):
    """Compare reconstruct_hgemm output against pure reference matmul."""
    all_linears, test_subset = model_linears

    for mkey, aname in test_subset:
        linear = None
        for mk, ak, lin in all_linears:
            if mk == mkey and ak == aname:
                linear = lin
                break
        assert linear is not None

        torch.manual_seed(0)
        x = torch.randn((1, batch_size, linear.in_features), dtype=torch.float16, device=device)

        # Run reconstruct_hgemm
        y_exl = linear.reconstruct_hgemm(x, None)

        # Reference: pure PyTorch matmul with full weight tensor
        w_ref = linear.get_weight_tensor()
        y_ref = torch.matmul(x.float(), w_ref.float()).half()

        # Allow some tolerance for half precision
        tol = 0.05
        try:
            torch.testing.assert_close(y_exl, y_ref, rtol=tol, atol=tol)
        except AssertionError as e:
            max_err = (y_exl.float() - y_ref.float()).abs().max().item()
            rel_err = max_err / (y_ref.float().abs().max().item() + 1e-9)
            pytest.fail(f"{mkey}.{aname} bsz={batch_size}: max_err={max_err:.6f}, rel_err={rel_err:.6f}\n{e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
