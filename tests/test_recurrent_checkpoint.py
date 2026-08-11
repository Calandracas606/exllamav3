"""Test recurrent checkpoint handling for hybrid models (Mamba2 + full attention).

Regression test for a bug where is_checkpoint_boundary() returns True when
seq.kv_position == 0, triggering an assertion failure in maybe_stash_recurrent()
because the page hasn't been filled yet (page.kv_position == 0 != PAGE_SIZE).

This affects hybrid models with linear attention layers that use recurrent
checkpoints, when generating from very short prompts.

Requires a real EXL3 model on disk. Set EXL_TEST_MODEL to the model directory,
or it will be auto-skipped.
"""
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
import torch

torch.set_printoptions(precision=5, sci_mode=False, linewidth=200)

test_model = os.environ.get("EXL_TEST_MODEL", "")
_skip = not (test_model and os.path.isdir(test_model))
_skip_reason = (
    "Set EXL_TEST_MODEL to a directory containing an EXL3-quantized model"
) if not test_model else f"Model directory not found: {test_model}"

pytestmark = pytest.mark.skipif(_skip, reason=_skip_reason)


@pytest.fixture(scope="module")
def generator():
    from exllamav3 import Config, Model, Cache, Tokenizer
    from exllamav3.generator.generator import Generator
    config = Config.from_directory(test_model)
    model = Model.from_config(config)
    cache = Cache(model=model, max_num_tokens=8192)
    model.load()
    tokenizer = Tokenizer.from_config(config)
    gen = Generator(model, cache, tokenizer)
    yield gen


def test_short_prompt_does_not_crash_recurrent_checkpoint(generator):
    """A 1-token prompt should not crash the recurrent checkpoint logic.

    Before the fix, kv_position == 0 after prefill caused
    is_checkpoint_boundary() to return True (0 % interval == 0),
    then maybe_stash_recurrent() hit:
        assert page.kv_position == PAGE_SIZE
    on an empty page.
    """
    from exllamav3.generator.sampler import DefaultSampler
    # "Hi" is a single token — minimal prompt
    output = generator.generate(
        "Hi",
        max_new_tokens=4,
        sampler=DefaultSampler(),
        add_bos=True,
    )
    assert len(output) > 0


def test_sequential_generate_calls(generator):
    """Two sequential generate() calls on the same generator should both work.

    Before the fix, the first call could leave recurrent checkpoint state
    that caused the second call to crash at a page boundary assertion.
    """
    from exllamav3.generator.sampler import DefaultSampler
    out1 = generator.generate(
        "The capital of France is",
        max_new_tokens=4,
        sampler=DefaultSampler(),
        add_bos=True,
    )
    out2 = generator.generate(
        "Hello world",
        max_new_tokens=4,
        sampler=DefaultSampler(),
        add_bos=True,
    )
    assert len(out1) > 0
    assert len(out2) > 0
