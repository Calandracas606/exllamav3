# AGENTS.md — Repository Guidelines for exllamav3

## Build System

- C++ extension is built via JIT (`torch.utils.cpp_extension`) on first import.
- ROCm build: `uv run --env-file .env python -c "from exllamav3 import exllamav3_ext"`
- Build cache: `~/.cache/torch_extensions/py311_rocm*/exllamav3_ext/`. Delete to force rebuild.
- ROCm 7.14, TheRock pip index, clang 22, `-Wno-register` flag required.
- gfx1100 (RX 7900 XTX) is the target architecture. Arch auto-detected via `arch_list.py`.

## Testing

```bash
# Full suite (162 tests, ~35s)
EXL_TEST_DEVICE=cuda:0 uv run --env-file .env python -m pytest tests/ -x -q

# Model-dependent tests require EXL_TEST_MODEL env var (auto-skipped if absent)
EXL_TEST_MODEL=/path/to/model EXL_TEST_DEVICE=cuda:0 uv run --env-file .env python -m pytest tests/test_rocm_qgemm.py -v
```

## CUDA/HIP Porting

See `.agents/skills/cuda-to-hip-porting/SKILL.md` for detailed patterns. Key rules:

1. **Never break upstream CUDA.** Macro or polyfill CUDA builtins, don't remove them.
2. **Use wrapper functions** for platform differences (e.g., warp shuffles), not mechanical sed replacements.
3. **Check HIPIFY compatibility tables** before writing polyfills — HIP may already have the function.
4. **abort() not wrong emulation** for unreachable code paths. Dead + wrong is worse than dead + crash.
5. **Use `rocminfo`** for architecture detection, not `device_properties().major/minor`.

## Code Style

- C++20, `-Ofast` for kernel code
- `#pragma once` before any includes in header files
- Named constants over magic numbers (`constexpr int kMaxSharedCarveout = 100;`)
- Document performance tradeoffs in TODO comments with concrete numbers

## Development Artifacts

`HANDOFF.md` and `prd-*.md` are working documents — do not ship them in PRs.
They contain machine-specific paths and may have stale or incorrect claims.

## Git

- Branch convention: feature branches off `master`
- Commits: `Co-authored-by: openhands <openhands@all-hands.dev>`
- Don't push to `master`/`main` directly
