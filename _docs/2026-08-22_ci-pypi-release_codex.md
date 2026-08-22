# CI and PyPI release audit

Date: 2026-08-22
Repository: `zapabob/hakua-memory`
Release: `v0.3.3`
Primary release commit: `7e1f71f1da98b3b49c287ceb9855d2116aaf6ca9`

## Scope

Restore the GitHub Actions quality gates and make the registered PyPI project publishable without changing unrelated repository state.

## Changes

- Updated the package version from `0.2.0` to `0.3.3`; `0.3.2` was already tagged and PyPI already contained `0.2.0`.
- Repaired Ruff findings in `scripts/benchmark_cross_validation.py`, `scripts/contradiction_detector.py`, and `scripts/hermes_llm_judge.py`.
- Updated the comprehensive-test version assertion to `0.3.3`.
- Kept the existing workflow, secret name, and release path; no secret value was read or recorded.

## Verification

Local checks passed with UTF-8 output: `ruff check src scripts`, `python -m compileall -q src scripts`, `python scripts/test_comprehensive.py`, `python scripts/test_rag_e2e.py`, package build, and `twine check` for both the wheel and sdist. The local isolated build initially hit the machine's zero-byte C: drive; the same artifacts built successfully in an H: temporary workspace with no source workaround.

GitHub Actions run `32554498254` passed on `main` at the primary release commit, including lint, the Python 3.11-3.13 Ubuntu/Windows/macOS matrix, and build. Tag run `32554579486` passed the same matrix, build, and the PyPI publish job. PyPI exposes version `0.3.3` with `hakua_memory-0.3.3-py3-none-any.whl` and `hakua_memory-0.3.3.tar.gz`.

## Residual observations

The standalone repository-wide pytest invocation has a pre-existing duplicate module-name/import mismatch and was not part of the GitHub workflow. The GitHub runner also reports non-blocking Node.js 20 deprecation annotations for existing actions.
