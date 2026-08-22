# Quality gates audit

Date: 2026-08-22
Repository: `zapabob/hakua-memory`
Working branch: `codex/hakua-memory-quality-gates`
Starting commit: `8ff5aa3923bc811b718d6775011e5c6499a9f753`
Package version: `0.3.3`

## Scope

Quality-only maintenance for the published Python library. The Ebbinghaus,
Semantic Graph, RAG, hybrid retrieval, contradiction detection, Obsidian, and
CompositeMemory feature contracts were preserved. No new memory algorithm,
embedding backend, edge type, database schema, public API redesign, tag, push,
or PyPI publish was performed.

## Baseline

The starting worktree was clean on `main`. Repository-wide pytest collection
failed because `tests/test_comprehensive.py` and `scripts/test_comprehensive.py`
had the same import name, and the old test module resolved `tests/src` and
`tests/pyproject.toml` paths. The direct comprehensive and RAG scripts passed.
Ruff passed only because directory-wide F821, F841, F401, and E402 ignores hid
source issues. The Windows `python` alias was unusable; `py -3.12` reported
Python 3.12.10. The first pip/uv installation attempts hung before an isolated
verification environment with pip was prepared.

## Implementation record

- The canonical suite is now entirely under `tests/`; former `scripts/test_*.py`
  runners were moved or removed, and temporary databases use `tmp_path`.
- The stale duplicate module and stale path assumptions were removed while
  preserving the integration assertions for memory recall, graph search, RAG,
  citations, meeting items, ACL, configuration, Obsidian rendering, and stats.
- `pyproject.toml` now defines `[all]` as the union of optional feature
  dependencies, keeps version `0.3.3`, and uses the current SPDX-style license
  metadata. Ruff blanket ignores were removed; exposed undefined names and
  unused imports/assignments were corrected locally.
- Tracked `.memory_*` backups, synthetic datasets, and judge request/result JSON
  were classified as generated runtime or benchmark output and removed from
  version control. Root-level generated outputs are ignored. Their recipe is
  retained in `benchmarks/synthetic_dataset_config.json`, and deterministic
  generation is implemented by `scripts/generate_synthetic_data.py`.
- `scripts/benchmark_cross_validation.py` records version, commit SHA, UTC
  timestamp, OS, Python, CPU, dataset id, seed, sample count, warmup, repetitions,
  and mean/median/stdev/p50/p95/p99 for each measured operation. It uses a
  temporary database root and emits no unsupported external comparison values.
- CI retains the Ubuntu, Windows, and macOS matrix across Python 3.11, 3.12,
  and 3.13. Its canonical test command is `python -m pytest -q`, and Ruff also
  checks `tests`.

## Verification evidence

The isolated Python 3.12.13 environment completed `python -m pip install -e
".[dev]"`, `pip check`, the full pytest suite, Ruff, compileall, `python -m
build`, and `python -m twine check dist/*`. The full suite passed 29 tests. A
seeded generator smoke run produced identical SHA-256 output for two runs, and
the benchmark schema test passed. The final exact command results and local
commit SHAs are recorded in the implementation handoff.

## Release boundary

This audit does not authorize a version bump, tag creation, main-branch push,
or PyPI publication. Version `0.3.4` remains only a future candidate after an
independent release decision.
