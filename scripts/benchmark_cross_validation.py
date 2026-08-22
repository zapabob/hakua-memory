from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import statistics
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Callable, Sequence

try:
    from synthetic_dataset import generate_business_dataset
except ModuleNotFoundError:
    from scripts.synthetic_dataset import generate_business_dataset

LOGGER = logging.getLogger(__name__)
REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--dataset-id", default="synthetic-business-v1")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.warmup < 0 or args.repetitions < 1:
        parser.error("samples and repetitions must be positive; warmup cannot be negative")
    return args


def _percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] + (ordered[upper] - ordered[lower]) * weight


def _measure(operation: Callable[[], Any], warmup: int, repetitions: int) -> dict[str, float | str]:
    for _ in range(warmup):
        operation()
    durations: list[float] = []
    for _ in range(repetitions):
        started = time.perf_counter_ns()
        operation()
        durations.append((time.perf_counter_ns() - started) / 1_000_000)
    return {
        "metric": "latency_ms",
        "mean": statistics.fmean(durations),
        "median": statistics.median(durations),
        "stdev": statistics.stdev(durations) if len(durations) > 1 else 0.0,
        "p50": _percentile(durations, 0.50),
        "p95": _percentile(durations, 0.95),
        "p99": _percentile(durations, 0.99),
    }


def _package_version() -> str:
    try:
        return version("hakua-memory")
    except PackageNotFoundError:
        import tomllib

        with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
            return str(tomllib.load(stream)["project"]["version"])


def _git_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "not measured"
    return result.stdout.strip()


def _prepare_memory(rows: list[dict[str, str]], root: Path) -> tuple[Any, str]:
    from hakua_memory.composite import CompositeMemory

    memory = CompositeMemory(root)
    for row in rows:
        memory.remember(row["content"], tags=[row["domain"], row["language"]])
        memory.add_node(
            {
                "node_id": f"benchmark-{row['id']}",
                "node_type": "Claim",
                "label": row["topic"],
                "summary": row["content"],
                "status": "asserted",
                "confidence": 0.9,
                "salience": 0.8,
            }
        )
    return memory, rows[0]["topic"]


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    from hakua_memory.rag.chunking import chunk_text

    rows = generate_business_dataset(args.seed, args.samples)
    rag_text = "\n".join(row["content"] for row in rows)
    rag_chunks = chunk_text(
        rag_text,
        document_id="benchmark",
        chunk_size=100,
        chunk_overlap=20,
    )

    with tempfile.TemporaryDirectory(prefix="hakua-benchmark-") as temporary_root:
        memory, query = _prepare_memory(rows, Path(temporary_root))
        try:
            rag_stats = _measure(
                lambda: chunk_text(
                    rag_text,
                    document_id="benchmark",
                    chunk_size=100,
                    chunk_overlap=20,
                ),
                args.warmup,
                args.repetitions,
            )
            graph_stats = _measure(
                lambda: memory.search(query, top_k=5),
                args.warmup,
                args.repetitions,
            )
            ebbinghaus_stats = _measure(
                lambda: memory.recall(query, top_k=5),
                args.warmup,
                args.repetitions,
            )
            return {
                "metadata": {
                    "hakua_memory_version": _package_version(),
                    "git_commit_sha": _git_commit_sha(),
                    "timestamp_utc": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "os": platform.platform(),
                    "python_version": platform.python_version(),
                    "cpu": platform.processor() or platform.machine() or "not measured",
                    "dataset_id": args.dataset_id,
                    "seed": args.seed,
                    "number_of_samples": len(rows),
                    "warmup_count": args.warmup,
                    "measurement_repetitions": args.repetitions,
                },
                "measurements": {
                    "rag": {"operation": "chunk_text", "chunks": len(rag_chunks), **rag_stats},
                    "semantic_graph": {
                        "operation": "hybrid lexical search without dense backend",
                        **graph_stats,
                    },
                    "ebbinghaus": {"operation": "recall", **ebbinghaus_stats},
                },
            }
        finally:
            memory.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    payload = run_benchmark(args)
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        LOGGER.info("Wrote benchmark results to %s", args.output)
    sys.stdout.write(serialized)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
