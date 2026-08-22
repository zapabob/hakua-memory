from __future__ import annotations

import argparse
import json
import logging
import math
import platform
import random
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
DEFAULT_CHUNK_SIZE = 100
DEFAULT_CHUNK_OVERLAP = 20
DEFAULT_TOP_K = 5
DEFAULT_CONFIDENCE_LEVEL = 0.95


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run paired same-hardware comparisons with optional external libraries."
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--dataset-id", default="synthetic-business-v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    args = parser.parse_args(argv)
    if args.samples < 1 or args.warmup < 0 or args.repetitions < 2:
        parser.error("samples and repetitions must be positive; repetitions must be at least 2")
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


def _package_version(distribution: str) -> str:
    try:
        return version(distribution)
    except PackageNotFoundError:
        return "not measured"


def _hakua_version() -> str:
    installed = _package_version("hakua-memory")
    if installed != "not measured":
        return installed
    try:
        import tomllib

        with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
            return str(tomllib.load(stream)["project"]["version"])
    except (OSError, KeyError, TypeError):
        return "not measured"


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


def _git_worktree_clean() -> bool | str:
    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "not measured"
    return not result.stdout.strip()


def _require_external_libraries() -> dict[str, Any]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        from llama_index.core import Document
        from llama_index.core.node_parser import SentenceSplitter
        from rank_bm25 import BM25Okapi
        from scipy import stats as scipy_stats
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "external benchmark dependencies are missing; install "
            "benchmarks/requirements.txt in an isolated environment"
        ) from exc
    return {
        "BM25Okapi": BM25Okapi,
        "Document": Document,
        "RecursiveCharacterTextSplitter": RecursiveCharacterTextSplitter,
        "SentenceSplitter": SentenceSplitter,
        "scipy_stats": scipy_stats,
    }


def _tokenize(text: str) -> list[str]:
    import re

    return re.findall(r"[A-Za-z0-9_]+|[\u3040-\u30ff\u3400-\u9fff]", text.casefold())


def _measure_variants(
    operations: dict[str, Callable[[], Any]],
    *,
    warmup: int,
    repetitions: int,
    seed: int,
) -> dict[str, list[float]]:
    names = list(operations)
    durations = {name: [] for name in names}
    order = random.Random(seed)
    for _ in range(warmup):
        warmup_order = names[:]
        order.shuffle(warmup_order)
        for name in warmup_order:
            operations[name]()
    for _ in range(repetitions):
        repetition_order = names[:]
        order.shuffle(repetition_order)
        for name in repetition_order:
            started = time.perf_counter_ns()
            operations[name]()
            durations[name].append((time.perf_counter_ns() - started) / 1_000_000)
    return durations


def _summary(values: list[float], scipy_stats: Any) -> dict[str, float | int | list[float]]:
    mean = statistics.fmean(values)
    standard_error = scipy_stats.sem(values) if len(values) > 1 else 0.0
    critical_value = scipy_stats.t.ppf(0.5 + DEFAULT_CONFIDENCE_LEVEL / 2, len(values) - 1)
    margin = float(critical_value * standard_error)
    return {
        "sample_count": len(values),
        "mean": mean,
        "median": statistics.median(values),
        "stdev": statistics.stdev(values) if len(values) > 1 else 0.0,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
        "mean_ci_95": [mean - margin, mean + margin],
        "raw_durations_ms": values,
    }


def _paired_tests(primary: list[float], comparator: list[float], scipy_stats: Any) -> dict[str, float | str]:
    try:
        wilcoxon = scipy_stats.wilcoxon(
            primary,
            comparator,
            alternative="two-sided",
            method="auto",
        )
        wilcoxon_p = float(wilcoxon.pvalue)
    except (ValueError, RuntimeWarning):
        wilcoxon_p = "not measured"
    try:
        paired_t = scipy_stats.ttest_rel(primary, comparator)
        paired_t_p = float(paired_t.pvalue)
    except (ValueError, RuntimeWarning):
        paired_t_p = "not measured"
    return {
        "wilcoxon_statistic": float(wilcoxon.statistic) if wilcoxon_p != "not measured" else "not measured",
        "p_value_wilcoxon": wilcoxon_p,
        "paired_t_statistic": float(paired_t.statistic) if paired_t_p != "not measured" else "not measured",
        "p_value_paired_t": paired_t_p,
    }


def _apply_holm_correction(comparisons: list[dict[str, Any]]) -> None:
    measured = [
        (index, item["tests"]["p_value_wilcoxon"])
        for index, item in enumerate(comparisons)
        if isinstance(item["tests"]["p_value_wilcoxon"], float)
    ]
    ordered = sorted(measured, key=lambda pair: pair[1])
    adjusted: dict[int, float] = {}
    previous = 0.0
    total = len(ordered)
    for rank, (index, p_value) in enumerate(ordered):
        corrected = min(1.0, max(previous, (total - rank) * p_value))
        adjusted[index] = corrected
        previous = corrected
    for index, item in enumerate(comparisons):
        item["tests"]["p_value_wilcoxon_holm"] = adjusted.get(index, "not measured")


def _comparison(
    operation: str,
    variants: dict[str, list[float]],
    primary: str,
    comparators: Sequence[str],
    scipy_stats: Any,
    configurations: dict[str, dict[str, Any]],
    unit: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparisons: list[dict[str, Any]] = []
    for comparator in comparators:
        primary_values = variants[primary]
        comparator_values = variants[comparator]
        primary_summary = _summary(primary_values, scipy_stats)
        comparator_summary = _summary(comparator_values, scipy_stats)
        comparisons.append(
            {
                "operation": operation,
                "unit": unit,
                "primary": primary,
                "comparator": comparator,
                "configurations": {
                    primary: configurations[primary],
                    comparator: configurations[comparator],
                },
                "primary_minus_comparator_mean_ms": (
                    float(primary_summary["mean"]) - float(comparator_summary["mean"])
                ),
                "primary_minus_comparator_median_ms": (
                    float(primary_summary["median"]) - float(comparator_summary["median"])
                ),
                "comparator_mean_divided_by_primary_mean": (
                    float(comparator_summary["mean"]) / float(primary_summary["mean"])
                    if primary_summary["mean"]
                    else "not measured"
                ),
                "tests": _paired_tests(primary_values, comparator_values, scipy_stats),
            }
        )
    return (
        {
            "operation": operation,
            "unit": unit,
            "variants": {
                name: {
                    "configuration": configurations[name],
                    "summary": _summary(values, scipy_stats),
                }
                for name, values in variants.items()
            },
        },
        comparisons,
    )


def _chunk_validation(chunks: Any) -> dict[str, Any]:
    contents: list[str] = []
    for chunk in chunks:
        if isinstance(chunk, str):
            contents.append(chunk)
        elif hasattr(chunk, "content"):
            contents.append(str(chunk.content))
        else:
            contents.append(str(chunk.get_content()))
    return {
        "output_count": len(contents),
        "non_empty_outputs": sum(bool(content.strip()) for content in contents),
        "total_characters": sum(len(content) for content in contents),
    }


def _retrieval_validation(
    result_sets: list[list[Any]],
    rows: list[dict[str, str]],
    variant: str,
) -> dict[str, Any]:
    expected_ids = [row["id"] for row in rows]
    hits = 0
    for expected_id, result_set in zip(expected_ids, result_sets, strict=True):
        if variant == "hakua_memory":
            returned_ids = {
                str(item.get("node_id", "")).removeprefix("benchmark-") for item in result_set
            }
        else:
            returned_ids = {str(item.get("id", "")) for item in result_set}
        hits += expected_id in returned_ids
    return {
        "query_count": len(expected_ids),
        "queries_with_expected_top_k_hit": hits,
        "expected_hit_rate_at_top_k": hits / len(expected_ids) if expected_ids else "not measured",
    }


def _prepare_memory(rows: list[dict[str, str]], root: Path) -> Any:
    from hakua_memory.composite import CompositeMemory

    memory = CompositeMemory(root)
    for row in rows:
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
    return memory


def _build_plot(payload: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {
        "hakua_memory": "#245B9A",
        "langchain_text_splitters": "#C98A2E",
        "llama_index_core": "#A15B7A",
        "rank_bm25": "#5C7C5B",
    }
    markers = {"hakua_memory": "o", "langchain_text_splitters": "s", "llama_index_core": "^", "rank_bm25": "D"}
    comparisons = payload["measurements"]
    figure, axes = plt.subplots(1, len(comparisons), figsize=(12, 5), squeeze=False)
    for axis, (operation, measurement) in zip(axes[0], comparisons.items(), strict=True):
        names = list(measurement["variants"])
        x_positions = list(range(len(names)))
        means = [measurement["variants"][name]["summary"]["mean"] for name in names]
        intervals = [measurement["variants"][name]["summary"]["mean_ci_95"] for name in names]
        lower_errors = [mean - interval[0] for mean, interval in zip(means, intervals, strict=True)]
        upper_errors = [interval[1] - mean for mean, interval in zip(means, intervals, strict=True)]
        axis.errorbar(
            x_positions,
            means,
            yerr=[lower_errors, upper_errors],
            fmt="none",
            ecolor="#333333",
            capsize=5,
            linewidth=1.2,
        )
        for x_position, name, mean in zip(x_positions, names, means, strict=True):
            axis.plot(
                x_position,
                mean,
                marker=markers[name],
                color=colors[name],
                markersize=8,
                linestyle="none",
                label=name,
            )
        axis.set_xticks(x_positions, [name.replace("_", "\n") for name in names])
        axis.set_ylabel("Latency (ms)")
        axis.set_title(operation.replace("_", " ").title())
        axis.grid(axis="y", color="#D9DDE3", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=8, loc="best")
    figure.suptitle(
        "Same-hardware paired benchmark: mean latency with 95% CI\n"
        f"n={payload['metadata']['measurement_repetitions']} paired repetitions",
        fontsize=12,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    libraries = _require_external_libraries()
    scipy_stats = libraries["scipy_stats"]
    from hakua_memory.rag.chunking import chunk_text

    rows = generate_business_dataset(args.seed, args.samples)
    rag_text = "\n\n".join(row["content"] for row in rows)
    queries = [row["topic"] for row in rows]
    langchain_splitter = libraries["RecursiveCharacterTextSplitter"](
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    llama_document = libraries["Document"](text=rag_text)
    llama_splitter = libraries["SentenceSplitter"](
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )
    bm25_documents = [
        _tokenize(f"{row['topic']} {row['content']}")
        for row in rows
    ]
    bm25 = libraries["BM25Okapi"](bm25_documents)

    with tempfile.TemporaryDirectory(prefix="hakua-external-benchmark-") as temporary_root:
        memory = _prepare_memory(rows, Path(temporary_root))
        try:
            rag_operations = {
                "hakua_memory": lambda: chunk_text(
                    rag_text,
                    document_id="benchmark",
                    chunk_size=DEFAULT_CHUNK_SIZE,
                    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
                ),
                "langchain_text_splitters": lambda: langchain_splitter.split_text(rag_text),
                "llama_index_core": lambda: llama_splitter.get_nodes_from_documents(
                    [llama_document], show_progress=False
                ),
            }
            lexical_operations = {
                "hakua_memory": lambda: [memory.search(query, top_k=DEFAULT_TOP_K) for query in queries],
                "rank_bm25": lambda: [
                    bm25.get_top_n(_tokenize(query), rows, n=DEFAULT_TOP_K) for query in queries
                ],
            }
            rag_outputs = {name: operation() for name, operation in rag_operations.items()}
            lexical_outputs = {name: operation() for name, operation in lexical_operations.items()}
            rag_durations = _measure_variants(
                rag_operations,
                warmup=args.warmup,
                repetitions=args.repetitions,
                seed=args.seed,
            )
            lexical_durations = _measure_variants(
                lexical_operations,
                warmup=args.warmup,
                repetitions=args.repetitions,
                seed=args.seed + 1,
            )
            configurations = {
                "hakua_memory": {
                    "chunk_size": DEFAULT_CHUNK_SIZE,
                    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
                    "length_unit": "hakua estimated tokens",
                },
                "langchain_text_splitters": {
                    "class": "RecursiveCharacterTextSplitter",
                    "chunk_size": DEFAULT_CHUNK_SIZE,
                    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
                    "length_unit": "characters",
                },
                "llama_index_core": {
                    "class": "SentenceSplitter",
                    "chunk_size": DEFAULT_CHUNK_SIZE,
                    "chunk_overlap": DEFAULT_CHUNK_OVERLAP,
                    "length_unit": "tokens/characters according to library",
                },
                "rank_bm25": {
                    "class": "BM25Okapi",
                    "tokenizer": "same Unicode-aware regex as benchmark query preparation",
                },
            }
            rag_measurement, rag_comparisons = _comparison(
                "rag_chunking",
                rag_durations,
                "hakua_memory",
                ["langchain_text_splitters", "llama_index_core"],
                scipy_stats,
                configurations,
                "milliseconds per full-corpus split",
            )
            lexical_measurement, lexical_comparisons = _comparison(
                "lexical_retrieval",
                lexical_durations,
                "hakua_memory",
                ["rank_bm25"],
                scipy_stats,
                configurations,
                "milliseconds per complete query set",
            )
            comparisons = rag_comparisons + lexical_comparisons
            _apply_holm_correction(comparisons)
            payload: dict[str, Any] = {
                "schema_version": "1.0",
                "metadata": {
                    "hakua_memory_version": _hakua_version(),
                    "git_commit_sha": _git_commit_sha(),
                    "git_worktree_clean": _git_worktree_clean(),
                    "timestamp_utc": datetime.now(timezone.utc)
                    .isoformat()
                    .replace("+00:00", "Z"),
                    "os": platform.platform(),
                    "python_version": platform.python_version(),
                    "cpu": platform.processor() or platform.machine() or "not measured",
                    "gpu": "not measured",
                    "dataset_id": args.dataset_id,
                    "seed": args.seed,
                    "number_of_samples": len(rows),
                    "query_count": len(queries),
                    "warmup_count": args.warmup,
                    "measurement_repetitions": args.repetitions,
                    "randomized_variant_order": True,
                    "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
                    "statistical_test": "paired two-sided Wilcoxon signed-rank; paired t-test sensitivity",
                    "multiple_testing_correction": "Holm across paired Wilcoxon comparisons",
                },
                "libraries": {
                    "hakua_memory": {"distribution": "hakua-memory", "version": _hakua_version()},
                    "langchain_text_splitters": {
                        "distribution": "langchain-text-splitters",
                        "version": _package_version("langchain-text-splitters"),
                    },
                    "llama_index_core": {
                        "distribution": "llama-index-core",
                        "version": _package_version("llama-index-core"),
                    },
                    "rank_bm25": {"distribution": "rank-bm25", "version": _package_version("rank-bm25")},
                    "scipy": {"distribution": "scipy", "version": _package_version("scipy")},
                    "matplotlib": {"distribution": "matplotlib", "version": _package_version("matplotlib")},
                },
                "measurements": {
                    "rag_chunking": rag_measurement,
                    "lexical_retrieval": lexical_measurement,
                },
                "validation": {
                    "rag_chunking": {
                        name: _chunk_validation(output) for name, output in rag_outputs.items()
                    },
                    "lexical_retrieval": {
                        name: _retrieval_validation(output, rows, name)
                        for name, output in lexical_outputs.items()
                    },
                },
                "comparisons": comparisons,
                "unmeasured": [
                    "dense embedding retrieval",
                    "end-to-end RAG answer quality",
                    "Ebbinghaus recall equivalent in external libraries",
                    "memory footprint",
                    "GPU performance",
                ],
            }
            if args.plot:
                _build_plot(payload, args.plot)
                payload["plot"] = {"path": str(args.plot), "error_bars": "95% CI for the mean"}
            return payload
        finally:
            memory.close()


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    try:
        payload = run_benchmark(args)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc
    serialized = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(serialized, encoding="utf-8")
        LOGGER.info("Wrote external benchmark results to %s", args.output)
    sys.stdout.write(serialized)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
