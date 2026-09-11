#!/usr/bin/env python3
"""Three-arm Friedman benchmark: lexical / always-hybrid / speculative-hybrid.

Sweeps margin thresholds on the same synthetic corpus, picks a data-driven
threshold (Pareto: keep hit-rate near always-hybrid, minimize latency), then
runs paired Friedman + Wilcoxon/Holm on the three modes at that threshold.
"""

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
DEFAULT_TOP_K = 5
DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_GGUF = Path(r"C:\Users\downl\Downloads\nsfw-bge-m3-v5-q6_k.gguf")
DEFAULT_EMBED_DIM = 1024
DEFAULT_MIN_TOP_SCORES = (0.40, 0.50, 0.55, 0.60, 0.70, 0.85, 0.95)
DEFAULT_MARGIN_THRESHOLD = 0.0
HIT_RATE_TOLERANCE = 0.05


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Isolated 3-group Friedman: lexical-only vs always-hybrid vs "
            "speculative-hybrid(threshold). Sweep thresholds and pick from data."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--dataset-id", default="synthetic-business-v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--embed-dimensions", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument(
        "--margin-threshold",
        type=float,
        default=DEFAULT_MARGIN_THRESHOLD,
        help="Secondary gate; absolute final_score margins often collapse to ~0.",
    )
    parser.add_argument(
        "--min-top-scores",
        type=float,
        nargs="+",
        default=list(DEFAULT_MIN_TOP_SCORES),
        help="Primary speculative gates to sweep (top-1 final_score thresholds).",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=None,
        help=argparse.SUPPRESS,  # backward alias ignored; use --min-top-scores
    )
    args = parser.parse_args(argv)
    if args.samples < 1 or args.warmup < 0 or args.repetitions < 2:
        parser.error("samples/repetitions invalid")
    if not args.min_top_scores:
        parser.error("at least one min-top-score required")
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
        "mean_ci_95": [mean - margin, mean + margin],
        "raw_durations_ms": values,
    }


def _paired_tests(primary: list[float], comparator: list[float], scipy_stats: Any) -> dict[str, float | str]:
    try:
        wilcoxon = scipy_stats.wilcoxon(
            primary, comparator, alternative="two-sided", method="auto"
        )
        wilcoxon_p = float(wilcoxon.pvalue)
        wilcoxon_stat = float(wilcoxon.statistic)
    except (ValueError, RuntimeWarning):
        wilcoxon_p = "not measured"
        wilcoxon_stat = "not measured"
    try:
        paired_t = scipy_stats.ttest_rel(primary, comparator)
        paired_t_p = float(paired_t.pvalue)
        paired_t_stat = float(paired_t.statistic)
    except (ValueError, RuntimeWarning):
        paired_t_p = "not measured"
        paired_t_stat = "not measured"
    return {
        "wilcoxon_statistic": wilcoxon_stat,
        "p_value_wilcoxon": wilcoxon_p,
        "paired_t_statistic": paired_t_stat,
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


def _friedman_test(variants: dict[str, list[float]], scipy_stats: Any) -> dict[str, Any]:
    names = list(variants)
    if len(names) < 3:
        return {"applicable": False, "reason": "Friedman requires >= 3 paired groups"}
    samples = [variants[name] for name in names]
    if len({len(sample) for sample in samples}) != 1:
        return {"applicable": False, "reason": "unequal paired sample lengths"}
    try:
        statistic, p_value = scipy_stats.friedmanchisquare(*samples)
    except ValueError as exc:
        return {"applicable": False, "reason": str(exc)}
    return {
        "applicable": True,
        "test": "Friedman chi-square (paired multi-group)",
        "groups": names,
        "statistic": float(statistic),
        "p_value": float(p_value),
        "alpha": 0.05,
        "reject_null_at_0_05": bool(float(p_value) < 0.05),
    }


def _measure_variants(
    operations: dict[str, Callable[[], Any]],
    *,
    warmup: int,
    repetitions: int,
    seed: int,
) -> dict[str, list[float]]:
    names = list(operations)
    durations = {name: [] for name in names}
    for _ in range(warmup):
        for name in names:
            operations[name]()
    rng = random.Random(seed)
    for _ in range(repetitions):
        order = names[:]
        rng.shuffle(order)
        for name in order:
            started = time.perf_counter_ns()
            operations[name]()
            durations[name].append((time.perf_counter_ns() - started) / 1_000_000)
    return durations


def _prepare_memory(rows: list[dict[str, str]], root: Path) -> Any:
    from hakua_memory.composite import CompositeMemory

    memory = CompositeMemory(root, enable_rag=False)
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


def _embed_nodes(memory: Any, backend: Any) -> dict[str, Any]:
    from hakua_memory.semantic_graph.embedding.serializer import (
        serialize_embedding_node,
        source_text_hash,
    )

    store = memory.semantic
    store.ensure_ready()
    nodes = store.list_nodes(limit=10_000)
    texts = [serialize_embedding_node(node) for node in nodes]
    started = time.perf_counter()
    vectors = backend.embed_documents(texts)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    for node, text, vector in zip(nodes, texts, vectors, strict=True):
        store.upsert_node_embedding(
            node_id=str(node["node_id"]),
            identity=backend.identity,
            vector=vector,
            source_text_hash=source_text_hash(text),
        )
    return {
        "embedded_nodes": len(nodes),
        "dimensions": backend.identity.dimensions,
        "namespace": backend.identity.namespace,
        "wall_time_ms": elapsed_ms,
        "model_path": str(getattr(backend, "model_path", "")),
    }


def _hit_rate(result_sets: list[list[dict[str, Any]]], rows: list[dict[str, str]]) -> float:
    hits = 0
    for expected, results in zip(rows, result_sets, strict=True):
        expected_id = expected["id"]
        returned = {
            str(item.get("node_id", "")).removeprefix("benchmark-") for item in results
        }
        hits += int(expected_id in returned)
    return hits / len(rows) if rows else 0.0


def _path_stats(result_sets: list[list[dict[str, Any]]]) -> dict[str, float | int]:
    early = 0
    verify = 0
    for results in result_sets:
        if not results:
            continue
        path = str(results[0].get("speculative_path") or "")
        if path == "lexical_early_accept":
            early += 1
        elif path == "hybrid_verify":
            verify += 1
    total = early + verify
    return {
        "queries": len(result_sets),
        "early_accept": early,
        "hybrid_verify": verify,
        "early_accept_rate": early / total if total else 0.0,
    }


def _evaluate_once(
    memory: Any,
    backend: Any,
    queries: list[str],
    rows: list[dict[str, str]],
    *,
    mode: str,
    margin_threshold: float,
    min_top_score: float,
) -> dict[str, Any]:
    started = time.perf_counter_ns()
    if mode == "lexical_only":
        result_sets = [memory.search(q, top_k=DEFAULT_TOP_K) for q in queries]
    elif mode == "always_hybrid":
        result_sets = [
            memory.search(q, top_k=DEFAULT_TOP_K, backend=backend) for q in queries
        ]
    elif mode == "speculative_hybrid":
        result_sets = [
            memory.search(
                q,
                top_k=DEFAULT_TOP_K,
                backend=backend,
                speculative=True,
                margin_threshold=margin_threshold,
                min_top_score=min_top_score,
            )
            for q in queries
        ]
    else:
        raise ValueError(f"unknown mode: {mode}")
    elapsed_ms = (time.perf_counter_ns() - started) / 1_000_000
    payload: dict[str, Any] = {
        "latency_ms": elapsed_ms,
        "hit_rate_at_top_k": _hit_rate(result_sets, rows),
        "result_sets": result_sets,
    }
    if mode == "speculative_hybrid":
        payload["path_stats"] = _path_stats(result_sets)
    return payload


def _pick_threshold(
    sweep: list[dict[str, Any]],
    *,
    hybrid_hit_rate: float,
    hybrid_latency_ms: float,
    lexical_latency_ms: float,
) -> dict[str, Any]:
    """Prefer hit-rate near hybrid; then min latency among eligible rows.

    When several gates share the same quality, also record the interior
    early-accept (~50%) candidate for non-degenerate speculative behavior.
    """
    span = max(1e-9, hybrid_latency_ms - lexical_latency_ms)
    eligible = [
        row
        for row in sweep
        if float(row["hit_rate_at_top_k"]) + 1e-12 >= hybrid_hit_rate - HIT_RATE_TOLERANCE
    ]
    if eligible:
        chosen = min(eligible, key=lambda row: float(row["latency_ms"]))
        rule = (
            f"min_latency among hit_rate >= hybrid-{HIT_RATE_TOLERANCE:.2f} "
            f"(hybrid_hit={hybrid_hit_rate:.4f})"
        )
    else:

        def score(row: dict[str, Any]) -> float:
            latency_norm = (float(row["latency_ms"]) - lexical_latency_ms) / span
            return float(row["hit_rate_at_top_k"]) - 0.15 * latency_norm

        chosen = max(sweep, key=score)
        rule = "max(hit_rate - 0.15 * latency_norm) fallback (no threshold within tolerance)"

    interior_pool = [
        row
        for row in sweep
        if 0.05 < float((row.get("path_stats") or {}).get("early_accept_rate") or -1) < 0.95
    ]
    interior = None
    if interior_pool:
        interior = min(
            interior_pool,
            key=lambda row: abs(
                float((row.get("path_stats") or {}).get("early_accept_rate") or 0.0) - 0.5
            ),
        )

    return {
        "min_top_score": chosen["min_top_score"],
        "margin_threshold": chosen["margin_threshold"],
        "selection_rule": rule,
        "selected_row": {
            key: value for key, value in chosen.items() if key != "result_sets"
        },
        "interior_early_accept_row": (
            {key: value for key, value in interior.items() if key != "result_sets"}
            if interior is not None
            else None
        ),
        "hit_rate_tolerance": HIT_RATE_TOLERANCE,
        "note": (
            "Primary gate is min_top_score; absolute final_score margins often "
            "collapse to ~0 on uniform confidence/salience corpora."
        ),
    }


def _write_plot(payload: dict[str, Any], path: Path) -> None:
    import matplotlib.pyplot as plt

    measurement = payload["measurements"]["retrieval_latency_ms"]
    variants = measurement["variants"]
    names = list(variants)
    means = [float(variants[name]["summary"]["mean"]) for name in names]
    cis = [variants[name]["summary"]["mean_ci_95"] for name in names]
    yerr = [
        [mean - float(ci[0]), float(ci[1]) - mean]
        for mean, ci in zip(means, cis, strict=True)
    ]
    x_positions = list(range(len(names)))
    figure, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    axis = axes[0]
    axis.errorbar(
        x_positions,
        means,
        yerr=list(zip(*yerr, strict=True)),
        fmt="o",
        color="#1f4e79",
        ecolor="#333333",
        capsize=5,
        linewidth=1.2,
    )
    axis.set_xticks(x_positions, [name.replace("_", "\n") for name in names], fontsize=8)
    axis.set_ylabel("Latency (ms)")
    multi = measurement.get("multi_group_test") or {}
    title = "3-group latency"
    if multi.get("applicable"):
        title += f"\nFriedman p={multi['p_value']:.3g}"
    axis.set_title(title)
    axis.grid(axis="y", color="#D9DDE3", linewidth=0.7)
    axis.set_axisbelow(True)
    axis.spines[["top", "right"]].set_visible(False)

    sweep = payload["threshold_sweep"]["rows"]
    axis2 = axes[1]
    xs = [float(row["min_top_score"]) for row in sweep]
    latencies = [float(row["latency_ms"]) for row in sweep]
    hits = [float(row["hit_rate_at_top_k"]) for row in sweep]
    axis2.plot(xs, latencies, marker="o", color="#1f4e79", label="Latency (ms)")
    twin = axis2.twinx()
    twin.plot(xs, hits, marker="s", color="#c45c26", label="Hit rate@k")
    chosen = float(payload["threshold_selection"]["min_top_score"])
    axis2.axvline(chosen, color="#666666", linestyle="--", linewidth=1.0, label="chosen")
    axis2.set_xlabel("min_top_score gate")
    axis2.set_ylabel("Latency (ms)")
    twin.set_ylabel("Hit rate @ top-k")
    axis2.set_title("min_top_score sweep")
    axis2.grid(axis="y", color="#D9DDE3", linewidth=0.7)
    lines_a, labels_a = axis2.get_legend_handles_labels()
    lines_b, labels_b = twin.get_legend_handles_labels()
    axis2.legend(lines_a + lines_b, labels_a + labels_b, fontsize=7, loc="best")
    axis2.spines[["top"]].set_visible(False)

    figure.suptitle(
        "Speculative-hybrid 3-group Friedman | mean ± 95% CI\n"
        f"chosen min_top_score={chosen}",
        fontsize=11,
    )
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    try:
        from scipy import stats as scipy_stats
    except ImportError as exc:
        raise RuntimeError("scipy is required; install benchmarks/requirements.txt") from exc

    if not args.gguf.is_file():
        raise RuntimeError(f"GGUF embedding model not found: {args.gguf}")

    from hakua_memory.semantic_graph.embedding.gguf_local import (
        LlamaCppPythonEmbeddingBackend,
    )

    rows = generate_business_dataset(args.seed, args.samples)
    queries = [row["topic"] for row in rows]

    backend = LlamaCppPythonEmbeddingBackend(
        model_path=args.gguf,
        dimensions=args.embed_dimensions,
        model=args.gguf.stem,
        revision="q6_k",
        n_ctx=512,
        n_gpu_layers=args.n_gpu_layers,
    )

    with tempfile.TemporaryDirectory(prefix="hakua-spec-hybrid-bench-") as temporary_root:
        memory = _prepare_memory(rows, Path(temporary_root))
        try:
            embedding_meta = _embed_nodes(memory, backend)

            LOGGER.info(
                "Running min_top_score sweep over %s (margin=%s)",
                args.min_top_scores,
                args.margin_threshold,
            )
            lexical_probe = _evaluate_once(
                memory,
                backend,
                queries,
                rows,
                mode="lexical_only",
                margin_threshold=float(args.margin_threshold),
                min_top_score=0.0,
            )
            hybrid_probe = _evaluate_once(
                memory,
                backend,
                queries,
                rows,
                mode="always_hybrid",
                margin_threshold=float(args.margin_threshold),
                min_top_score=0.0,
            )

            sweep_rows: list[dict[str, Any]] = []
            for min_top in args.min_top_scores:
                evaluation = _evaluate_once(
                    memory,
                    backend,
                    queries,
                    rows,
                    mode="speculative_hybrid",
                    margin_threshold=float(args.margin_threshold),
                    min_top_score=float(min_top),
                )
                sweep_rows.append(
                    {
                        "margin_threshold": float(args.margin_threshold),
                        "min_top_score": float(min_top),
                        "latency_ms": evaluation["latency_ms"],
                        "hit_rate_at_top_k": evaluation["hit_rate_at_top_k"],
                        "path_stats": evaluation.get("path_stats"),
                    }
                )
                LOGGER.info(
                    "min_top=%.3f latency=%.1fms hit=%.3f early=%.2f",
                    min_top,
                    evaluation["latency_ms"],
                    evaluation["hit_rate_at_top_k"],
                    (evaluation.get("path_stats") or {}).get("early_accept_rate", 0.0),
                )

            selection = _pick_threshold(
                sweep_rows,
                hybrid_hit_rate=float(hybrid_probe["hit_rate_at_top_k"]),
                hybrid_latency_ms=float(hybrid_probe["latency_ms"]),
                lexical_latency_ms=float(lexical_probe["latency_ms"]),
            )
            chosen_min_top = float(selection["min_top_score"])
            chosen_margin = float(selection["margin_threshold"])
            LOGGER.info(
                "Chosen min_top_score=%.3f via %s",
                chosen_min_top,
                selection["selection_rule"],
            )

            operations: dict[str, Callable[[], Any]] = {
                "lexical_only": lambda: [
                    memory.search(q, top_k=DEFAULT_TOP_K) for q in queries
                ],
                "always_hybrid": lambda: [
                    memory.search(q, top_k=DEFAULT_TOP_K, backend=backend) for q in queries
                ],
                "speculative_hybrid": lambda: [
                    memory.search(
                        q,
                        top_k=DEFAULT_TOP_K,
                        backend=backend,
                        speculative=True,
                        margin_threshold=chosen_margin,
                        min_top_score=chosen_min_top,
                    )
                    for q in queries
                ],
            }

            # Quality snapshot once (not timed) for chosen threshold
            lexical_quality = [
                memory.search(q, top_k=DEFAULT_TOP_K) for q in queries
            ]
            hybrid_quality = [
                memory.search(q, top_k=DEFAULT_TOP_K, backend=backend) for q in queries
            ]
            speculative_quality = [
                memory.search(
                    q,
                    top_k=DEFAULT_TOP_K,
                    backend=backend,
                    speculative=True,
                    margin_threshold=chosen_margin,
                    min_top_score=chosen_min_top,
                )
                for q in queries
            ]
            quality = {
                "lexical_only": {
                    "hit_rate_at_top_k": _hit_rate(lexical_quality, rows),
                },
                "always_hybrid": {
                    "hit_rate_at_top_k": _hit_rate(hybrid_quality, rows),
                },
                "speculative_hybrid": {
                    "hit_rate_at_top_k": _hit_rate(speculative_quality, rows),
                    "path_stats": _path_stats(speculative_quality),
                },
            }

            durations = _measure_variants(
                operations,
                warmup=args.warmup,
                repetitions=args.repetitions,
                seed=args.seed,
            )
            configurations = {
                "lexical_only": {
                    "family": "hakua-memory",
                    "role": "lexical FTS + rank",
                    "embedding": False,
                },
                "always_hybrid": {
                    "family": "hakua-memory",
                    "role": "lexical + dense RRF",
                    "embedding": True,
                    "gguf": str(args.gguf),
                },
                "speculative_hybrid": {
                    "family": "hakua-memory",
                    "role": "draft lexical / verify hybrid",
                    "embedding": "conditional",
                    "margin_threshold": chosen_margin,
                    "min_top_score": chosen_min_top,
                    "gguf": str(args.gguf),
                },
            }
            comparisons: list[dict[str, Any]] = []
            names = list(durations)
            for i, left in enumerate(names):
                for right in names[i + 1 :]:
                    left_summary = _summary(durations[left], scipy_stats)
                    right_summary = _summary(durations[right], scipy_stats)
                    comparisons.append(
                        {
                            "operation": "retrieval_latency_ms",
                            "unit": "ms",
                            "primary": left,
                            "comparator": right,
                            "configurations": {
                                left: configurations[left],
                                right: configurations[right],
                            },
                            "primary_minus_comparator_mean_ms": (
                                float(left_summary["mean"]) - float(right_summary["mean"])
                            ),
                            "tests": _paired_tests(
                                durations[left], durations[right], scipy_stats
                            ),
                        }
                    )
            _apply_holm_correction(comparisons)

            measurement = {
                "operation": "retrieval_latency_ms",
                "unit": "ms",
                "variants": {
                    name: {
                        "configuration": configurations[name],
                        "summary": _summary(values, scipy_stats),
                        "quality": quality[name],
                    }
                    for name, values in durations.items()
                },
                "multi_group_test": _friedman_test(durations, scipy_stats),
            }

            payload = {
                "metadata": {
                    "generated_at_utc": datetime.now(timezone.utc).isoformat(),
                    "dataset_id": args.dataset_id,
                    "seed": args.seed,
                    "samples": args.samples,
                    "warmup": args.warmup,
                    "measurement_repetitions": args.repetitions,
                    "top_k": DEFAULT_TOP_K,
                    "hakua_memory_version": _hakua_version(),
                    "git_commit": _git_commit_sha(),
                    "python": sys.version.split()[0],
                    "platform": platform.platform(),
                    "scipy_version": _package_version("scipy"),
                    "matplotlib_version": _package_version("matplotlib"),
                    "llama_cpp_python_version": _package_version("llama-cpp-python"),
                    "gguf": str(args.gguf),
                    "embed_dimensions": args.embed_dimensions,
                    "n_gpu_layers": args.n_gpu_layers,
                    "margin_threshold": float(args.margin_threshold),
                    "min_top_scores_swept": [float(value) for value in args.min_top_scores],
                    "chosen_min_top_score": chosen_min_top,
                    "embedding_index": embedding_meta,
                },
                "baseline_probes": {
                    "lexical_only": {
                        "latency_ms": lexical_probe["latency_ms"],
                        "hit_rate_at_top_k": lexical_probe["hit_rate_at_top_k"],
                    },
                    "always_hybrid": {
                        "latency_ms": hybrid_probe["latency_ms"],
                        "hit_rate_at_top_k": hybrid_probe["hit_rate_at_top_k"],
                    },
                },
                "threshold_sweep": {"rows": sweep_rows},
                "threshold_selection": selection,
                "measurements": {"retrieval_latency_ms": measurement},
                "pairwise_comparisons": comparisons,
            }
            return payload
        finally:
            memory.close()
            if hasattr(backend, "close"):
                backend.close()


def main(argv: Sequence[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args(argv)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = args.output or (
        REPO_ROOT / "benchmarks" / "results" / f"speculative_hybrid_3group_{stamp}.json"
    )
    plot = args.plot or (
        REPO_ROOT / "benchmarks" / "results" / f"speculative_hybrid_3group_{stamp}.png"
    )
    payload = run_benchmark(args)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    try:
        _write_plot(payload, plot)
        LOGGER.info("Wrote plot %s", plot)
    except Exception as exc:  # noqa: BLE001 — plot is best-effort
        LOGGER.warning("Plot skipped: %s", exc)
        plot = None
    chosen = payload["threshold_selection"]["min_top_score"]
    friedman = payload["measurements"]["retrieval_latency_ms"]["multi_group_test"]
    LOGGER.info("Wrote %s", output)
    LOGGER.info("Chosen min_top_score=%.3f Friedman=%s", chosen, friedman)
    print(json.dumps({"output": str(output), "plot": str(plot) if plot else None, "chosen_min_top_score": chosen, "friedman": friedman}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
