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
DEFAULT_GGUF = Path(r"C:\Users\downl\Downloads\nsfw-bge-m3-v5-q6_k.gguf")
DEFAULT_EMBED_DIM = 1024


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Paired multi-group same-hardware comparison of hakua-memory against "
            "external RAG chunkers and CoG/retrieval libraries "
            "(LangChain BM25, LlamaIndex BM25, rank-bm25, FAISS+GGUF)."
        )
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--samples", type=int, default=40)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--dataset-id", default="synthetic-business-v1")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--plot", type=Path)
    parser.add_argument("--gguf", type=Path, default=DEFAULT_GGUF)
    parser.add_argument("--embed-dimensions", type=int, default=DEFAULT_EMBED_DIM)
    parser.add_argument("--n-gpu-layers", type=int, default=-1)
    parser.add_argument(
        "--skip-embedding",
        action="store_true",
        help="Skip GGUF hybrid and FAISS dense arms.",
    )
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
    missing: list[str] = []
    loaded: dict[str, Any] = {}
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        loaded["RecursiveCharacterTextSplitter"] = RecursiveCharacterTextSplitter
    except ModuleNotFoundError:
        missing.append("langchain-text-splitters")
    try:
        from langchain_community.retrievers import BM25Retriever as LangChainBM25Retriever
        from langchain_core.documents import Document as LCDocument

        loaded["LCDocument"] = LCDocument
        loaded["LangChainBM25Retriever"] = LangChainBM25Retriever
    except ModuleNotFoundError:
        missing.append("langchain-community / langchain-core")
    try:
        from llama_index.core import Document as LIDocument
        from llama_index.core.node_parser import SentenceSplitter
        from llama_index.core.schema import TextNode
        from llama_index.retrievers.bm25 import BM25Retriever as LlamaIndexBM25Retriever

        loaded["LIDocument"] = LIDocument
        loaded["SentenceSplitter"] = SentenceSplitter
        loaded["TextNode"] = TextNode
        loaded["LlamaIndexBM25Retriever"] = LlamaIndexBM25Retriever
    except ModuleNotFoundError:
        missing.append("llama-index-core / llama-index-retrievers-bm25")
    try:
        from rank_bm25 import BM25Okapi

        loaded["BM25Okapi"] = BM25Okapi
    except ModuleNotFoundError:
        missing.append("rank-bm25")
    try:
        import faiss
        import numpy as np
        from scipy import stats as scipy_stats

        loaded["faiss"] = faiss
        loaded["np"] = np
        loaded["scipy_stats"] = scipy_stats
    except ModuleNotFoundError:
        missing.append("faiss-cpu / scipy / numpy")
    if missing:
        raise RuntimeError(
            "external multi-group benchmark dependencies missing: "
            + ", ".join(missing)
            + "; install benchmarks/requirements.txt in an isolated environment"
        )
    return loaded


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


def _all_pairwise_comparisons(
    operation: str,
    variants: dict[str, list[float]],
    scipy_stats: Any,
    configurations: dict[str, dict[str, Any]],
    unit: str,
) -> list[dict[str, Any]]:
    """Emit every unordered pair for multi-group Holm correction."""
    names = list(variants)
    comparisons: list[dict[str, Any]] = []
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            left_summary = _summary(variants[left], scipy_stats)
            right_summary = _summary(variants[right], scipy_stats)
            comparisons.append(
                {
                    "operation": operation,
                    "unit": unit,
                    "primary": left,
                    "comparator": right,
                    "configurations": {
                        left: configurations[left],
                        right: configurations[right],
                    },
                    "primary_minus_comparator_mean_ms": (
                        float(left_summary["mean"]) - float(right_summary["mean"])
                    ),
                    "primary_minus_comparator_median_ms": (
                        float(left_summary["median"]) - float(right_summary["median"])
                    ),
                    "comparator_mean_divided_by_primary_mean": (
                        float(right_summary["mean"]) / float(left_summary["mean"])
                        if left_summary["mean"]
                        else "not measured"
                    ),
                    "tests": _paired_tests(variants[left], variants[right], scipy_stats),
                }
            )
    return comparisons


def _measurement_block(
    operation: str,
    variants: dict[str, list[float]],
    scipy_stats: Any,
    configurations: dict[str, dict[str, Any]],
    unit: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    comparisons = _all_pairwise_comparisons(
        operation, variants, scipy_stats, configurations, unit
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
            "multi_group_test": _friedman_test(variants, scipy_stats),
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
        elif hasattr(chunk, "page_content"):
            contents.append(str(chunk.page_content))
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
        returned_ids: set[str] = set()
        for item in result_set:
            if isinstance(item, dict):
                node_id = str(item.get("node_id", item.get("id", "")))
                returned_ids.add(node_id.removeprefix("benchmark-"))
                meta = item.get("metadata") or {}
                if isinstance(meta, dict) and meta.get("id"):
                    returned_ids.add(str(meta["id"]))
            else:
                meta = getattr(item, "metadata", None) or {}
                if isinstance(meta, dict) and meta.get("id"):
                    returned_ids.add(str(meta["id"]))
                node_id = getattr(item, "id_", None) or getattr(item, "node_id", None)
                if node_id:
                    returned_ids.add(str(node_id).removeprefix("benchmark-"))
        hits += expected_id in returned_ids
    return {
        "query_count": len(expected_ids),
        "queries_with_expected_top_k_hit": hits,
        "expected_hit_rate_at_top_k": hits / len(expected_ids) if expected_ids else "not measured",
        "variant": variant,
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


def _embed_nodes(memory: Any, backend: Any) -> tuple[dict[str, Any], list[list[float]], list[str]]:
    from hakua_memory.semantic_graph.embedding.serializer import (
        serialize_embedding_node,
        source_text_hash,
    )

    store = memory.semantic
    store.ensure_ready()
    with store._connect() as conn:
        nodes = [dict(row) for row in conn.execute("SELECT * FROM nodes").fetchall()]
    texts = [serialize_embedding_node(node) for node in nodes]
    started = time.perf_counter()
    vectors = backend.embed_documents(texts)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    ids: list[str] = []
    for node, text, vector in zip(nodes, texts, vectors, strict=True):
        node_id = str(node["node_id"])
        ids.append(node_id)
        store.upsert_node_embedding(
            node_id=node_id,
            identity=backend.identity,
            vector=vector,
            source_text_hash=source_text_hash(text),
        )
    meta = {
        "embedded_nodes": len(nodes),
        "dimensions": backend.identity.dimensions,
        "namespace": backend.identity.namespace,
        "wall_time_ms": elapsed_ms,
        "model_path": str(getattr(backend, "model_path", "")),
    }
    return meta, vectors, ids


class _FaissGgufRetriever:
    """Dense peer using the same GGUF embeddings + FAISS Inner Product index."""

    def __init__(
        self,
        *,
        backend: Any,
        vectors: list[list[float]],
        node_ids: list[str],
        rows: list[dict[str, str]],
        faiss: Any,
        np: Any,
        top_k: int,
    ) -> None:
        self._backend = backend
        self._node_ids = node_ids
        self._id_to_row = {
            f"benchmark-{row['id']}": row for row in rows
        }
        matrix = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        matrix = matrix / norms
        self._index = faiss.IndexFlatIP(matrix.shape[1])
        self._index.add(matrix)
        self._np = np
        self._top_k = top_k

    def search_all(self, queries: list[str]) -> list[list[dict[str, Any]]]:
        from hakua_memory.semantic_graph.embedding.serializer import serialize_embedding_query

        outputs: list[list[dict[str, Any]]] = []
        for query in queries:
            q = self._np.asarray(
                [self._backend.embed_query(serialize_embedding_query(query))],
                dtype=self._np.float32,
            )
            q_norm = self._np.linalg.norm(q, axis=1, keepdims=True)
            q_norm[q_norm == 0.0] = 1.0
            q = q / q_norm
            scores, indices = self._index.search(q, self._top_k)
            batch: list[dict[str, Any]] = []
            for score, idx in zip(scores[0], indices[0], strict=True):
                if int(idx) < 0:
                    continue
                node_id = self._node_ids[int(idx)]
                row = self._id_to_row.get(node_id, {})
                batch.append(
                    {
                        "node_id": node_id,
                        "id": str(row.get("id", node_id.removeprefix("benchmark-"))),
                        "score": float(score),
                        "metadata": {"id": row.get("id")},
                    }
                )
            outputs.append(batch)
        return outputs


def _build_plot(payload: dict[str, Any], path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = [
        "#245B9A",
        "#1B7F7A",
        "#C98A2E",
        "#A15B7A",
        "#5C7C5B",
        "#6B4E71",
        "#8C5A2E",
    ]
    comparisons = payload["measurements"]
    figure, axes = plt.subplots(
        1, len(comparisons), figsize=(4.6 * len(comparisons), 5.2), squeeze=False
    )
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
            capsize=4,
            linewidth=1.1,
        )
        for x_position, name, mean in zip(x_positions, names, means, strict=True):
            axis.plot(
                x_position,
                mean,
                marker="o",
                color=palette[x_position % len(palette)],
                markersize=7,
                linestyle="none",
                label=name,
            )
        axis.set_xticks(x_positions, [name.replace("_", "\n") for name in names], fontsize=7)
        axis.set_ylabel("Latency (ms)")
        title = operation.replace("_", " ").title()
        multi = measurement.get("multi_group_test") or {}
        if multi.get("applicable"):
            title += f"\nFriedman p={multi['p_value']:.3g}"
        axis.set_title(title, fontsize=10)
        axis.grid(axis="y", color="#D9DDE3", linewidth=0.7)
        axis.set_axisbelow(True)
        axis.spines[["top", "right"]].set_visible(False)
        axis.legend(fontsize=6, loc="best")
    figure.suptitle(
        "Multi-group paired benchmark vs external RAG/CoG libraries\n"
        f"mean latency ± 95% CI | n={payload['metadata']['measurement_repetitions']} | "
        "Friedman + pairwise Wilcoxon/Holm",
        fontsize=11,
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
    corpus_texts = [f"{row['topic']}\n{row['content']}" for row in rows]

    langchain_splitter = libraries["RecursiveCharacterTextSplitter"](
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        length_function=len,
        separators=["\n\n", "\n", " ", ""],
    )
    llama_document = libraries["LIDocument"](text=rag_text)
    llama_splitter = libraries["SentenceSplitter"](
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    )

    # --- External CoG/retrieval peers ---
    bm25 = libraries["BM25Okapi"]([_tokenize(text) for text in corpus_texts])
    lc_docs = [
        libraries["LCDocument"](page_content=text, metadata={"id": row["id"]})
        for text, row in zip(corpus_texts, rows, strict=True)
    ]
    langchain_bm25 = libraries["LangChainBM25Retriever"].from_documents(
        lc_docs,
        k=DEFAULT_TOP_K,
        preprocess_func=_tokenize,
    )
    li_nodes = [
        libraries["TextNode"](text=text, id_=f"benchmark-{row['id']}", metadata={"id": row["id"]})
        for text, row in zip(corpus_texts, rows, strict=True)
    ]
    llamaindex_bm25 = libraries["LlamaIndexBM25Retriever"].from_defaults(
        nodes=li_nodes,
        similarity_top_k=DEFAULT_TOP_K,
    )

    backend = None
    embedding_meta: dict[str, Any] | None = None
    faiss_retriever: _FaissGgufRetriever | None = None
    if not args.skip_embedding:
        if not args.gguf.is_file():
            raise RuntimeError(f"GGUF embedding model not found: {args.gguf}")
        from hakua_memory.semantic_graph.embedding.gguf_local import (
            LlamaCppPythonEmbeddingBackend,
        )

        backend = LlamaCppPythonEmbeddingBackend(
            model_path=args.gguf,
            dimensions=args.embed_dimensions,
            model=args.gguf.stem,
            revision="q6_k",
            n_ctx=512,
            n_gpu_layers=args.n_gpu_layers,
        )

    with tempfile.TemporaryDirectory(prefix="hakua-external-benchmark-") as temporary_root:
        memory = _prepare_memory(rows, Path(temporary_root))
        try:
            if backend is not None:
                embedding_meta, vectors, node_ids = _embed_nodes(memory, backend)
                faiss_retriever = _FaissGgufRetriever(
                    backend=backend,
                    vectors=vectors,
                    node_ids=node_ids,
                    rows=rows,
                    faiss=libraries["faiss"],
                    np=libraries["np"],
                    top_k=DEFAULT_TOP_K,
                )

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

            cog_operations: dict[str, Callable[[], Any]] = {
                "hakua_cog_lexical": lambda: [
                    memory.search(query, top_k=DEFAULT_TOP_K) for query in queries
                ],
                "rank_bm25": lambda: [
                    [
                        {"id": item["id"], "metadata": {"id": item["id"]}}
                        for item in bm25.get_top_n(_tokenize(query), rows, n=DEFAULT_TOP_K)
                    ]
                    for query in queries
                ],
                "langchain_bm25": lambda: [
                    langchain_bm25.invoke(query) for query in queries
                ],
                "llamaindex_bm25": lambda: [
                    llamaindex_bm25.retrieve(query) for query in queries
                ],
            }
            if backend is not None:
                cog_operations["hakua_hybrid_gguf"] = lambda: [
                    memory.search(query, top_k=DEFAULT_TOP_K, backend=backend)
                    for query in queries
                ]
            if faiss_retriever is not None:
                cog_operations["faiss_gguf"] = lambda: faiss_retriever.search_all(queries)

            rag_outputs = {name: operation() for name, operation in rag_operations.items()}
            cog_outputs = {name: operation() for name, operation in cog_operations.items()}
            rag_durations = _measure_variants(
                rag_operations,
                warmup=args.warmup,
                repetitions=args.repetitions,
                seed=args.seed,
            )
            cog_durations = _measure_variants(
                cog_operations,
                warmup=args.warmup,
                repetitions=args.repetitions,
                seed=args.seed + 1,
            )

            configurations = {
                "hakua_memory": {
                    "family": "hakua-memory",
                    "role": "RAG chunker",
                    "chunk_size": DEFAULT_CHUNK_SIZE,
                    "length_unit": "hakua estimated tokens",
                },
                "langchain_text_splitters": {
                    "family": "LangChain",
                    "role": "RAG chunker",
                    "class": "RecursiveCharacterTextSplitter",
                    "length_unit": "characters",
                },
                "llama_index_core": {
                    "family": "LlamaIndex",
                    "role": "RAG chunker",
                    "class": "SentenceSplitter",
                },
                "hakua_cog_lexical": {
                    "family": "hakua-memory",
                    "role": "CoG lexical retrieval",
                    "engine": "SQLite FTS5 trigram + rank",
                },
                "hakua_hybrid_gguf": {
                    "family": "hakua-memory",
                    "role": "CoG hybrid retrieval",
                    "engine": "lexical + dense RRF",
                    "gguf": str(args.gguf),
                    "dimensions": args.embed_dimensions,
                },
                "rank_bm25": {
                    "family": "rank-bm25",
                    "role": "lexical retrieval peer",
                    "class": "BM25Okapi",
                },
                "langchain_bm25": {
                    "family": "LangChain",
                    "role": "RAG/CoG lexical retriever peer",
                    "class": "BM25Retriever",
                },
                "llamaindex_bm25": {
                    "family": "LlamaIndex",
                    "role": "RAG/CoG lexical retriever peer",
                    "class": "BM25Retriever",
                },
                "faiss_gguf": {
                    "family": "FAISS",
                    "role": "dense retrieval peer (same GGUF embeddings)",
                    "index": "IndexFlatIP",
                    "gguf": str(args.gguf),
                },
            }

            rag_measurement, rag_comparisons = _measurement_block(
                "rag_chunking",
                rag_durations,
                scipy_stats,
                configurations,
                "milliseconds per full-corpus split",
            )
            cog_measurement, cog_comparisons = _measurement_block(
                "cog_retrieval",
                cog_durations,
                scipy_stats,
                configurations,
                "milliseconds per complete query set",
            )
            comparisons = rag_comparisons + cog_comparisons
            _apply_holm_correction(comparisons)

            payload: dict[str, Any] = {
                "schema_version": "1.2",
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
                    "gpu": (
                        "CUDA via llama-cpp n_gpu_layers"
                        if backend is not None and args.n_gpu_layers != 0
                        else "not measured"
                    ),
                    "dataset_id": args.dataset_id,
                    "seed": args.seed,
                    "number_of_samples": len(rows),
                    "query_count": len(queries),
                    "warmup_count": args.warmup,
                    "measurement_repetitions": args.repetitions,
                    "randomized_variant_order": True,
                    "confidence_level": DEFAULT_CONFIDENCE_LEVEL,
                    "statistical_test": (
                        "Friedman chi-square for multi-group paired latencies; "
                        "all pairwise two-sided Wilcoxon signed-rank; "
                        "paired t-test sensitivity"
                    ),
                    "multiple_testing_correction": "Holm across all pairwise Wilcoxon comparisons",
                    "embedding": embedding_meta,
                    "comparison_scope": (
                        "RAG chunking vs LangChain/LlamaIndex; "
                        "CoG/retrieval vs rank-bm25, LangChain BM25, LlamaIndex BM25, "
                        "FAISS+same GGUF embeddings, hakua lexical/hybrid"
                    ),
                },
                "libraries": {
                    name: {"distribution": name, "version": _package_version(name)}
                    for name in (
                        "hakua-memory",
                        "langchain-text-splitters",
                        "langchain-community",
                        "langchain-core",
                        "llama-index-core",
                        "llama-index-retrievers-bm25",
                        "rank-bm25",
                        "faiss-cpu",
                        "llama-cpp-python",
                        "scipy",
                        "matplotlib",
                        "numpy",
                    )
                },
                "measurements": {
                    "rag_chunking": rag_measurement,
                    "cog_retrieval": cog_measurement,
                },
                "validation": {
                    "rag_chunking": {
                        name: _chunk_validation(output) for name, output in rag_outputs.items()
                    },
                    "cog_retrieval": {
                        name: _retrieval_validation(output, rows, name)
                        for name, output in cog_outputs.items()
                    },
                },
                "comparisons": comparisons,
                "unmeasured": [
                    "end-to-end answer quality / LLM judge",
                    "mem0 / zep / other hosted memory SaaS APIs",
                    "memory footprint RSS",
                    "NetworkX property-graph equivalents",
                ],
            }
            if args.plot:
                _build_plot(payload, args.plot)
                payload["plot"] = {
                    "path": str(args.plot),
                    "error_bars": "95% CI for the mean",
                }
            return payload
        finally:
            memory.close()
            if backend is not None:
                closer = getattr(backend, "close", None)
                if callable(closer):
                    closer()


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
        LOGGER.info("Wrote external multi-group benchmark results to %s", args.output)
    sys.stdout.write(serialized)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
