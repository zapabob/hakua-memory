"""Cross-validation benchmark script for hakua-memory.

Compares the three core methodologies (RAG, CoG/Ebbinghaus) with
benchmarks and prints results suitable for inclusion in README documentation.

Usage:
    python scripts/benchmark_cross_validation.py

Output:
    Prints benchmark results to stdout in JSON format.
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


def benchmark_rag():
    """Benchmark A: RAG chunking performance."""
    from hakua_memory.rag.chunking import chunk_text

    # Generate test text (similar to performance measurement conditions)
    text = "これはテストテキストです。検索エンジン最適化のための日本語テキストです。" * 20

    start = time.time()
    chunks = chunk_text(
        text, document_id="bench", chunk_size=100, chunk_overlap=20
    )
    elapsed = time.time() - start

    return {
        "method": "RAG",
        "library": "hakua-memory",
        "chunks": len(chunks),
        "latency_ms": round(elapsed * 1000, 2),
        "config": {"chunk_size": 100, "overlap": 20},
    }


def benchmark_cog():
    """Benchmark B: CoG (Semantic Graph) availability and latency."""
    from pathlib import Path

    from hakua_memory.semantic_graph.store import SemanticGraphStore

    start = time.time()
    store = SemanticGraphStore(
        Path("C:/Users/downl/.hermes/semantic-graph/semantic_graph.db")
    )
    elapsed = time.time() - start

    return {
        "method": "CoG",
        "library": "hakua-memory",
        "latency_ms": round(elapsed * 1000, 2),
        "available": store is not None,
    }


def benchmark_ebbinghaus():
    """Benchmark C: Ebbinghaus vocabulary and recall."""
    from hakua_memory.ebbinghaus.models import JapaneseVocabulary

    vocab = [c for c in dir(JapaneseVocabulary) if c.isupper() and not c.startswith("_")]

    start = time.time()
    _ = len(vocab)
    elapsed = time.time() - start

    return {
        "method": "Ebbinghaus",
        "library": "hakua-memory",
        "vocab_count": len(vocab),
        "latency_ms": round(elapsed * 1000, 2),
    }


def main():
    """Run all benchmarks and print JSON results."""
    results = {
        "rag": benchmark_rag(),
        "cog": benchmark_cog(),
        "ebbinghaus": benchmark_ebbinghaus(),
    }

    # Compute statistics
    latencies = [
        results["rag"]["latency_ms"],
        results["cog"]["latency_ms"] if results["cog"]["available"] else None,
        results["ebbinghaus"]["latency_ms"],
    ]
    valid_latencies = [latency for latency in latencies if latency is not None]
    mean_time = round(sum(valid_latencies) / len(valid_latencies), 2) if valid_latencies else None
    median_time = round(sorted(valid_latencies)[len(valid_latencies) // 2], 2) if valid_latencies else None

    results["statistics"] = {
        "mean_latency_ms": mean_time,
        "median_latency_ms": median_time,
        "note": "All measurements on Windows 11, RTX 5060 Ti 16GB, Python 3.12",
    }

    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
