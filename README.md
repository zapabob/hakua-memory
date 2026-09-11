# hakua-memory

> **TL;DR:** `hakua-memory` is a self-contained Python memory library for autonomous agents. It combines SQLite-backed Ebbinghaus memory, a structured semantic graph (CoG), optional embedding-based retrieval, RAG (Retrieval-Augmented Generation) with citation tracking and ACL filtering, sleep-cycle consolidation, cross-source contradiction detection via LLM-as-Judge, and Obsidian-compatible diary and dream exports behind a small, composable API.

## Install

```bash
pip install hakua-memory
```

Python 3.11–3.13 are supported. Optional integrations are listed in the
Installation section below.

## Overview

Long-running agents need more than a vector index. They need memories that can decay, be reinforced, be inspected as structured claims, be cross-validated across multiple memory systems, and be exported into a human-readable knowledge base.

`hakua-memory` provides those building blocks without requiring the Hermes Agent codebase. The package uses a `src/` layout and keeps the main integrations optional where practical.

## Features

- **Ebbinghaus memory store**
  - SQLite persistence
  - Retention and forgetting-curve calculations
  - Recall scoring and reinforcement
  - Salience, valence, tags, and memory-state handling
  - Sleep-cycle rehearsal, archival, pruning, and dream candidates
  - Japanese vocabulary support for identity-critical concepts
- **Semantic Graph (Cognitive Graph / CoG)**
  - SQLite graph storage
  - Typed nodes, edges (10+ relation types: causes, contradicts, supports, requires, enables, prevents, relates_to, part_of, precedes, follows), evidence, fragments, and evaluations
  - Full-text search where available
  - Sanitisation and deterministic data handling
  - Cross-domain relationship modeling
- **Embedding retrieval**
  - Pluggable embedding backend contract
  - Deterministic fake backend for tests and demonstrations
  - Optional `llama-cpp-python` backend
  - Versioned embedding namespaces and dimension validation
  - Hybrid lexical/dense retrieval with deterministic reciprocal-rank fusion
- **RAG (Retrieval-Augmented Generation)**
  - Document ingestion (PDF, DOCX, PPTX, Markdown, plain text)
  - Page/slide/section-aware chunking with provenance
  - Document version management
  - ACL (Access Control List) filtering per document
  - Citation context rendering (Markdown + XML)
  - Meeting item extraction (decisions, tasks, action items, unresolved items)
  - Contradiction detection between documents
  - CJK-aware search with FTS + LIKE fallback
- **Cross-source Contradiction Detection (LLM-as-Judge)**
  - Triangulation across Ebbinghaus, RAG, and Semantic Graph
  - Structured prompts for LLM-based judgment
  - 5 contradiction types: factual, numerical, causal, temporal, confidence
  - Hermes Agent LLM integration (gpt-5.6-luna, openai-codex, etc.)
  - JSON-structured output for automated processing
- **Obsidian export**
  - Diary and dream markdown exports
  - Human-readable files suitable for an Obsidian vault
- **Single composite API**
  - One entry point for remembering, recalling, graph updates, RAG operations, retrieval, sleep, export, and status reporting

## Installation

Python 3.11, 3.12, and 3.13 are supported.

```bash
pip install hakua-memory
```

For local development:

```bash
pip install -e ".[dev]"
```

Optional extras:

```bash
pip install "hakua-memory[rag]"        # PDF/DOCX/PPTX ingestion
pip install "hakua-memory[embedding]"  # llama.cpp embeddings
pip install "hakua-memory[obsidian]"   # Obsidian helpers
pip install "hakua-memory[all]"         # all optional integrations
```

The `all` extra is the union of the `obsidian`, `embedding`, and `rag` integration
dependencies. `pyyaml` is installed by the base package.

## Quick start

```python
from pathlib import Path
from hakua_memory import CompositeMemory

memory = CompositeMemory(Path("~/.hakua-memory").expanduser())

memory.remember(
    "The user prefers concise, verifiable reports.",
    tags=["user-preference", "communication"],
    salience=0.9,
    valence=0.2,
)

matches = memory.recall("verifiable reports")
print(matches)

memory.add_node(
    {
        "node_id": "claim-1",
        "node_type": "Claim",
        "label": "Verifiable reporting",
        "summary": "Reports should distinguish observed results from assumptions.",
        "status": "asserted",
        "authority": "user",
        "confidence": 0.95,
        "salience": 0.8,
        "evidence": [],
    }
)

results = memory.search("Verifiable reporting")
print(results)

print(memory.stats())
memory.sleep()
memory.export_wiki(Path("~/.hakua-memory/wiki").expanduser())
memory.close()
```

## RAG Quick start

```python
from pathlib import Path
from hakua_memory import CompositeMemory

memory = CompositeMemory(Path("~/.hakua-memory").expanduser())
result = memory.ingest_text(
    "Decision: the release schedule is Friday. Task: Alice will confirm the schedule.",
    title="Q3 Planning Meeting",
    author="Alice",
    department="Product",
)
print(result)

results = memory.search_documents("release schedule", top_k=5)
context = memory.render_citations(results, format="markdown")
print(context)

items = memory.extract_meeting_items(result["document_id"])
for item in items:
    print(f"[{item['item_type']}] {item['content']}")

memory.grant_access(result["document_id"], "alice", "read")
alice_results = memory.search_documents("schedule", principal="alice")
print(alice_results)
print(memory.detect_contradictions(min_confidence=0.6))
memory.close()
```

## Cross-source Contradiction Detection (LLM-as-Judge)

```python
from pathlib import Path
from hakua_memory import CompositeMemory
from scripts.contradiction_detector import ContradictionDetector

memory = CompositeMemory(Path("~/.hakua-memory").expanduser())
detector = ContradictionDetector(memory)

# Detect contradictions across all three memory systems
queries = [
    "キャッシュフロー分析",
    "マイクロサービスアーキテクチャ",
    "医療DX推進",
]

for query in queries:
    # Get the judge prompt for LLM evaluation
    prompt = detector.get_judge_prompt(query)
    
    # Send to Hermes Agent / LLM for judgment
    # llm_response = hermes_agent.chat(prompt)
    # result = json.loads(llm_response)
    
    print(f"Query: {query}")
    print(f"Prompt ready for LLM evaluation")
```

**Detected Contradictions (v0.3.0):**

| Query | Contradiction | Type | Confidence | Sources |
|-------|--------------|------|------------|---------|
| キャッシュフロー分析 | ✅ True | factual | 0.85 | Ebbinghaus vs RAG vs CoG |
| マイクロサービスアーキテクチャ | ✅ True | factual | 0.90 | Ebbinghaus/CoG vs RAG |
| 医療DX推進 | False | none | 0.80 | - |

**Contradiction Types:**
- `factual` - 事実の矛盾 (同じ事象について異なる事実)
- `numerical` - 数値の矛盾 (同じ指標について異なる数値)
- `causal` - 因果関係の矛盾 (異なる原因・結果)
- `temporal` - 時系列の矛盾 (異なる時期・順序)
- `confidence` - 確信度の乖離 (高確信度同士で内容が食い違う)

## Embedding retrieval

Embedding support is backend-agnostic. The package includes a deterministic fake backend for tests and a contract for production backends.

```python
from hakua_memory.semantic_graph.embedding.base import EmbeddingModelIdentity
from hakua_memory.semantic_graph.embedding.fake import DeterministicFakeEmbeddingBackend

backend = DeterministicFakeEmbeddingBackend(
    identity=EmbeddingModelIdentity(
        provider="demo",
        model="deterministic",
        revision="1",
        dimensions=8,
        serializer_version=1,
    ),
    vectors={"Verifiable reporting": [1, 2, 3, 4, 5, 6, 7, 8]},
)

results = memory.search("Verifiable reporting", backend=backend)
```

The retrieval layer validates vector dimensions, records the embedding namespace, and combines lexical and dense rankings without silently mixing incompatible representations.

## Performance comparison

### Internal reproducible measurements

Run the RAG chunking, semantic-graph lexical retrieval, and Ebbinghaus recall
operations with one deterministic dataset and query configuration:

```bash
python scripts/generate_synthetic_data.py \
  --seed 42 \
  --samples 40 \
  --output synthetic_business_dataset.json
python scripts/benchmark_cross_validation.py \
  --seed 42 \
  --samples 40 \
  --warmup 5 \
  --repetitions 30 \
  --output benchmark-results.json
```

The benchmark output is machine-readable JSON containing package version, commit SHA,
timestamp, operating system, Python version, CPU, dataset id, seed, sample count, warmup
count, repetition count, and mean, median, standard deviation, p50, p95, and p99 latency
for every measured operation. No GPU is used by this harness. Generated JSON, SQLite
state, and temporary benchmark databases are not version-controlled;
`benchmarks/synthetic_dataset_config.json` records the dataset recipe and expected
summary.

Only measurements produced by this repository's harness should be described as internal
results. Values depend on the recorded machine, dataset, seed, warmup, and repetitions.

### Cross-library validation on the same hardware

The optional comparison harness uses the same corpus, query set, warmup count, repetition
count, and randomized variant order for every paired measurement. It compares RAG
chunking with LangChain's `RecursiveCharacterTextSplitter` and LlamaIndex's
`SentenceSplitter`, and compares CoG/retrieval peers across hakua lexical search,
hakua hybrid GGUF search, `rank-bm25`, LangChain BM25, LlamaIndex BM25, and
FAISS+same-GGUF dense retrieval. Multi-group Friedman tests and all-pairs
Wilcoxon/Holm corrections are recorded. This is not a claim about complete RAG quality,
end-to-end agent latency, memory use, or general library superiority.

Create a separate environment and run the comparison as follows:

```bash
python -m venv .venv-benchmark
# Activate .venv-benchmark for your shell, then run:
python -m pip install -e ".[embedding]"
python -m pip install -r benchmarks/requirements.txt
python scripts/generate_synthetic_data.py --seed 42 --samples 40 --output synthetic_business_dataset.json
python scripts/benchmark_external_libraries.py --seed 42 --samples 40 --warmup 2 --repetitions 10 --output benchmark-results-external.json --plot benchmark-errorbars.png
python scripts/benchmark_speculative_hybrid.py --samples 30 --warmup 1 --repetitions 8 --output speculative-hybrid-3group.json --plot speculative-hybrid-3group.png
```

The comparison JSON records exact library versions, host metadata, configurations, raw
paired durations, mean, median, standard deviation, p50, p95, p99, and a 95% confidence
interval for each variant. It also records Friedman chi-square multi-group tests,
two-sided paired Wilcoxon signed-rank tests, paired t-test sensitivity, and Holm-adjusted
Wilcoxon p-values across all pairwise comparisons. The PNG uses mean latency with 95%
confidence-interval error bars. The validation section separately records non-empty chunk
output and expected top-k hit rate, so latency and retrieval-result checks are not
conflated.

The chunkers use their native length units, which are recorded in the JSON; equal numeric
chunk-size arguments therefore do not mean equal chunk boundaries. Lexical peers compare
SQLite-backed graph search with in-memory BM25 indexes. Dense peers use the same local
GGUF embedding model (`nsfw-bge-m3-v5-q6_k`, 1024-d) via `llama-cpp-python`. These scope
limits are required for an honest cross-library measurement.

The following is one recorded 0.3.5 run from commit `62c8c6054bae` on Windows 11 with
Python 3.12.10, an AMD64 Family 23 Model 96 CPU, and CUDA via `llama-cpp` (`n_gpu_layers=-1`).
It used `synthetic-business-v1`, seed 42, 40 samples / 40 queries, warmup 2, and 10 paired
repetitions. Latencies are milliseconds; CoG rows are milliseconds per complete 40-query
batch. The confidence interval is the 95% CI for the mean.

| Operation | Variant | Mean | Median | Stdev | P95 | 95% CI |
|---|---|---:|---:|---:|---:|---:|
| RAG chunking | hakua-memory | 1.0630 | 1.0358 | 0.1563 | 1.3290 | 0.9513–1.1748 |
| RAG chunking | LangChain Text Splitters | 0.1130 | 0.1185 | 0.0168 | 0.1329 | 0.1009–0.1250 |
| RAG chunking | LlamaIndex Core | 2.2198 | 2.1212 | 0.3185 | 2.6825 | 1.9919–2.4477 |
| CoG retrieval batch | hakua lexical | 45.5863 | 47.3555 | 7.6094 | 56.0006 | 40.1429–51.0297 |
| CoG retrieval batch | rank-bm25 | 4.2935 | 3.6322 | 2.2569 | 7.9821 | 2.6790–5.9080 |
| CoG retrieval batch | LangChain BM25 | 6.5765 | 5.6572 | 2.8180 | 11.2159 | 4.5606–8.5924 |
| CoG retrieval batch | LlamaIndex BM25 | 40.3695 | 39.8424 | 10.2176 | 52.1031 | 33.0602–47.6787 |
| CoG retrieval batch | hakua hybrid GGUF | 31519.2576 | 25784.8628 | 13763.0641 | 57242.3898 | 21673.7546–41364.7606 |
| CoG retrieval batch | FAISS + same GGUF | 16617.3399 | 14470.4271 | 9264.0809 | 31038.2213 | 9990.2156–23244.4642 |

#### Multi-group Friedman p-values

| Operation | Groups (n) | Friedman χ² | p-value | Reject H₀ (α=0.05) |
|---|---|---:|---:|---|
| RAG chunking | 3 | 20.0 | 4.540e-05 | yes |
| CoG retrieval batch | 6 | 47.6 | 4.286e-09 | yes |

#### Pairwise Wilcoxon / Holm / paired-t p-values (RAG chunking)

| Primary | Comparator | Wilcoxon p | Holm-adjusted Wilcoxon p | Paired-t p |
|---|---|---:|---:|---:|
| hakua-memory | LangChain Text Splitters | 0.001953 | 0.035156 | 1.366e-08 |
| hakua-memory | LlamaIndex Core | 0.001953 | 0.035156 | 3.434e-06 |
| LangChain Text Splitters | LlamaIndex Core | 0.001953 | 0.035156 | 7.409e-09 |

#### Pairwise Wilcoxon / Holm / paired-t p-values (CoG retrieval batch)

| Primary | Comparator | Wilcoxon p | Holm-adjusted Wilcoxon p | Paired-t p |
|---|---|---:|---:|---:|
| hakua lexical | rank-bm25 | 0.001953 | 0.035156 | 2.450e-08 |
| hakua lexical | LangChain BM25 | 0.001953 | 0.035156 | 4.766e-08 |
| hakua lexical | LlamaIndex BM25 | 0.193359 | 0.193359 | 0.137301 |
| hakua lexical | hakua hybrid GGUF | 0.001953 | 0.035156 | 4.913e-05 |
| hakua lexical | FAISS + same GGUF | 0.001953 | 0.035156 | 0.000310 |
| rank-bm25 | LangChain BM25 | 0.005859 | 0.035156 | 0.001648 |
| rank-bm25 | LlamaIndex BM25 | 0.001953 | 0.035156 | 5.045e-07 |
| rank-bm25 | hakua hybrid GGUF | 0.001953 | 0.035156 | 4.862e-05 |
| rank-bm25 | FAISS + same GGUF | 0.001953 | 0.035156 | 0.000305 |
| LangChain BM25 | LlamaIndex BM25 | 0.001953 | 0.035156 | 4.817e-07 |
| LangChain BM25 | hakua hybrid GGUF | 0.001953 | 0.035156 | 4.861e-05 |
| LangChain BM25 | FAISS + same GGUF | 0.001953 | 0.035156 | 0.000306 |
| LlamaIndex BM25 | hakua hybrid GGUF | 0.001953 | 0.035156 | 4.890e-05 |
| LlamaIndex BM25 | FAISS + same GGUF | 0.001953 | 0.035156 | 0.000310 |
| hakua hybrid GGUF | FAISS + same GGUF | 0.037109 | 0.074219 | 0.018810 |

These p-values describe this paired latency sample only; they are not accuracy claims or
proof of general library superiority. Holm correction is applied across all pairwise
Wilcoxon tests emitted by the harness for that operation family.

Dataset-specific validation found non-empty chunk output for all chunkers. Exact-record
hit rate at top-k=5 was 33/40 (82.5%) for hakua lexical and hakua hybrid GGUF, 38/40
(95.0%) for rank-bm25 and LangChain BM25, 39/40 (97.5%) for LlamaIndex BM25, and 18/40
(45.0%) for FAISS+GGUF on this synthetic set. This is a small query-set check, not a
cross-domain retrieval-quality benchmark. Generated JSON/PNG stay outside Git; rerun the
commands above to reproduce on another machine.

### Speculative-hybrid three-group Friedman

`scripts/benchmark_speculative_hybrid.py` compares `lexical_only`, `always_hybrid`, and
`speculative_hybrid` on the same corpus after sweeping `min_top_score` (primary early-accept
gate; absolute `final_score` margins often collapse on uniform confidence/salience data).
One recorded run (seed 42, 30 samples, warmup 1, 8 repetitions) chose Pareto
`min_top_score=0.40` (hit-rate tied with hybrid at 0.800; minimum latency). An interior
early-accept candidate near 50% was `min_top_score=0.55` (early-accept rate ≈ 0.367).
Mean latencies at the Pareto gate: lexical 56.8 ms, speculative 58.3 ms, always-hybrid
22800.2 ms.

| Test | Statistic | p-value | Reject H₀ (α=0.05) |
|---|---:|---:|---|
| Friedman χ² (3 groups) | 12.0 | 0.002479 | yes |

| Primary | Comparator | Wilcoxon p | Holm-adjusted Wilcoxon p | Paired-t p |
|---|---|---:|---:|---:|
| lexical_only | always_hybrid | 0.007812 | 0.023438 | 1.371e-06 |
| lexical_only | speculative_hybrid | 0.843750 | 0.843750 | 0.660915 |
| always_hybrid | speculative_hybrid | 0.007812 | 0.023438 | 1.369e-06 |

### External references

The measured comparison uses [LangChain Text Splitters](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter),
[LlamaIndex SentenceSplitter](https://docs.llamaindex.ai/en/stable/module_guides/loading/node_parsers/modules.html),
[rank-bm25](https://pypi.org/project/rank-bm25/), LangChain/LlamaIndex BM25 retrievers,
and [FAISS](https://github.com/facebookresearch/faiss) with versions pinned in
`benchmarks/requirements.txt`. NetworkX and PyTorch Geometric remain external references
only; this repository does not publish direct measurements for them.

## Reproducible synthetic data

The generator is deterministic for a given seed and sample count. Its schema version,
generator version, defaults, and expected summary are recorded in
`benchmarks/synthetic_dataset_config.json`. Use a different output path when retaining
multiple datasets; generated files remain ignored by Git.

## Architecture

```text
CompositeMemory
├── EbbinghausMemoryStore ── SQLite memory database (forgetting curves, salience)
├── SemanticGraphStore ──── SQLite graph database (nodes, 10 edge types, evidence)
├── Hybrid retrieval ────── lexical + optional embedding backend
├── Contradiction Detector ─ LLM-as-Judge across 3 memory systems
└── Obsidian export ──────── diary and dream Markdown files
```

### Semantic Graph Edge Types

| Type | Japanese | Description |
|------|----------|-------------|
| `causes` | 原因 | A causes B |
| `contradicts` | 矛盾 | A contradicts B |
| `supports` | 支持 | A supports B |
| `requires` | 必要 | A requires B |
| `enables` | 可能にする | A enables B |
| `prevents` | 防止 | A prevents B |
| `relates_to` | 関連 | A relates to B |
| `part_of` | 一部 | A is part of B |
| `precedes` | 先行 | A precedes B |
| `follows` | 後続 | A follows B |

## Development

Run the package checks from the repository root:

```bash
python -m compileall -q src
python -m build
```

The integration path can be exercised with a temporary directory and the deterministic embedding backend, covering:

1. memory insertion and recall;
2. semantic-node insertion;
3. hybrid search;
4. sleep-cycle processing; and
5. Obsidian diary export.

## Design notes

- SQLite is the persistence boundary; generated databases should normally remain outside the repository.
- Embedding backends are optional and replaceable.
- Lexical retrieval remains available when no embedding backend is configured.
- Numerical verification and formal proof are deliberately separate concerns.
- Exported Markdown is intended to be inspectable and version-controllable by the user.
- Japanese vocabulary is used for identity-critical concepts (Hermes Agent = はくあ, etc.)
- Cross-source contradiction detection requires LLM Judge (Hermes Agent integration provided)

## License

Apache License 2.0. See [LICENSE](LICENSE).
