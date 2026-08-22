# hakua-memory

> **TL;DR:** `hakua-memory` is a self-contained Python memory library for autonomous agents. It combines SQLite-backed Ebbinghaus memory, a structured semantic graph (CoG), optional embedding-based retrieval, RAG (Retrieval-Augmented Generation) with citation tracking and ACL filtering, sleep-cycle consolidation, cross-source contradiction detection via LLM-as-Judge, and Obsidian-compatible diary and dream exports behind a small, composable API.

## Install

```bash
pip install hakua-memory
```

That's it. Python 3.11–3.13 supported.

Optional extras:

```bash
pip install "hakua-memory[rag]"        # PDF/DOCX/PPTX ingestion
pip install "hakua-memory[embedding]"  # llama.cpp embeddings
pip install "hakua-memory[obsidian]"   # Obsidian helpers
pip install "hakua-memory[all]"         # all optional integrations
```

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

## Quick start

```python
from pathlib import Path
from hakua_memory import CompositeMemory

memory = CompositeMemory(Path(".memory"))

memory.remember(
    "The user prefers concise, verifiable reports.",
    tags=["user-preference", "communication"],
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

# Add edges between nodes
memory.semantic.upsert_edge({
    "edge_id": "edge-1",
    "source_node_id": "claim-1",
    "target_node_id": "claim-2",
    "edge_type": "supports",
    "label": "支持",
    "confidence": 0.8,
    "evidence": [],
})

print(memory.stats())
memory.sleep()
memory.export_wiki(Path("knowledge-base"))
```

## RAG Quick start

```python
from pathlib import Path
from hakua_memory import CompositeMemory

memory = CompositeMemory(Path(".memory"))

# Ingest a PDF document
result = memory.ingest_document(
    Path("meeting_notes.pdf"),
    title="Q3 Planning Meeting",
    author="Alice",
    department="Product",
)
print(result)  # {'document_id': '...', 'title': '...', 'chunks': 12, ...}

# Search with citation tracking
results = memory.search_documents("release schedule", top_k=5)
for r in results:
    print(f"[{r['rank']}] {r['document_title']} (p.{r.get('page_number')})")
    print(f"    {r['content'][:100]}...")

# Render citation context for RAG answers
context = memory.render_citations(results, format="markdown")
print(context)

# Extract meeting items (decisions, tasks, action items)
items = memory.extract_meeting_items(results[0]["document_id"])
for item in items:
    print(f"[{item['item_type']}] {item['content']}")

# Grant ACL access and search with filtering
memory.grant_access(results[0]["document_id"], "alice", "read")
alice_results = memory.search_documents("schedule", principal="alice")

# Detect contradictions between documents
contradictions = memory.detect_contradictions(min_confidence=0.6)
for c in contradictions:
    print(f"[{c['type']}] {c['description']}")
```

## Cross-source Contradiction Detection (LLM-as-Judge)

```python
from pathlib import Path
from hakua_memory import CompositeMemory
from scripts.contradiction_detector import ContradictionDetector

memory = CompositeMemory(Path(".memory"))
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

## Performance Comparison

### Internal reproducible benchmarks

Run the existing RAG chunking, semantic-graph lexical retrieval, and Ebbinghaus recall
operations with the same deterministic dataset and query configuration:

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

The benchmark output is machine-readable JSON containing the package version, commit
SHA, timestamp, operating system, Python version, CPU, dataset id, seed, sample count,
warmup count, repetition count, and mean, median, standard deviation, p50, p95, and p99
latency for each measured operation. The benchmark does not use a GPU, so no GPU value
is emitted. Generated JSON, SQLite state, and temporary benchmark databases are not
version-controlled; `benchmarks/synthetic_dataset_config.json` records the recipe and
expected dataset summary.

Only measurements produced by this repository's harness should be described as internal
results. This README intentionally does not publish fixed latency or accuracy numbers;
those values depend on the recorded machine, dataset, seed, warmup, and repetitions.

### External references

LangChain, LlamaIndex, PyTorch Geometric, NetworkX, and FAISS are relevant external
references. They are not run by this harness, so no direct latency, accuracy, or memory
comparison is claimed here.

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
