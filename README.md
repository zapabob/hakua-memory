# hakua-memory

> **TL;DR:** `hakua-memory` is a self-contained Python memory library for autonomous agents. It combines SQLite-backed Ebbinghaus memory, a structured semantic graph, optional embedding-based retrieval, sleep-cycle consolidation, and Obsidian-compatible diary and dream exports behind a small, composable API.

## Overview

Long-running agents need more than a vector index. They need memories that can decay, be reinforced, be inspected as structured claims, and be exported into a human-readable knowledge base.

`hakua-memory` provides those building blocks without requiring the Hermes Agent codebase. The package uses a `src/` layout and keeps the main integrations optional where practical.

## Features

- **Ebbinghaus memory store**
  - SQLite persistence
  - Retention and forgetting-curve calculations
  - Recall scoring and reinforcement
  - Salience, valence, tags, and memory-state handling
  - Sleep-cycle rehearsal, archival, pruning, and dream candidates
- **Semantic Graph**
  - SQLite graph storage
  - Typed nodes, edges, evidence, fragments, and evaluations
  - Full-text search where available
  - Sanitisation and deterministic data handling
- **Embedding retrieval**
  - Pluggable embedding backend contract
  - Deterministic fake backend for tests and demonstrations
  - Optional `llama-cpp-python` backend
  - Versioned embedding namespaces and dimension validation
  - Hybrid lexical/dense retrieval with deterministic reciprocal-rank fusion
- **Obsidian export**
  - Diary and dream markdown exports
  - Human-readable files suitable for an Obsidian vault
- **Single composite API**
  - One entry point for remembering, recalling, graph updates, retrieval, sleep, export, and status reporting

## Installation

Python 3.11, 3.12, and 3.13 are supported.

```bash
python -m pip install hakua-memory
```

For local development:

```bash
python -m pip install -e ".[dev]"
```

Optional extras:

```bash
# Obsidian/Git integration helpers
python -m pip install "hakua-memory[obsidian]"

# llama.cpp embedding support
python -m pip install "hakua-memory[embedding]"
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

print(memory.stats())
memory.sleep()
memory.export_wiki(Path("knowledge-base"))
```

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

## Architecture

```text
CompositeMemory
├── EbbinghausMemoryStore ── SQLite memory database
├── SemanticGraphStore ──── SQLite graph database
├── Hybrid retrieval ────── lexical + optional embedding backend
└── Obsidian export ──────── diary and dream Markdown files
```

The package is intentionally independent from Hermes Agent. It can be embedded into another agent runtime, used as a local library, or extended through its backend interfaces.

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

## License

Apache License 2.0. See [LICENSE](LICENSE).
