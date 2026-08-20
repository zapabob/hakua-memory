"""High-level composite memory API tying Ebbinghaus, Semantic Graph, and Embedding."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional

from hakua_memory.ebbinghaus.store import EbbinghausMemoryStore
from hakua_memory.obsidian import write_diary, write_dream
from hakua_memory.semantic_graph.embedding.base import EmbeddingBackend
from hakua_memory.semantic_graph.retrieval import hybrid_search_and_rank
from hakua_memory.semantic_graph.store import SemanticGraphStore


class CompositeMemory:
    """Single entrypoint for local composite memory."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ebbinghaus = EbbinghausMemoryStore(self.root / "ebbinghaus.db")
        self.semantic = SemanticGraphStore(self.root / "semantic.db")

    def remember(self, content: str, tags: Optional[list[str]] = None) -> dict[str, Any]:
        return self.ebbinghaus.remember(content=content, tags=tags or [])

    def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.ebbinghaus.recall(query, limit=top_k)

    def sleep(self) -> dict[str, Any]:
        return self.ebbinghaus.sleep_cycle()

    def add_node(self, node: dict[str, Any]) -> dict[str, Any]:
        self.semantic.ensure_ready()
        return self.semantic.upsert_node(node)

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        backend: Optional[EmbeddingBackend] = None,
    ) -> list[dict[str, Any]]:
        return hybrid_search_and_rank(
            self.semantic,
            query,
            backend=backend,
            top_k=top_k,
        )

    def export_wiki(self, wiki_root: Path) -> dict[str, Any]:
        path = write_diary(wiki_root, "composite-export", "# CompositeMemory export\n")
        return {"diary": str(path)}

    def stats(self) -> dict[str, Any]:
        return {
            "ebbinghaus": self.ebbinghaus.stats(),
            "semantic_graph": self.semantic.get_status_counts(),
        }
