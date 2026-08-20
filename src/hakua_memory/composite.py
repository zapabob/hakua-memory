"""High-level composite memory API tying Ebbinghaus, Semantic Graph, Embedding, and RAG."""

from __future__ import annotations

import sqlite3
from pathlib import Path
from typing import Any, Optional

from hakua_memory.ebbinghaus.store import EbbinghausMemoryStore
from hakua_memory.obsidian import write_diary
from hakua_memory.rag import (
    AclEntry,
    DocumentStore,
    RagResult,
    detect_contradictions,
    extract_meeting_items,
    ingest_document,
    ingest_markdown_string,
    ingest_text_string,
    render_citation_context,
    search_chunks,
)
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
        self.documents = DocumentStore(self.root / "rag.db")

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

    # ── RAG methods ─────────────────────────────────────────────────

    def ingest_document(self, path: Path, **kwargs: Any) -> dict[str, Any]:
        """Ingest a document file and store it with chunks."""
        doc, chunks = ingest_document(path, **kwargs)
        self.documents.insert_document(doc)
        chunk_ids = self.documents.insert_chunks(chunks)
        return {
            "document_id": doc.document_id,
            "title": doc.title,
            "chunks": len(chunk_ids),
            "document_type": doc.document_type,
        }

    def ingest_markdown(self, text: str, **kwargs: Any) -> dict[str, Any]:
        """Ingest a markdown string."""
        doc, chunks = ingest_markdown_string(text, **kwargs)
        self.documents.insert_document(doc)
        chunk_ids = self.documents.insert_chunks(chunks)
        return {
            "document_id": doc.document_id,
            "title": doc.title,
            "chunks": len(chunk_ids),
        }

    def ingest_text(self, text: str, **kwargs: Any) -> dict[str, Any]:
        """Ingest a plain text string."""
        doc, chunks = ingest_text_string(text, **kwargs)
        self.documents.insert_document(doc)
        chunk_ids = self.documents.insert_chunks(chunks)
        return {
            "document_id": doc.document_id,
            "title": doc.title,
            "chunks": len(chunk_ids),
        }

    def search_documents(
        self,
        query: str,
        *,
        top_k: int = 8,
        principal: str = "",
        document_type: str = "",
        department: str = "",
    ) -> list[dict[str, Any]]:
        """Search ingested documents with full-text search."""
        results = search_chunks(
            self.documents,
            query,
            top_k=top_k,
            principal=principal,
            document_type=document_type,
            department=department,
        )
        return [r.to_dict() for r in results]

    def render_citations(
        self,
        results: list[dict[str, Any]],
        *,
        max_chars: int = 4000,
        format: str = "markdown",
    ) -> str:
        """Render citation context from search results."""
        # Rebuild RagResult objects from dicts
        rag_results = []
        for r in results:
            chunk = self.documents.get_chunk(r["chunk_id"])
            doc = self.documents.get_document(r["document_id"])
            if chunk and doc:
                rag_results.append(
                    RagResult(
                        chunk=chunk,
                        document=doc,
                        score=r.get("score", 0.0),
                        rank=r.get("rank", 0),
                    )
                )
        return render_citation_context(rag_results, max_chars=max_chars, format=format)

    def extract_meeting_items(
        self, document_id: str, *, auto_store: bool = True
    ) -> list[dict[str, Any]]:
        """Extract meeting items (decisions, tasks, action items) from a document."""
        chunks = self._get_chunks(document_id)
        items = extract_meeting_items(
            document_id, chunks, store=self.documents, auto_store=auto_store
        )
        return [
            {
                "item_id": i.item_id,
                "item_type": i.item_type,
                "content": i.content,
                "assignee": i.assignee,
                "due_date": i.due_date,
                "page_number": i.page_number,
                "slide_number": i.slide_number,
            }
            for i in items
        ]

    def detect_contradictions(
        self,
        document_ids: Optional[list[str]] = None,
        min_confidence: float = 0.6,
    ) -> list[dict[str, Any]]:
        """Detect contradictions between documents."""
        contradictions = detect_contradictions(
            self.documents,
            document_ids=document_ids,
            min_confidence=min_confidence,
        )
        return [
            {
                "contradiction_id": c.contradiction_id,
                "type": c.contradiction_type,
                "description": c.description,
                "document_a": c.document_a_title,
                "document_b": c.document_b_title,
                "confidence": c.confidence,
                "page_a": c.page_a,
                "page_b": c.page_b,
            }
            for c in contradictions
        ]

    def grant_access(
        self, document_id: str, principal: str, permission: str, *, department: str = ""
    ) -> dict[str, Any]:
        """Grant ACL permission on a document.
        Args:
            document_id: The document ID.
            principal: The user, group, or role.
            permission: "read", "write", or "delete".
            department: Optional department scope.
        """
        entry = AclEntry(
            document_id=document_id,
            principal=principal,
            permission=permission,
            department=department,
        )
        key = self.documents.grant_acl(entry)
        return {"acl_key": key, "permission": entry.permission}

    def revoke_access(self, document_id: str, principal: str, permission: str) -> dict[str, Any]:
        """Revoke ACL permission on a document."""
        success = self.documents.revoke_acl(document_id, principal, permission)
        return {"revoked": success}

    def check_access(self, document_id: str, principal: str) -> dict[str, Any]:
        """Check ACL permissions for a principal on a document.
        Returns a dict with boolean flags and the permissions list.
        """
        result = self.documents.check_acl_detailed(document_id, principal)
        return {
            "document_id": result.document_id,
            "principal": result.principal,
            "can_read": result.can_read,
            "can_write": result.can_write,
            "can_delete": result.can_delete,
            "permissions": result.permissions,
        }

    def check_access_department(self, document_id: str, department: str) -> list[str]:
        """Check ACL permissions for a department on a document."""
        return self.documents.check_acl_department(document_id, department)

    def _get_chunks(self, document_id: str) -> list:
        """Get all chunks for a document."""
        with self.documents._conn() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
                (document_id,),
            ).fetchall()
        from hakua_memory.rag.store import _row_to_chunk
        return [_row_to_chunk(row) for row in rows]

    def close(self) -> None:
        """Close all underlying stores."""
        self.ebbinghaus.close()

    def export_wiki(self, wiki_root: Path) -> dict[str, Any]:
        path = write_diary(wiki_root, "composite-export", "# CompositeMemory export\n")
        return {"diary": str(path)}

    def stats(self) -> dict[str, Any]:
        return {
            "ebbinghaus": self.ebbinghaus.stats(),
            "semantic_graph": self.semantic.get_status_counts(),
            "rag": self.documents.stats(),
        }
