"""High-level composite memory API tying Ebbinghaus, Semantic Graph, Embedding, and RAG."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Optional, TYPE_CHECKING

from hakua_memory.ebbinghaus.store import EbbinghausMemoryStore
from hakua_memory.obsidian import write_diary
from hakua_memory.semantic_graph.embedding.base import EmbeddingBackend
from hakua_memory.semantic_graph.retrieval import (
    hybrid_search_and_rank,
    speculative_hybrid_search_and_rank,
)
from hakua_memory.semantic_graph.store import SemanticGraphStore

if TYPE_CHECKING:
    from hakua_memory.rag.models import AclEntry
    from hakua_memory.rag.retrieval import RagResult
    from hakua_memory.rag.store import DocumentStore


class CompositeMemory:
    """Single entrypoint for local composite memory.

    Embedding backends remain first-class for hybrid search quality. Pass an
    ``EmbeddingBackend`` to :meth:`search` (llama.cpp HTTP or in-process via
    the ``[embedding]`` extra). RAG modules load lazily so remember/recall-only
    paths stay light.
    """

    def __init__(self, root: Path, *, enable_rag: bool = True) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.ebbinghaus = EbbinghausMemoryStore(self.root / "ebbinghaus.db")
        self.semantic = SemanticGraphStore(self.root / "semantic.db")
        self._enable_rag = enable_rag
        self._documents: DocumentStore | None = None

    @property
    def documents(self) -> DocumentStore:
        """Lazy DocumentStore so import/RSS stay light until RAG is used."""
        if self._documents is None:
            if not self._enable_rag:
                raise RuntimeError("RAG store is disabled for this CompositeMemory")
            from hakua_memory.rag.store import DocumentStore

            self._documents = DocumentStore(self.root / "rag.db")
        return self._documents

    def remember(
        self,
        content: str,
        tags: Optional[list[str]] = None,
        *,
        salience: float = 0.65,
        valence: float = 0.0,
        source: str = "",
        session_id: str = "",
        memory_type: str = "episodic",
    ) -> dict[str, Any]:
        """Store an episodic/semantic cue with optional salience and valence.

        ``salience`` and ``valence`` are passed through to
        :meth:`EbbinghausMemoryStore.remember` so sleep/dream consolidation can
        keep high-value traces without dropping to the store API.
        """
        return self.ebbinghaus.remember(
            content=content,
            tags=tags or [],
            salience=salience,
            valence=valence,
            source=source,
            session_id=session_id,
            memory_type=memory_type,
        )

    def recall(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self.ebbinghaus.recall(query, limit=top_k)

    def sleep(
        self,
        **sleep_kwargs: Any,
    ) -> dict[str, Any]:
        """Run one sleep-cycle consolidation pass (optional kwargs forwarded)."""
        return self.ebbinghaus.sleep_cycle(**sleep_kwargs)

    def add_node(self, node: dict[str, Any]) -> dict[str, Any]:
        self.semantic.ensure_ready()
        return self.semantic.upsert_node(node)

    def search(
        self,
        query: str,
        *,
        top_k: int = 8,
        backend: Optional[EmbeddingBackend] = None,
        speculative: bool = False,
        margin_threshold: float = 0.0,
        min_top_score: float = 0.55,
    ) -> list[dict[str, Any]]:
        """Hybrid lexical + dense search. Pass ``backend`` for embedding quality.

        When ``speculative=True`` and ``backend`` is set, lexical drafts early-accept
        on high margin; otherwise dense hybrid verifies (speculative-hybrid path).
        """
        if speculative and backend is not None:
            return speculative_hybrid_search_and_rank(
                self.semantic,
                query,
                backend=backend,
                top_k=top_k,
                margin_threshold=margin_threshold,
                min_top_score=min_top_score,
            )
        return hybrid_search_and_rank(
            self.semantic,
            query,
            backend=backend,
            top_k=top_k,
        )

    # ── RAG methods ─────────────────────────────────────────────────

    def ingest_document(self, path: Path, **kwargs: Any) -> dict[str, Any]:
        """Ingest a document file and store it with chunks."""
        from hakua_memory.rag.ingestion import ingest_document

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
        from hakua_memory.rag.ingestion import ingest_markdown_string

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
        from hakua_memory.rag.ingestion import ingest_text_string

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
        from hakua_memory.rag.retrieval import search_chunks

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
        from hakua_memory.rag.retrieval import RagResult, render_citation_context

        rag_results: list[RagResult] = []
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
        from hakua_memory.rag.meeting import extract_meeting_items

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
        from hakua_memory.rag.contradiction import detect_contradictions

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
        from hakua_memory.rag.models import AclEntry

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
        return self.documents.list_chunks_for_document(document_id)

    def close(self) -> None:
        """Close all underlying stores."""
        self.ebbinghaus.close()
        self.semantic.close()
        if self._documents is not None:
            self._documents.close()
            self._documents = None

    def export_wiki(self, wiki_root: Path) -> dict[str, Any]:
        path = write_diary(wiki_root, "composite-export", "# CompositeMemory export\n")
        return {"diary": str(path)}

    def stats(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "ebbinghaus": self.ebbinghaus.stats(),
            "semantic_graph": self.semantic.get_status_counts(),
        }
        if self._documents is not None:
            result["rag"] = self._documents.stats()
        elif self._enable_rag:
            # Keep API shape without forcing RAG open on remember-only paths.
            result["rag"] = {"initialized": False}
        return result
