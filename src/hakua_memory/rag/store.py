"""Document store with version management, ACL filtering, and citation support."""

from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

from .models import AclCheckResult, AclEntry, Chunk, Citation, Document, MeetingItem
from .schema import ALL_DDL

# Import ACL constants for convenience
from .models import ACL_READ, ACL_WRITE, ACL_DELETE, ALL_ACL_PERMISSIONS


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class DocumentStore:
    """SQLite-backed document store with versioning and ACL."""

    def __init__(self, db_path: Path | str) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            for ddl in ALL_DDL:
                conn.executescript(ddl)
            conn.commit()

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
        finally:
            conn.close()

    # ── Document CRUD ─────────────────────────────────────────────

    def insert_document(self, doc: Document) -> str:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO documents (
                    document_id, title, source_uri, document_type, version,
                    author, department, created_at, ingested_at, content_hash,
                    page_count, slide_count, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc.document_id, doc.title, doc.source_uri, doc.document_type,
                    doc.version, doc.author, doc.department, doc.created_at,
                    doc.ingested_at, doc.content_hash, doc.page_count,
                    doc.slide_count, _json_dump(doc.metadata),
                ),
            )
            conn.execute(
                "INSERT INTO documents_fts (document_id, title) VALUES (?, ?)",
                (doc.document_id, doc.title),
            )
            conn.commit()
        return doc.document_id

    def get_document(self, document_id: str) -> Optional[Document]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM documents WHERE document_id = ?", (document_id,)
            ).fetchone()
        if not row:
            return None
        return _row_to_document(row)

    def update_document_version(
        self, document_id: str, new_version: str, *, new_content_hash: str = ""
    ) -> bool:
        """Insert a new version of an existing document."""
        doc = self.get_document(document_id)
        if not doc:
            return False
        with self._conn() as conn:
            conn.execute(
                """
                UPDATE documents SET version = ?, content_hash = COALESCE(?, content_hash),
                ingested_at = ? WHERE document_id = ?
                """,
                (new_version, new_content_hash, _now_iso(), document_id),
            )
            conn.commit()
        return True

    def delete_document(self, document_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM documents WHERE document_id = ?", (document_id,)
            )
            conn.commit()
            return cur.rowcount > 0

    def list_documents(
        self,
        *,
        document_type: str = "",
        department: str = "",
        principal: str = "",
        limit: int = 100,
    ) -> list[Document]:
        """List documents filtered by type, department, and ACL."""
        query = "SELECT d.* FROM documents d"
        params: list[Any] = []
        conditions = []
        if principal:
            conditions.append(
                "d.document_id IN (SELECT document_id FROM acl WHERE principal = ?)"
            )
            params.append(principal)
        if document_type:
            conditions.append("d.document_type = ?")
            params.append(document_type)
        if department:
            conditions.append("d.department = ?")
            params.append(department)
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY d.ingested_at DESC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_document(row) for row in rows]

    # ── Chunk CRUD ────────────────────────────────────────────────

    def insert_chunk(self, chunk: Chunk) -> str:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chunks (
                    chunk_id, document_id, chunk_index, content, content_hash,
                    page_number, slide_number, section, start_char, end_char,
                    token_count, metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    chunk.chunk_id, chunk.document_id, chunk.chunk_index,
                    chunk.content, chunk.content_hash, chunk.page_number,
                    chunk.slide_number, chunk.section, chunk.start_char,
                    chunk.end_char, chunk.token_count, _json_dump(chunk.metadata),
                    _now_iso(),
                ),
            )
            conn.execute(
                "INSERT INTO chunks_fts (chunk_id, content) VALUES (?, ?)",
                (chunk.chunk_id, chunk.content),
            )
            conn.commit()
        return chunk.chunk_id

    def insert_chunks(self, chunks: list[Chunk]) -> list[str]:
        return [self.insert_chunk(c) for c in chunks]

    def get_chunk(self, chunk_id: str) -> Optional[Chunk]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chunks WHERE chunk_id = ?", (chunk_id,)
            ).fetchone()
        if not row:
            return None
        return _row_to_chunk(row)

    def search_chunks_fts(
        self, query: str, *, top_k: int = 16, principal: str = ""
    ) -> list[Chunk]:
        """Full-text search on chunks, optionally filtered by ACL.
        Falls back to LIKE for CJK-heavy queries where FTS may not tokenize well."""
        # Build FTS query: split into terms and AND them for better CJK handling
        terms = query.split()
        if len(terms) > 1:
            fts_query = " AND ".join(terms)
        else:
            fts_query = query

        if principal:
            sql_fts = """
                SELECT c.* FROM chunks c
                JOIN chunks_fts f ON c.chunk_id = f.chunk_id
                JOIN acl a ON c.document_id = a.document_id
                WHERE chunks_fts MATCH ? AND a.principal = ?
                ORDER BY rank LIMIT ?
            """
            sql_like = """
                SELECT c.* FROM chunks c
                JOIN acl a ON c.document_id = a.document_id
                WHERE ({}) AND a.principal = ?
                ORDER BY chunk_index LIMIT ?
            """.format(" AND ".join(["c.content LIKE ?"] * len(terms)))
            params_fts: list[Any] = [fts_query, principal, top_k]
            like_terms = [f"%{t}%" for t in terms]
            params_like: list[Any] = like_terms + [principal, top_k]
        else:
            sql_fts = """
                SELECT c.* FROM chunks c
                JOIN chunks_fts f ON c.chunk_id = f.chunk_id
                WHERE chunks_fts MATCH ?
                ORDER BY rank LIMIT ?
            """
            sql_like = """
                SELECT c.* FROM chunks c
                WHERE ({})
                ORDER BY chunk_index LIMIT ?
            """.format(" AND ".join(["c.content LIKE ?"] * len(terms)))
            params_fts = [fts_query, top_k]
            like_terms = [f"%{t}%" for t in terms]
            params_like = like_terms + [top_k]
        with self._conn() as conn:
            rows = conn.execute(sql_fts, params_fts).fetchall()
            if not rows:
                rows = conn.execute(sql_like, params_like).fetchall()
        return [_row_to_chunk(row) for row in rows]

    # ── Citation CRUD ─────────────────────────────────────────────

    def insert_citation(self, citation: Citation) -> str:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO citations (
                    citation_id, chunk_id, document_id, document_title,
                    source_uri, page_number, slide_number, section,
                    quote, version, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    citation.citation_id, citation.chunk_id, citation.document_id,
                    citation.document_title, citation.source_uri, citation.page_number,
                    citation.slide_number, citation.section, citation.quote,
                    citation.version, _now_iso(),
                ),
            )
            conn.commit()
        return citation.citation_id

    def get_citations_for_document(self, document_id: str) -> list[Citation]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM citations WHERE document_id = ?", (document_id,)
            ).fetchall()
        return [_row_to_citation(row) for row in rows]

    # ── ACL CRUD ──────────────────────────────────────────────────

    def grant_acl(self, entry: AclEntry) -> str:
        """Grant an ACL entry. Returns the ACL key."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO acl (document_id, principal, permission, department, granted_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (entry.document_id, entry.principal, entry.permission, entry.department, _now_iso()),
            )
            conn.commit()
        return f"{entry.document_id}:{entry.principal}:{entry.permission}"

    def revoke_acl(self, document_id: str, principal: str, permission: str) -> bool:
        """Revoke a specific ACL entry."""
        with self._conn() as conn:
            cur = conn.execute(
                "DELETE FROM acl WHERE document_id = ? AND principal = ? AND permission = ?",
                (document_id, principal, permission),
            )
            conn.commit()
            return cur.rowcount > 0

    def check_acl(self, document_id: str, principal: str) -> list[str]:
        """Return list of permission strings the principal has on the document."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT permission FROM acl WHERE document_id = ? AND principal = ?",
                (document_id, principal),
            ).fetchall()
        return [row["permission"] for row in rows]

    def check_acl_detailed(self, document_id: str, principal: str) -> AclCheckResult:
        """Return detailed ACL result for a principal on a document."""
        perms = self.check_acl(document_id, principal)
        return AclCheckResult(
            document_id=document_id,
            principal=principal,
            can_read=ACL_READ in perms,
            can_write=ACL_WRITE in perms,
            can_delete=ACL_DELETE in perms,
            permissions=perms,
        )

    def check_acl_department(self, document_id: str, department: str) -> list[str]:
        """Check ACL permissions for a department on a document."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT permission FROM acl WHERE document_id = ? AND department = ?",
                (document_id, department),
            ).fetchall()
        return [row["permission"] for row in rows]

    def list_acls(
        self,
        *,
        document_id: str = "",
        principal: str = "",
        department: str = "",
        permission: str = "",
        limit: int = 100,
    ) -> list[AclEntry]:
        """List ACL entries with optional filtering."""
        conditions = []
        params: list[Any] = []
        if document_id:
            conditions.append("document_id = ?")
            params.append(document_id)
        if principal:
            conditions.append("principal = ?")
            params.append(principal)
        if department:
            conditions.append("department = ?")
            params.append(department)
        if permission:
            conditions.append("permission = ?")
            params.append(permission)

        query = "SELECT * FROM acl"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()

        return [
            AclEntry(
                document_id=row["document_id"],
                principal=row["principal"],
                permission=row["permission"],
                department=row.get("department", ""),
                granted_at=row["granted_at"],
            )
            for row in rows
        ]

    # ── Meeting Items CRUD ────────────────────────────────────────

    def insert_meeting_item(self, item: MeetingItem) -> str:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO meeting_items (
                    item_id, document_id, item_type, content, assignee,
                    due_date, priority, status, page_number, slide_number,
                    metadata_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    item.item_id, item.document_id, item.item_type, item.content,
                    item.assignee, item.due_date, item.priority, item.status,
                    item.page_number, item.slide_number, _json_dump(item.metadata),
                    _now_iso(),
                ),
            )
            conn.commit()
        return item.item_id

    def get_meeting_items(
        self,
        document_id: str,
        *,
        item_type: str = "",
        assignee: str = "",
        status: str = "",
    ) -> list[MeetingItem]:
        query = "SELECT * FROM meeting_items WHERE document_id = ?"
        params: list[Any] = [document_id]
        if item_type:
            query += " AND item_type = ?"
            params.append(item_type)
        if assignee:
            query += " AND assignee = ?"
            params.append(assignee)
        if status:
            query += " AND status = ?"
            params.append(status)
        query += " ORDER BY created_at DESC"
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_meeting_item(row) for row in rows]

    def list_meeting_items(
        self,
        *,
        item_type: str = "",
        assignee: str = "",
        due_before: str = "",
        due_after: str = "",
        limit: int = 100,
    ) -> list[MeetingItem]:
        """List meeting items across all documents."""
        conditions = []
        params: list[Any] = []
        if item_type:
            conditions.append("item_type = ?")
            params.append(item_type)
        if assignee:
            conditions.append("assignee = ?")
            params.append(assignee)
        if due_before:
            conditions.append("due_date <= ?")
            params.append(due_before)
        if due_after:
            conditions.append("due_date >= ?")
            params.append(due_after)
        query = "SELECT * FROM meeting_items"
        if conditions:
            query += " WHERE " + " AND ".join(conditions)
        query += " ORDER BY due_date ASC LIMIT ?"
        params.append(limit)
        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
        return [_row_to_meeting_item(row) for row in rows]

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        with self._conn() as conn:
            doc_count = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
            chunk_count = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
            citation_count = conn.execute("SELECT COUNT(*) FROM citations").fetchone()[0]
            meeting_count = conn.execute("SELECT COUNT(*) FROM meeting_items").fetchone()[0]
            acl_count = conn.execute("SELECT COUNT(*) FROM acl").fetchone()[0]
        return {
            "documents": doc_count,
            "chunks": chunk_count,
            "citations": citation_count,
            "meeting_items": meeting_count,
            "acl_entries": acl_count,
        }


# ── Helpers ───────────────────────────────────────────────────────

import json


def _json_dump(obj: dict[str, Any]) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))


def _json_load(text: str) -> dict[str, Any]:
    try:
        return json.loads(text or "{}")
    except json.JSONDecodeError:
        return {}  # type: ignore[return-value]


def _row_to_document(row: sqlite3.Row) -> Document:
    return Document(
        document_id=row["document_id"],
        title=row["title"],
        source_uri=row["source_uri"],
        document_type=row["document_type"],
        version=row["version"],
        author=row["author"],
        department=row["department"],
        created_at=row["created_at"],
        ingested_at=row["ingested_at"],
        content_hash=row["content_hash"],
        page_count=row["page_count"],
        slide_count=row["slide_count"],
        metadata=_json_load(row["metadata_json"]),
    )


def _row_to_chunk(row: sqlite3.Row) -> Chunk:
    return Chunk(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        chunk_index=row["chunk_index"],
        content=row["content"],
        content_hash=row["content_hash"],
        page_number=row["page_number"],
        slide_number=row["slide_number"],
        section=row["section"],
        start_char=row["start_char"],
        end_char=row["end_char"],
        token_count=row["token_count"],
        metadata=_json_load(row["metadata_json"]),
    )


def _row_to_citation(row: sqlite3.Row) -> Citation:
    return Citation(
        citation_id=row["citation_id"],
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        document_title=row["document_title"],
        source_uri=row["source_uri"],
        page_number=row["page_number"],
        slide_number=row["slide_number"],
        section=row["section"],
        quote=row["quote"],
        version=row["version"],
    )


def _row_to_meeting_item(row: sqlite3.Row) -> MeetingItem:
    return MeetingItem(
        item_id=row["item_id"],
        document_id=row["document_id"],
        item_type=row["item_type"],
        content=row["content"],
        assignee=row["assignee"],
        due_date=row["due_date"],
        priority=row["priority"],
        status=row["status"],
        page_number=row["page_number"],
        slide_number=row["slide_number"],
        metadata=_json_load(row["metadata_json"]),
    )
