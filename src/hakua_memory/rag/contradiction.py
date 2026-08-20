"""Contradiction detection between documents."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from .models import Chunk
from .store import DocumentStore

# Contradiction signal patterns (Japanese + English)
NEGATION_PATTERNS = [
    r"(?:ない|ません|ではない|できない|禁止|不可|しない)",
    r"(?:not|no|never|cannot|must not|prohibited|forbidden)",
]

CONTRAST_PATTERNS = [
    r"(?:しかし|だが|一方|逆に|反対|ただし|ところが)",
    r"(?:however|but|on the other hand|conversely|instead|nevertheless|although)",
]

DATE_NUM_PATTERNS = [
    r"(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})",
    r"(\d+(?:\.\d+)?)\s*(%|percent|円|ドル|USD|JPY|人|件|個)",
]


@dataclass(frozen=True)
class Contradiction:
    """A detected contradiction between two chunks."""

    contradiction_id: str
    chunk_a_id: str
    chunk_b_id: str
    document_a_id: str
    document_b_id: str
    document_a_title: str
    document_b_title: str
    contradiction_type: str  # negation, date_mismatch, number_mismatch, direct_contrast
    description: str
    quote_a: str = ""
    quote_b: str = ""
    confidence: float = 0.5
    page_a: int | None = None
    page_b: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


def detect_contradictions(
    store: DocumentStore,
    *,
    document_ids: list[str] | None = None,
    min_confidence: float = 0.6,
) -> list[Contradiction]:
    """Detect contradictions between documents or within a single document.

    Uses a heuristic approach based on:
    - Negation signals on similar topics
    - Date mismatches for the same event/milestone
    - Number mismatches for the same metric
    - Direct contrast markers (しかし, however)

    This is intentionally conservative: false positives are preferable to
    missing real contradictions in a RAG system.

    Args:
        store: DocumentStore with ingested chunks.
        document_ids: If provided, only check these documents.
        min_confidence: Minimum confidence threshold.

    Returns:
        List of detected contradictions.
    """
    # Get all chunks for the specified documents
    docs: dict[str, Any] = {}

    if document_ids:
        for doc_id in document_ids:
            doc = store.get_document(doc_id)
            if doc:
                docs[doc_id] = doc
    else:
        # Get all documents
        all_docs = store.list_documents(limit=1000)
        for doc in all_docs:
            docs[doc.document_id] = doc

    # Collect chunks per document
    doc_chunks: dict[str, list[Chunk]] = {}
    for doc_id in docs:
        doc_chunks[doc_id] = _get_chunks_for_document(store, doc_id)

    contradictions: list[Contradiction] = []
    doc_list = list(docs.keys())

    # Compare chunks
    for i, doc_a in enumerate(doc_list):
        for doc_b in doc_list[i:]:
            chunks_a = doc_chunks.get(doc_a, [])
            chunks_b = doc_chunks.get(doc_b, [])
            for chunk_a in chunks_a:
                for chunk_b in chunks_b:
                    if chunk_a.chunk_id == chunk_b.chunk_id:
                        continue
                    contradiction = _compare_chunks(
                        chunk_a, chunk_b, docs[doc_a], docs[doc_b]
                    )
                    if contradiction and contradiction.confidence >= min_confidence:
                        contradictions.append(contradiction)

    return contradictions


def _compare_chunks(
    chunk_a: Chunk,
    chunk_b: Chunk,
    doc_a: Any,
    doc_b: Any,
) -> Contradiction | None:
    """Compare two chunks for contradictions."""
    # Check for direct contrast markers
    has_contrast_a = any(re.search(p, chunk_a.content) for p in CONTRAST_PATTERNS)

    # Check for negation
    has_negation_a = any(re.search(p, chunk_a.content) for p in NEGATION_PATTERNS)
    has_negation_b = any(re.search(p, chunk_b.content) for p in NEGATION_PATTERNS)

    # If one has negation and the other doesn't, and they share similar terms
    if has_negation_a != has_negation_b:
        similarity = _chunk_similarity(chunk_a.content, chunk_b.content)
        if similarity > 0.3:
            return Contradiction(
                contradiction_id=f"c-{chunk_a.chunk_id[:6]}-{chunk_b.chunk_id[:6]}",
                chunk_a_id=chunk_a.chunk_id,
                chunk_b_id=chunk_b.chunk_id,
                document_a_id=doc_a.document_id,
                document_b_id=doc_b.document_id,
                document_a_title=doc_a.title,
                document_b_title=doc_b.title,
                contradiction_type="negation",
                description=f"Possible negation contradiction between '{doc_a.title}' and '{doc_b.title}'",
                quote_a=chunk_a.content[:200],
                quote_b=chunk_b.content[:200],
                confidence=0.6 + similarity * 0.3,
                page_a=chunk_a.page_number,
                page_b=chunk_b.page_number,
            )

    # Check for date/number mismatches on similar topics
    dates_a = set(re.findall(DATE_NUM_PATTERNS[0], chunk_a.content))
    dates_b = set(re.findall(DATE_NUM_PATTERNS[0], chunk_b.content))
    if dates_a and dates_b and dates_a != dates_b:
        similarity = _chunk_similarity(chunk_a.content, chunk_b.content)
        if similarity > 0.4:
            return Contradiction(
                contradiction_id=f"d-{chunk_a.chunk_id[:6]}-{chunk_b.chunk_id[:6]}",
                chunk_a_id=chunk_a.chunk_id,
                chunk_b_id=chunk_b.chunk_id,
                document_a_id=doc_a.document_id,
                document_b_id=doc_b.document_id,
                document_a_title=doc_a.title,
                document_b_title=doc_b.title,
                contradiction_type="date_mismatch",
                description=f"Date mismatch: {dates_a} vs {dates_b}",
                quote_a=chunk_a.content[:200],
                quote_b=chunk_b.content[:200],
                confidence=0.7,
                page_a=chunk_a.page_number,
                page_b=chunk_b.page_number,
            )

    # Check for direct contrast
    if has_contrast_a or has_negation_a != has_negation_b:
        similarity = _chunk_similarity(chunk_a.content, chunk_b.content)
        if similarity > 0.5:
            return Contradiction(
                contradiction_id=f"dc-{chunk_a.chunk_id[:6]}-{chunk_b.chunk_id[:6]}",
                chunk_a_id=chunk_a.chunk_id,
                chunk_b_id=chunk_b.chunk_id,
                document_a_id=doc_a.document_id,
                document_b_id=doc_b.document_id,
                document_a_title=doc_a.title,
                document_b_title=doc_b.title,
                contradiction_type="direct_contrast",
                description="Direct contrast detected between chunks",
                quote_a=chunk_a.content[:200],
                quote_b=chunk_b.content[:200],
                confidence=0.65,
                page_a=chunk_a.page_number,
                page_b=chunk_b.page_number,
            )

    return None


def _chunk_similarity(text_a: str, text_b: str) -> float:
    """Simple Jaccard similarity of token sets."""
    tokens_a = set(_tokenize(text_a))
    tokens_b = set(_tokenize(text_b))
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def _tokenize(text: str) -> list[str]:
    """Simple tokenization: split on whitespace and extract CJK chars."""
    tokens = text.split()
    cjk = re.findall(r"[一-鿿぀-ヿ가-힯]", text)
    return tokens + cjk


def _get_chunks_for_document(store: DocumentStore, document_id: str) -> list[Chunk]:
    """Get all chunks for a document."""
    # Use the existing search method with a broad query or add a direct query
    with store._conn() as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM chunks WHERE document_id = ? ORDER BY chunk_index",
            (document_id,),
        ).fetchall()
    from .store import _row_to_chunk
    return [_row_to_chunk(row) for row in rows]


import sqlite3  # noqa: E402
