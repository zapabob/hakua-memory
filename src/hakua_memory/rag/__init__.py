"""RAG module: document ingestion, retrieval, meeting extraction, contradictions."""

from .chunking import chunk_pages, chunk_slides, chunk_text
from .contradiction import Contradiction, detect_contradictions
from .ingestion import (
    ingest_document,
    ingest_markdown_string,
    ingest_text_string,
)
from .meeting import extract_meeting_items
from .models import AclEntry, Chunk, Citation, Document, MeetingItem
from .retrieval import RagResult, render_citation_context, search_chunks
from .schema import ALL_DDL
from .store import DocumentStore

__all__ = [
    "DocumentStore",
    "Document",
    "Chunk",
    "Citation",
    "MeetingItem",
    "AclEntry",
    "RagResult",
    "Contradiction",
    "ingest_document",
    "ingest_markdown_string",
    "ingest_text_string",
    "chunk_text",
    "chunk_pages",
    "chunk_slides",
    "search_chunks",
    "render_citation_context",
    "extract_meeting_items",
    "detect_contradictions",
    "ALL_DDL",
]
