"""Data models for the RAG (Retrieval-Augmented Generation) module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """A source document ingested into the RAG system."""

    document_id: str
    title: str
    source_uri: str
    document_type: str  # pdf, docx, pptx, markdown, text
    version: str = "1"
    author: str = ""
    department: str = ""
    created_at: str = ""
    ingested_at: str = ""
    content_hash: str = ""
    page_count: int = 0
    slide_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Chunk:
    """A chunk of a document with provenance information."""

    chunk_id: str
    document_id: str
    chunk_index: int
    content: str
    content_hash: str = ""
    page_number: int | None = None
    slide_number: int | None = None
    section: str = ""
    start_char: int = 0
    end_char: int = 0
    token_count: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Citation:
    """A citation pointing to a specific chunk in a document."""

    citation_id: str
    chunk_id: str
    document_id: str
    document_title: str
    source_uri: str
    page_number: int | None = None
    slide_number: int | None = None
    section: str = ""
    quote: str = ""
    version: str = ""


@dataclass(frozen=True)
class MeetingItem:
    """An extracted item from a meeting document."""

    item_id: str
    document_id: str
    item_type: str  # decision, action_item, unresolved, task
    content: str
    assignee: str = ""
    due_date: str = ""
    priority: str = ""
    status: str = ""
    page_number: int | None = None
    slide_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AclEntry:
    """Access control entry for a document."""

    document_id: str
    principal: str  # user, group, or role
    permission: str  # read, write, admin
    granted_at: str = ""
