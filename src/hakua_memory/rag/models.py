"""Data models for the RAG (Retrieval-Augmented Generation) module."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# ── ACL permissions ─────────────────────────────────────────────────

#: Three-level ACL: read / write / delete
ACL_READ = "read"
ACL_WRITE = "write"
ACL_DELETE = "delete"

ALL_ACL_PERMISSIONS = (ACL_READ, ACL_WRITE, ACL_DELETE)


def _normalize_permission(permission: str) -> str:
    """Normalize and validate an ACL permission string."""
    p = permission.strip().lower()
    if p not in ALL_ACL_PERMISSIONS:
        raise ValueError(
            f"Invalid ACL permission: {permission!r}. "
            f"Must be one of: {', '.join(ALL_ACL_PERMISSIONS)}"
        )
    return p


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
    permission: str = ACL_READ  # "read" | "write" | "delete"
    department: str = ""  # optional department scope
    granted_at: str = ""

    def __post_init__(self) -> None:
        # Normalize and validate permission
        object.__setattr__(
            self, "permission", _normalize_permission(self.permission)
        )
        if not self.document_id.strip():
            raise ValueError("document_id must not be empty")
        if not self.principal.strip():
            raise ValueError("principal must not be empty")

    def can_read(self) -> bool:
        return self.permission == ACL_READ

    def can_write(self) -> bool:
        return self.permission == ACL_WRITE

    def can_delete(self) -> bool:
        return self.permission == ACL_DELETE


@dataclass(frozen=True)
class AclCheckResult:
    """Result of an ACL check for a principal on a document."""

    document_id: str
    principal: str
    can_read: bool
    can_write: bool
    can_delete: bool
    permissions: list[str] = field(default_factory=list)

    def has(self, permission: str) -> bool:
        """Check if a specific permission is granted."""
        if permission == ACL_READ:
            return self.can_read
        if permission == ACL_WRITE:
            return self.can_write
        if permission == ACL_DELETE:
            return self.can_delete
        return False

    def has_all(self, *permissions: str) -> bool:
        """Check if all specified permissions are granted."""
        return all(self.has(p) for p in permissions)

    def has_any(self, *permissions: str) -> bool:
        """Check if any of the specified permissions is granted."""
        return any(self.has(p) for p in permissions)
