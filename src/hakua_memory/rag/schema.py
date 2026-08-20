"""SQLite schema for the RAG module."""

from __future__ import annotations

DDL_DOCUMENTS = """
CREATE TABLE IF NOT EXISTS documents (
    document_id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    document_type TEXT NOT NULL,
    version TEXT NOT NULL DEFAULT '1',
    author TEXT NOT NULL DEFAULT '',
    department TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL DEFAULT '',
    ingested_at TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    page_count INTEGER NOT NULL DEFAULT 0,
    slide_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_documents_type ON documents(document_type);
CREATE INDEX IF NOT EXISTS idx_documents_hash ON documents(content_hash);
CREATE INDEX IF NOT EXISTS idx_documents_source ON documents(source_uri);
CREATE INDEX IF NOT EXISTS idx_documents_department ON documents(department);
"""

DDL_CHUNKS = """
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    content TEXT NOT NULL,
    content_hash TEXT NOT NULL,
    page_number INTEGER,
    slide_number INTEGER,
    section TEXT NOT NULL DEFAULT '',
    start_char INTEGER NOT NULL DEFAULT 0,
    end_char INTEGER NOT NULL DEFAULT 0,
    token_count INTEGER NOT NULL DEFAULT 0,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunks_document ON chunks(document_id);
CREATE INDEX IF NOT EXISTS idx_chunks_hash ON chunks(content_hash);
CREATE INDEX IF NOT EXISTS idx_chunks_page ON chunks(document_id, page_number);
CREATE INDEX IF NOT EXISTS idx_chunks_slide ON chunks(document_id, slide_number);
"""

DDL_CHUNK_EMBEDDINGS = """
CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id TEXT NOT NULL,
    namespace TEXT NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL,
    revision TEXT NOT NULL DEFAULT '',
    serializer_version INTEGER NOT NULL CHECK(serializer_version > 0),
    dimensions INTEGER NOT NULL CHECK(dimensions > 0),
    dtype TEXT NOT NULL DEFAULT 'float32-le' CHECK(dtype = 'float32-le'),
    vector_blob BLOB NOT NULL CHECK(length(vector_blob) = dimensions * 4),
    source_text_hash TEXT NOT NULL CHECK(length(source_text_hash) = 64),
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY(chunk_id, namespace),
    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_namespace ON chunk_embeddings(namespace);
CREATE INDEX IF NOT EXISTS idx_chunk_embeddings_source_hash ON chunk_embeddings(namespace, source_text_hash);
"""

DDL_CITATIONS = """
CREATE TABLE IF NOT EXISTS citations (
    citation_id TEXT PRIMARY KEY,
    chunk_id TEXT NOT NULL,
    document_id TEXT NOT NULL,
    document_title TEXT NOT NULL,
    source_uri TEXT NOT NULL,
    page_number INTEGER,
    slide_number INTEGER,
    section TEXT NOT NULL DEFAULT '',
    quote TEXT NOT NULL DEFAULT '',
    version TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL,
    FOREIGN KEY(chunk_id) REFERENCES chunks(chunk_id) ON DELETE CASCADE,
    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_citations_chunk ON citations(chunk_id);
CREATE INDEX IF NOT EXISTS idx_citations_document ON citations(document_id);
"""

DDL_MEETING_ITEMS = """
CREATE TABLE IF NOT EXISTS meeting_items (
    item_id TEXT PRIMARY KEY,
    document_id TEXT NOT NULL,
    item_type TEXT NOT NULL,
    content TEXT NOT NULL,
    assignee TEXT NOT NULL DEFAULT '',
    due_date TEXT NOT NULL DEFAULT '',
    priority TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT '',
    page_number INTEGER,
    slide_number INTEGER,
    metadata_json TEXT NOT NULL DEFAULT '{}',
    created_at TEXT NOT NULL,
    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_meeting_items_document ON meeting_items(document_id);
CREATE INDEX IF NOT EXISTS idx_meeting_items_type ON meeting_items(item_type);
CREATE INDEX IF NOT EXISTS idx_meeting_items_assignee ON meeting_items(assignee);
CREATE INDEX IF NOT EXISTS idx_meeting_items_due ON meeting_items(due_date);
"""

DDL_ACL = """
CREATE TABLE IF NOT EXISTS acl (
    document_id TEXT NOT NULL,
    principal TEXT NOT NULL,
    permission TEXT NOT NULL,
    granted_at TEXT NOT NULL,
    PRIMARY KEY(document_id, principal, permission),
    FOREIGN KEY(document_id) REFERENCES documents(document_id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_acl_principal ON acl(principal);
CREATE INDEX IF NOT EXISTS idx_acl_document ON acl(document_id);
"""

DDL_CHUNKS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS chunks_fts USING fts5(
    chunk_id UNINDEXED,
    content,
    tokenize='unicode61'
);
"""

DDL_DOCUMENTS_FTS = """
CREATE VIRTUAL TABLE IF NOT EXISTS documents_fts USING fts5(
    document_id UNINDEXED,
    title,
    tokenize='unicode61'
);
"""

ALL_DDL = [
    DDL_DOCUMENTS,
    DDL_CHUNKS,
    DDL_CHUNK_EMBEDDINGS,
    DDL_CITATIONS,
    DDL_MEETING_ITEMS,
    DDL_ACL,
    DDL_CHUNKS_FTS,
    DDL_DOCUMENTS_FTS,
]
