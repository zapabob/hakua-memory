"""Document chunking with page/slide/section awareness."""

from __future__ import annotations

import hashlib
import uuid
from typing import Sequence

from .models import Chunk


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _estimate_tokens(text: str) -> int:
    """Rough token estimate without regex: CJK chars + whitespace-separated words."""
    cjk = 0
    words = 0
    in_word = False
    for ch in text:
        code = ord(ch)
        # CJK Unified + Hiragana/Katakana + Hangul syllables
        if (
            0x4E00 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0xAC00 <= code <= 0xD7AF
        ):
            cjk += 1
            in_word = False
        elif ch.isspace():
            in_word = False
        else:
            if not in_word:
                words += 1
                in_word = True
    return cjk + words


def chunk_text(
    text: str,
    document_id: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    page_number: int | None = None,
    slide_number: int | None = None,
    section: str = "",
    separator: str = "\n\n",
) -> list[Chunk]:
    """Split text into overlapping chunks with provenance."""
    if not text.strip():
        return []
    paragraphs = [p.strip() for p in text.split(separator) if p.strip()]
    if not paragraphs:
        paragraphs = [text.strip()]

    # Cache paragraph token lengths to avoid repeated full scans.
    para_lens = [_estimate_tokens(p) for p in paragraphs]

    chunks: list[Chunk] = []
    current: list[str] = []
    current_len = 0
    char_offset = 0
    idx = 0

    for para, para_len in zip(paragraphs, para_lens, strict=True):
        if current_len + para_len > chunk_size and current:
            content = "\n\n".join(current)
            token_count = _estimate_tokens(content)
            chunks.append(
                Chunk(
                    chunk_id=f"chunk-{uuid.uuid4().hex[:12]}",
                    document_id=document_id,
                    chunk_index=idx,
                    content=content,
                    content_hash=_sha256(content),
                    page_number=page_number,
                    slide_number=slide_number,
                    section=section,
                    start_char=char_offset,
                    end_char=char_offset + len(content),
                    token_count=token_count,
                )
            )
            idx += 1
            char_offset += len(content) + len(separator)
            # overlap: keep last chunk's tail
            if chunk_overlap > 0 and chunks:
                tail = chunks[-1].content[-chunk_overlap * 4:]
                current = [tail, para]
                current_len = _estimate_tokens(tail) + para_len
            else:
                current = [para]
                current_len = para_len
        else:
            current.append(para)
            current_len += para_len

    if current:
        content = "\n\n".join(current)
        chunks.append(
            Chunk(
                chunk_id=f"chunk-{uuid.uuid4().hex[:12]}",
                document_id=document_id,
                chunk_index=idx,
                content=content,
                content_hash=_sha256(content),
                page_number=page_number,
                slide_number=slide_number,
                section=section,
                start_char=char_offset,
                end_char=char_offset + len(content),
                token_count=_estimate_tokens(content),
            )
        )
    return chunks


def chunk_pages(
    pages: Sequence[str],
    document_id: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    section: str = "",
) -> list[Chunk]:
    """Chunk a sequence of pages, preserving page provenance."""
    all_chunks: list[Chunk] = []
    for page_num, page_text in enumerate(pages, start=1):
        page_chunks = chunk_text(
            page_text,
            document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            page_number=page_num,
            section=section,
        )
        all_chunks.extend(page_chunks)
    return all_chunks


def chunk_slides(
    slides: Sequence[str],
    document_id: str,
    *,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    section: str = "",
) -> list[Chunk]:
    """Chunk a sequence of slides, preserving slide provenance."""
    all_chunks: list[Chunk] = []
    for slide_num, slide_text in enumerate(slides, start=1):
        slide_chunks = chunk_text(
            slide_text,
            document_id,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            slide_number=slide_num,
            section=section,
        )
        all_chunks.extend(slide_chunks)
    return all_chunks
