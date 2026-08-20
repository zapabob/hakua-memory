"""Obsidian Wiki bridge for hakua-memory."""

from __future__ import annotations

from pathlib import Path

from hakua_memory.ebbinghaus.store import _normalize_text


def _safe_slug(text: str, max_len: int = 64) -> str:
    text = _normalize_text(text)
    text = text.replace(" ", "-")
    text = "".join(ch for ch in text if ch.isalnum() or ch in "-_.")
    return text[:max_len] or "note"


def write_diary(wiki_root: Path, title: str, body: str) -> Path:
    wiki_root.mkdir(parents=True, exist_ok=True)
    diary = wiki_root / "diary"
    diary.mkdir(exist_ok=True)
    path = diary / f"{_safe_slug(title)}.md"
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")
    return path


def write_dream(wiki_root: Path, body: str) -> Path:
    wiki_root.mkdir(parents=True, exist_ok=True)
    dreams = wiki_root / "dreams"
    dreams.mkdir(exist_ok=True)
    path = dreams / "latest.md"
    path.write_text(f"# Dream\n\n{body}\n", encoding="utf-8")
    return path
