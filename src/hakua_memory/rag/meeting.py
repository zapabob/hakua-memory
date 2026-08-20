"""Meeting extraction: decisions, tasks, action items, unresolved items."""

from __future__ import annotations

import re
import uuid
from typing import Any, Optional

from .models import Chunk, MeetingItem
from .store import DocumentStore


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


# Patterns for extracting meeting items (Japanese + English)
DECISION_PATTERNS = [
    r"(?:決定|決定事項|決議|承認|合意|結論)[：:\s]\s*(.+)",
    r"(?:it was decided|decision|resolved|agreed|concluded)[:\s]+(.+)",
]

TASK_PATTERNS = [
    r"(?:タスク|作業|担当|実施|実行)[：:\s]\s*(.+)",
    r"(?:action item|task|todo|assigned to|owner)[:\s]+(.+)",
]

ACTION_ITEM_PATTERNS = [
    r"(?:アクションアイテム|対応|処置|フォローアップ)[：:\s]\s*(.+)",
    r"(?:follow.?up|next step|action required)[:\s]+(.+)",
]

UNRESOLVED_PATTERNS = [
    r"(?:未決|保留|課題|検討中)[：:\s]\s*(.+)",
    r"(?:pending|open issue|tbd)[:\s]+(.+)",
]

ASSIGNEE_PATTERNS = [
    r"(?:担当|担当者|assigned to|owner|responsible)[：:\s]*([^\n、,]+?)(?:\n|$|、|,|\)|）)",
]

DUE_DATE_PATTERNS = [
    r"(?:期限|予定日|due date|deadline|by)[：:\s]*(\d{4}[-/]\d{1,2}[-/]\d{1,2})",
    r"(?:期限|予定日|due date|deadline|by)[：:\s]*(.+?)(?:\n|$|、|,)",
]


def extract_meeting_items(
    document_id: str,
    chunks: list[Chunk],
    *,
    store: Optional[DocumentStore] = None,
    auto_store: bool = True,
) -> list[MeetingItem]:
    """Extract structured meeting items from chunks.

    Args:
        document_id: The document ID.
        chunks: List of chunks to analyze.
        store: Optional DocumentStore to persist items.
        auto_store: Whether to automatically store items in the database.

    Returns:
        List of extracted MeetingItem objects.
    """
    items: list[MeetingItem] = []

    for chunk in chunks:
        # Extract decisions
        for pattern in DECISION_PATTERNS:
            for match in re.finditer(pattern, chunk.content, re.IGNORECASE):
                items.append(_make_item(document_id, "decision", match.group(1).strip(), chunk))

        # Extract tasks
        for pattern in TASK_PATTERNS:
            for match in re.finditer(pattern, chunk.content, re.IGNORECASE):
                item = _make_item(document_id, "task", match.group(1).strip(), chunk)
                # Try to find assignee
                assignee = _find_assignee(chunk.content)
                if assignee:
                    item = MeetingItem(
                        item_id=item.item_id,
                        document_id=item.document_id,
                        item_type=item.item_type,
                        content=item.content,
                        assignee=assignee,
                        due_date=_find_due_date(chunk.content),
                        page_number=item.page_number,
                        slide_number=item.slide_number,
                    )
                items.append(item)

        # Extract action items
        for pattern in ACTION_ITEM_PATTERNS:
            for match in re.finditer(pattern, chunk.content, re.IGNORECASE):
                item = _make_item(document_id, "action_item", match.group(1).strip(), chunk)
                assignee = _find_assignee(chunk.content)
                if assignee:
                    item = MeetingItem(
                        item_id=item.item_id,
                        document_id=item.document_id,
                        item_type=item.item_type,
                        content=item.content,
                        assignee=assignee,
                        due_date=_find_due_date(chunk.content),
                        page_number=item.page_number,
                        slide_number=item.slide_number,
                    )
                items.append(item)

        # Extract unresolved items
        for pattern in UNRESOLVED_PATTERNS:
            for match in re.finditer(pattern, chunk.content, re.IGNORECASE):
                items.append(_make_item(document_id, "unresolved", match.group(1).strip(), chunk))

    if store and auto_store:
        for item in items:
            store.insert_meeting_item(item)

    return items


def _make_item(
    document_id: str, item_type: str, content: str, chunk: Chunk
) -> MeetingItem:
    return MeetingItem(
        item_id=f"mi-{uuid.uuid4().hex[:12]}",
        document_id=document_id,
        item_type=item_type,
        content=content,
        page_number=chunk.page_number,
        slide_number=chunk.slide_number,
    )


def _find_assignee(text: str) -> str:
    for pattern in ASSIGNEE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""


def _find_due_date(text: str) -> str:
    for pattern in DUE_DATE_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
    return ""
