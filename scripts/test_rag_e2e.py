"""End-to-end test for RAG module: ingestion, retrieval, meeting extraction, contradictions."""

from __future__ import annotations

import tempfile
from pathlib import Path

from hakua_memory.composite import CompositeMemory


def test_rag_e2e() -> None:
    tmpdir = tempfile.mkdtemp()
    try:
        root = Path(tmpdir) / "test_composite"
        cm = CompositeMemory(root)

        # ── 1. Ingest a meeting document (markdown string) ──────────
        meeting_md = """
# Project Kickoff Meeting 2026-08-20

## Attendees
- Alice (Engineering Lead)
- Bob (Product Manager)
- Carol (Design)

## Decisions
決定: 次期リリースは2026年9月30日を予定する
決定: 新機能Xを優先して開発する

## Tasks
タスク: 設計書の作成をAliceが担当する
タスク: プロトタイプをBobが作成する

## Action Items
アクションアイテム: 来週までにスケジュールを共有する（担当: Carol）

## Unresolved
未決: 予算枠の決定は来月まで保留
"""
        result = cm.ingest_markdown(
            meeting_md,
            title="Project Kickoff Meeting",
            source_uri="meeting://kickoff-2026-08-20",
            author="Alice",
            department="Engineering",
        )
        print(f"[ingest] {result}")
        assert result["chunks"] > 0

        # ── 2. Ingest a second document (contradicting info) ─────────
        update_md = """
# Project Update 2026-08-25

## Revised Schedule
決定: リリースは2026年10月15日に変更された
理由: 開発リソースの遅延のため
"""
        result2 = cm.ingest_markdown(
            update_md,
            title="Project Update",
            source_uri="meeting://update-2026-08-25",
            author="Bob",
            department="Product",
        )
        print(f"[ingest2] {result2}")

        # ── 3. Search documents ──────────────────────────────────────
        results = cm.search_documents("リリース", top_k=5)
        print(f"[search] {len(results)} results")
        for r in results:
            print(f"  - [{r['rank']}] {r['document_title']} (score={r['score']:.4f})")
            print(f"    page={r.get('page_number')}, section={r.get('section')}")
            print(f"    content: {r['content'][:80]}…")
        assert len(results) >= 2

        # ── 4. Render citation context ───────────────────────────────
        context = cm.render_citations(results, format="markdown")
        print(f"[citation] {len(context)} chars")
        assert "rag_context" in context
        assert "data_only" in context

        # ── 5. Extract meeting items ─────────────────────────────────
        doc_id = results[0]["document_id"]
        items = cm.extract_meeting_items(doc_id, auto_store=True)
        print(f"[meeting] {len(items)} items extracted")
        for item in items:
            print(f"  - [{item['item_type']}] {item['content'][:50]}")
            if item.get("assignee"):
                print(f"    assignee={item['assignee']}")

        # ── 6. Detect contradictions ─────────────────────────────────
        contradictions = cm.detect_contradictions(min_confidence=0.5)
        print(f"[contradictions] {len(contradictions)} found")
        for c in contradictions:
            print(f"  - [{c['type']}] {c['description'][:60]}")
            print(f"    confidence={c['confidence']:.2f}")

        # ── 7. ACL test ──────────────────────────────────────────────
        doc_list = cm.search_documents("Project", top_k=1)
        if doc_list:
            did = doc_list[0]["document_id"]
            cm.grant_access(did, "alice", "read")
            cm.grant_access(did, "bob", "read")
            perms = cm.check_access(did, "alice")
            print(f"[acl] alice permissions: {perms}")
            assert "read" in perms

            # Search with ACL filter: alice should see the doc
            alice_results = cm.search_documents("Project", top_k=5, principal="alice")
            print(f"[acl] alice sees {len(alice_results)} results")

        # ── 8. Stats ─────────────────────────────────────────────────
        stats = cm.stats()
        print(f"[stats] {stats}")
        assert stats["rag"]["documents"] >= 2
        assert stats["rag"]["chunks"] > 0

        print("\n★ ALL RAG E2E TESTS PASSED ★")
    finally:
        # Close the ebbinghaus connection before cleanup (Windows file lock)
        cm.close()
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)
        print("★ TEST COMPLETE ★")


if __name__ == "__main__":
    test_rag_e2e()
