from pathlib import Path

from hakua_memory.ebbinghaus.store import EbbinghausMemoryStore, forgetting_retention


def test_forgetting_retention_boundaries() -> None:
    assert forgetting_retention(0, 3) == 1.0
    assert 0.0 < forgetting_retention(3, 3) < 1.0
    assert forgetting_retention(30, 3) < forgetting_retention(3, 3)


def test_remember_recall_and_reinforcement(tmp_path: Path) -> None:
    store = EbbinghausMemoryStore(tmp_path / "ebbinghaus.db")
    try:
        created = store.remember(
            "User prefers concise Japanese reports.",
            tags=["user-pref"],
        )
        memory_id = int(created["memory_id"])
        before = int(store.get(memory_id)["retrieval_count"])

        results = store.recall("Japanese reports", reinforce=True)

        assert results
        assert results[0]["memory_id"] == memory_id
        assert int(store.get(memory_id)["retrieval_count"]) > before
    finally:
        store.close()
