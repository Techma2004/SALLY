from core.memory import MemoryStore, MemoryType


def test_memory_store_save_and_search(tmp_path):
    db_path = tmp_path / "test_memory.db"
    store = MemoryStore(db_path=str(db_path))

    saved = store.save(
        "SALLY isolated memory store test.",
        memory_type=MemoryType.FACT,
        importance=0.8,
    )

    assert saved.content == "SALLY isolated memory store test."
    assert saved.memory_type is MemoryType.FACT

    results = store.search("isolated memory store", limit=5)

    assert any(item.id == saved.id for item in results)


def test_memory_store_recent(tmp_path):
    db_path = tmp_path / "test_memory.db"
    store = MemoryStore(db_path=str(db_path))

    saved = store.save(
        "Recent memory test.",
        memory_type=MemoryType.EPISODE,
    )

    recent = store.recent(limit=10)

    assert any(item.id == saved.id for item in recent)
