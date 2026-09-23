from core.memory import MemoryManager, MemoryStore, MemoryType


def make_memory_manager(tmp_path):
    store = MemoryStore(db_path=str(tmp_path / "test_memory.db"))
    return MemoryManager(store=store)


def test_memory_manager_remember_and_search(tmp_path):
    memory = make_memory_manager(tmp_path)

    saved = memory.remember(
        "SALLY isolated MemoryManager test.",
        memory_type=MemoryType.EPISODE,
        importance=0.5,
    )

    assert saved.content == "SALLY isolated MemoryManager test."
    assert saved.memory_type is MemoryType.EPISODE

    results = memory.search("isolated MemoryManager", limit=10)

    assert any(item.id == saved.id for item in results)


def test_memory_manager_recent(tmp_path):
    memory = make_memory_manager(tmp_path)

    saved = memory.remember(
        "SALLY recent memory test.",
        memory_type=MemoryType.FACT,
    )

    recent = memory.recent(limit=10)

    assert any(item.id == saved.id for item in recent)


def test_memory_manager_profile(tmp_path):
    memory = make_memory_manager(tmp_path)

    entries = memory.profile()

    assert isinstance(entries, list)
