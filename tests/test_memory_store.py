from core.memory import MemoryStore, MemoryType


def main() -> None:
    print("=== SALLY V3.5 MEMORY STORE ===")

    store = MemoryStore()

    before = store.recent(100)
    print(f"Existing memories: {len(before)}")

    memory = store.save(
        "SALLY memory store V3.5 test.",
        memory_type=MemoryType.EPISODE,
        importance=0.7,
    )

    print("\nSaved:")
    print("  ID        :", memory.id)
    print("  Type      :", memory.memory_type.value)
    print("  Importance:", memory.importance)

    results = store.search("SALLY memory store V3.5", limit=5)

    print("\nSearch results:")
    for item in results:
        print(f"  [{item.memory_type.value}] {item.content}")

    assert results
    assert any(item.id == memory.id for item in results)

    recent = store.recent(5)

    print("\nRecent memories:")
    for item in recent:
        print(f"  [{item.memory_type.value}] {item.content}")

    user_model = store.get_user_model()
    print(f"\nUser model entries: {len(user_model)}")

    print("\n✅ MEMORY STORE TEST PASSED")


if __name__ == "__main__":
    main()
