from core.memory import MemoryManager, MemoryType


TEST_CONTENT = "SALLY V3.5 MemoryManager test record."


def main() -> None:
    print("=== SALLY V3.5 MEMORY MANAGER ===")

    memory = MemoryManager()

    before = memory.search("SALLY V3.5 MemoryManager", limit=10)
    print(f"Existing matching memories: {len(before)}")

    saved = memory.remember(
        TEST_CONTENT,
        memory_type=MemoryType.EPISODE,
        importance=0.7,
    )

    print("\nSaved:")
    print("  ID        :", saved.id)
    print("  Type      :", saved.memory_type.value)
    print("  Importance:", saved.importance)

    results = memory.search("SALLY V3.5 MemoryManager", limit=5)

    print("\nSearch results:")
    for item in results:
        print(f"  [{item.memory_type.value}] {item.content}")

    assert any(item.id == saved.id for item in results)

    recent = memory.recent(limit=5)

    print("\nRecent memories:")
    for item in recent:
        print(f"  [{item.memory_type.value}] {item.content}")

    profile = memory.profile()
    print(f"\nUser model entries: {len(profile)}")

    print("\n✅ MEMORY MANAGER TEST PASSED")


if __name__ == "__main__":
    main()
