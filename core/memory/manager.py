from __future__ import annotations

from .models import Memory, MemoryType, UserModelEntry
from .store import MemoryStore


class MemoryManager:
    """
    High-level memory interface for SALLY.

    MemoryStore handles SQLite/FTS5 persistence.
    MemoryManager handles how the rest of SALLY interacts with memory.
    """

    def __init__(self, store: MemoryStore | None = None) -> None:
        self.store = store or MemoryStore()

    def remember(
        self,
        content: str,
        *,
        memory_type: MemoryType = MemoryType.EPISODE,
        importance: float = 0.5,
    ) -> Memory:
        """Store an explicit memory."""
        content = content.strip()

        if not content:
            raise ValueError("Memory content cannot be empty.")

        return self.store.save(
            content,
            memory_type=memory_type,
            importance=importance,
        )

    def search(
        self,
        query: str,
        *,
        limit: int = 5,
    ) -> list[Memory]:
        """Find memories relevant to a query."""
        return self.store.search(query, limit=limit)

    def recent(self, *, limit: int = 10) -> list[Memory]:
        """Return the most recently stored memories."""
        return self.store.recent(limit=limit)

    def profile(self) -> list[UserModelEntry]:
        """Return SALLY's structured user-model entries."""
        return self.store.get_user_model()
