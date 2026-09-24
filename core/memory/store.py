from __future__ import annotations

import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

from core.config import PROJECT_ROOT, settings

from .models import (
    Conversation,
    ConversationMessage,
    Memory,
    MemoryType,
    UserModelEntry,
)


class MemoryStore:
    """Low-level SQLite storage for SALLY memory and conversations."""

    def __init__(self, db_path: str | Path | None = None) -> None:
        path = Path(db_path or settings.memory.db_path)

        if not path.is_absolute():
            path = PROJECT_ROOT / path

        path.parent.mkdir(parents=True, exist_ok=True)
        self.db_path = path

        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(
            self.db_path,
            timeout=10,
        )
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=5000")
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS memories (
                    id TEXT PRIMARY KEY,
                    type TEXT,
                    content TEXT,
                    importance REAL DEFAULT 0.5,
                    created_at TEXT
                );

                CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
                USING fts5(
                    content,
                    content='memories',
                    content_rowid='rowid'
                );

                CREATE TRIGGER IF NOT EXISTS memories_ai
                AFTER INSERT ON memories
                BEGIN
                    INSERT INTO memories_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_ad
                AFTER DELETE ON memories
                BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);
                END;

                CREATE TRIGGER IF NOT EXISTS memories_au
                AFTER UPDATE ON memories
                BEGIN
                    INSERT INTO memories_fts(memories_fts, rowid, content)
                    VALUES ('delete', old.rowid, old.content);

                    INSERT INTO memories_fts(rowid, content)
                    VALUES (new.rowid, new.content);
                END;

                CREATE TABLE IF NOT EXISTS user_model (
                    key TEXT PRIMARY KEY,
                    value TEXT,
                    confidence REAL DEFAULT 0.5,
                    updated_at TEXT
                );

                CREATE TABLE IF NOT EXISTS sessions (
                    id TEXT PRIMARY KEY,
                    started_at TEXT,
                    summary TEXT
                );

                CREATE TABLE IF NOT EXISTS tool_usage (
                    tool TEXT PRIMARY KEY,
                    count INTEGER DEFAULT 0,
                    last_used TEXT,
                    examples TEXT
                );

                CREATE TABLE IF NOT EXISTS conversations (
                    id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    title TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    conversation_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    FOREIGN KEY(conversation_id)
                        REFERENCES conversations(id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_user_updated
                ON conversations(user_id, updated_at DESC);

                CREATE INDEX IF NOT EXISTS idx_conversation_messages_conversation
                ON conversation_messages(conversation_id, id);
                """
            )

    def save(
        self,
        content: str,
        memory_type: MemoryType = MemoryType.EPISODE,
        importance: float = 0.5,
    ) -> Memory:
        content = content.strip()

        if not content:
            raise ValueError("Memory content cannot be empty.")

        if len(content) > 5000:
            raise ValueError("Memory content cannot exceed 5000 characters.")

        importance = max(0.0, min(1.0, float(importance)))

        memory_id = uuid.uuid4().hex[:12]
        created_at = datetime.now(timezone.utc)

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO memories
                    (id, type, content, importance, created_at)
                VALUES
                    (?, ?, ?, ?, ?)
                """,
                (
                    memory_id,
                    memory_type.value,
                    content,
                    importance,
                    created_at.isoformat(),
                ),
            )

        return Memory(
            id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=created_at,
        )

    def search(
        self,
        query: str,
        limit: int = 5,
    ) -> list[Memory]:
        query = query.strip()

        if not query:
            return []

        limit = max(1, min(limit, 50))

        terms = [
            term.strip('.,!?;:"\'()[]{}')
            for term in query.split()
        ]
        terms = [term for term in terms if term]

        fts_query = " OR ".join(
            f'"{term}"*'
            for term in terms
        )

        with self._connect() as connection:
            try:
                rows = connection.execute(
                    """
                    SELECT
                        m.id,
                        m.content,
                        m.type,
                        m.importance,
                        m.created_at
                    FROM memories_fts f
                    JOIN memories m
                        ON m.rowid = f.rowid
                    WHERE memories_fts MATCH ?
                    ORDER BY rank
                    LIMIT ?
                    """,
                    (fts_query, limit),
                ).fetchall()
            except sqlite3.OperationalError:
                rows = connection.execute(
                    """
                    SELECT id, content, type, importance, created_at
                    FROM memories
                    WHERE content LIKE ?
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (f"%{query}%", limit),
                ).fetchall()

        return [self._memory_from_row(row) for row in rows]

    def recent(self, limit: int = 10) -> list[Memory]:
        limit = max(1, min(limit, 50))

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, content, type, importance, created_at
                FROM memories
                ORDER BY created_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [self._memory_from_row(row) for row in rows]

    def get_user_model(self) -> list[UserModelEntry]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT key, value, confidence, updated_at
                FROM user_model
                ORDER BY key
                """
            ).fetchall()

        return [
            UserModelEntry(
                key=row["key"],
                value=row["value"],
                confidence=float(row["confidence"]),
                updated_at=self._parse_datetime(row["updated_at"]),
            )
            for row in rows
        ]

    def create_conversation(
        self,
        user_id: str,
        *,
        title: str = "New Conversation",
    ) -> Conversation:
        conversation_id = uuid.uuid4().hex
        now = datetime.now(timezone.utc).isoformat()

        clean_user_id = user_id.strip() or "anonymous"
        clean_title = title.strip() or "New Conversation"
        clean_title = clean_title[:120]

        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO conversations
                    (id, user_id, title, created_at, updated_at)
                VALUES
                    (?, ?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    clean_user_id,
                    clean_title,
                    now,
                    now,
                ),
            )

        return Conversation(
            id=conversation_id,
            user_id=clean_user_id,
            title=clean_title,
            created_at=self._parse_datetime(now),
            updated_at=self._parse_datetime(now),
        )

    def get_conversation(
        self,
        conversation_id: str,
        user_id: str,
    ) -> Conversation | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM conversations
                WHERE id = ? AND user_id = ?
                """,
                (
                    conversation_id,
                    user_id,
                ),
            ).fetchone()

        if row is None:
            return None

        return self._conversation_from_row(row)

    def recent_conversations(
        self,
        user_id: str,
        limit: int = 20,
    ) -> list[Conversation]:
        limit = max(1, min(limit, 100))

        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, user_id, title, created_at, updated_at
                FROM conversations
                WHERE user_id = ?
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (
                    user_id,
                    limit,
                ),
            ).fetchall()

        return [self._conversation_from_row(row) for row in rows]

    def add_conversation_message(
        self,
        conversation_id: str,
        *,
        role: str,
        content: str,
    ) -> ConversationMessage:
        clean_role = role.strip().lower()
        clean_content = content.strip()

        if clean_role not in {"user", "assistant", "system"}:
            raise ValueError("Unsupported conversation message role.")

        if not clean_content:
            raise ValueError("Conversation message cannot be empty.")

        created_at = datetime.now(timezone.utc)

        with self._connect() as connection:
            cursor = connection.execute(
                """
                INSERT INTO conversation_messages
                    (conversation_id, role, content, created_at)
                VALUES
                    (?, ?, ?, ?)
                """,
                (
                    conversation_id,
                    clean_role,
                    clean_content,
                    created_at.isoformat(),
                ),
            )
            connection.execute(
                """
                UPDATE conversations
                SET updated_at = ?
                WHERE id = ?
                """,
                (
                    created_at.isoformat(),
                    conversation_id,
                ),
            )

            message_id = int(cursor.lastrowid)

        return ConversationMessage(
            id=message_id,
            conversation_id=conversation_id,
            role=clean_role,
            content=clean_content,
            created_at=created_at,
        )

    def conversation_messages(
        self,
        conversation_id: str,
    ) -> list[ConversationMessage]:
        with self._connect() as connection:
            rows = connection.execute(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY id ASC
                """,
                (conversation_id,),
            ).fetchall()

        return [self._conversation_message_from_row(row) for row in rows]

    @staticmethod
    def _memory_from_row(row: sqlite3.Row) -> Memory:
        try:
            memory_type = MemoryType(row["type"])
        except ValueError:
            memory_type = MemoryType.DAILY

        return Memory(
            id=row["id"],
            content=row["content"],
            memory_type=memory_type,
            importance=float(row["importance"] or 0.5),
            created_at=MemoryStore._parse_datetime(row["created_at"]),
        )

    @staticmethod
    def _conversation_from_row(row: sqlite3.Row) -> Conversation:
        return Conversation(
            id=row["id"],
            user_id=row["user_id"],
            title=row["title"],
            created_at=MemoryStore._parse_datetime(row["created_at"]),
            updated_at=MemoryStore._parse_datetime(row["updated_at"]),
        )

    @staticmethod
    def _conversation_message_from_row(
        row: sqlite3.Row,
    ) -> ConversationMessage:
        return ConversationMessage(
            id=int(row["id"]),
            conversation_id=row["conversation_id"],
            role=row["role"],
            content=row["content"],
            created_at=MemoryStore._parse_datetime(row["created_at"]),
        )

    @staticmethod
    def _parse_datetime(value: str | None) -> datetime:
        if not value:
            return datetime.now(timezone.utc)

        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))

        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)

        return parsed
