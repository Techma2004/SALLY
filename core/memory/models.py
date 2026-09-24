from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum


class MemoryType(str, Enum):
    FACT = "fact"
    PREFERENCE = "preference"
    EPISODE = "episode"
    PROFILE = "profile"
    DAILY = "daily"
    SESSION = "session"


@dataclass(frozen=True)
class Memory:
    id: str
    content: str
    memory_type: MemoryType
    importance: float
    created_at: datetime


@dataclass(frozen=True)
class UserModelEntry:
    key: str
    value: str
    confidence: float
    updated_at: datetime


@dataclass(frozen=True)
class Session:
    id: str
    started_at: datetime
    summary: str


@dataclass(frozen=True)
class Conversation:
    id: str
    user_id: str
    title: str
    created_at: datetime
    updated_at: datetime


@dataclass(frozen=True)
class ConversationMessage:
    id: int
    conversation_id: str
    role: str
    content: str
    created_at: datetime
