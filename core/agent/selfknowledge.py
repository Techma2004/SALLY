"""Questions about SALLY herself, and small talk, answered instantly.

Deterministic on purpose: a small local model invents capabilities ("I can
schedule appointments") and takes seconds to say "You're welcome". These
answers are true, immediate, and cost no model time.
"""

from __future__ import annotations

import re
import zlib
from pathlib import Path

_SELF = (
    (
        "capabilities",
        r"\bwhat (?:can|do) you (?:do|help)\b|\bwhat are you (?:able|capable)\b|"
        r"\bhow can you help\b|\bwhat are your (?:features|abilities|capabilities|skills)\b|"
        r"\bhow (?:do|can) i use you\b|^\s*(?:sally[ ,]+)?help\s*[?!.]*$|"
        r"\bwhat can i (?:ask|do with) you\b",
    ),
    (
        "version",
        r"\bwhat(?:'s| is) your version\b|\bwhich version\b|\bwhat version\b",
    ),
    (
        "model",
        r"\bwhat model\b|\bwhich model\b|\bwhat (?:llm|ai) are you\b|"
        r"\bwhat are you running on\b",
    ),
    (
        "identity",
        r"\bwhat(?:'s| is) your name\b|\bwho are you\b|\bintroduce yourself\b|"
        r"\btell me about yourself\b|\bwhat are you\s*[?!.]*$",
    ),
)

_SMALLTALK = (
    (
        "greeting",
        r"^(?:hi|hello|hey|hiya|yo|howdy|good (?:morning|afternoon|evening))"
        r"(?:[ ,]+(?:sally|there|again))*\s*[!.?]*$",
    ),
    (
        "thanks",
        r"^(?:ok(?:ay)?[ ,]+)?(?:thanks|thank you|thx|cheers|much appreciated)"
        r"(?:[ ,]+(?:sally|so much|a lot|very much))*\s*[!.]*$",
    ),
    (
        "farewell",
        r"^(?:bye|goodbye|good night|goodnight|cya|see you|see ya|"
        r"i'?ll see you|i will see you|talk to you|talk later|catch you)"
        r"(?:[ ,]+[a-z']+){0,3}\s*[!.]*$",
    ),
)

_REPLIES = {
    "greeting": (
        "Hey{name}! What's up?",
        "Hi{name}! How can I help?",
        "Hello{name}! What are we doing today?",
    ),
    "thanks": (
        "You're welcome!",
        "Anytime{name}.",
        "Happy to help!",
    ),
    "farewell": (
        "See you{name}! I'll be here.",
        "Bye{name}, take care!",
        "Talk soon{name}.",
    ),
}

_CAPABILITIES = (
    "Here's what I can do:\n"
    "- Exact answers: maths, the time and date, unit conversions and "
    "physics constants\n"
    "- This machine: CPU, memory, disk, battery and network\n"
    "- Memory: I remember what you tell me about yourself, like your name, "
    "where you live and what you like\n"
    "- Conversation, writing, planning, coding help and comparing options "
    "(I'm a small local model, so double-check anything important)\n\n"
    "Just ask in your own words. The Help page has examples you can click."
)


def self_kind(text: str) -> str | None:
    lowered = text.strip().lower()

    for kind, pattern in _SELF:
        if re.search(pattern, lowered):
            return kind

    return None


def smalltalk_kind(text: str) -> str | None:
    lowered = text.strip().lower()

    if len(lowered) > 48:
        return None

    for kind, pattern in _SMALLTALK:
        if re.match(pattern, lowered):
            return kind

    return None


def answer_self(kind: str, *, name: str | None = None, variant: int = 0) -> str:
    from core.config import settings
    from core.version import get_version

    if kind == "capabilities":
        return _CAPABILITIES

    if kind == "version":
        return f"I'm SALLY v{get_version()}."

    if kind == "model":
        model = Path(str(getattr(settings.llm, "model_path", ""))).name or "a local model"
        return f"I run locally on this machine, using {model}."

    if kind == "identity":
        return (
            "I'm SALLY, your personal assistant. I run privately on your own "
            "machine, so what we talk about stays here."
        )

    options = _REPLIES[kind]
    reply = options[variant % len(options)]

    return reply.format(name=f", {name}" if name else "")


def variant_for(text: str, turns: int) -> int:
    return (zlib.crc32(text.lower().encode()) + turns) % 997
