"""Understanding what a person means without commands.

Deterministic and instant: it recognises the things a user naturally says
about themselves ("my name is...", "I live in...", "remember that...") and
the natural questions about them ("what's my name?"). Nothing here calls the
model, so it costs no time on a small CPU and never invents a memory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.memory import Memory, MemoryType

_QUESTION_START = re.compile(
    r"^\s*(?:what|who|where|when|why|how|which|can|could|would|should|do|does|"
    r"did|is|are|was|were|will|have|has)\b",
    re.IGNORECASE,
)

_END = r"(?=[.,!?;]|\s+(?:and|but|because|so|though|however|located|which|that)\b|$)"
_NAME_STOP = {
    "not", "a", "an", "the", "going", "still", "so", "just", "very", "really",
    "sorry", "fine", "good", "here", "tired", "busy", "bored", "ok", "okay",
}
_PROFESSIONS = (
    "developer|programmer|engineer|student|teacher|lecturer|nurse|doctor|"
    "designer|writer|lawyer|farmer|accountant|musician|artist|manager|"
    "entrepreneur|founder|researcher|scientist|pilot|chef|trader|banker"
)
_OBJECT_STOP = {
    "that", "this", "it", "you", "what", "how", "when", "where", "why",
    "who", "the way", "your", "those", "these", "them", "to be honest",
    # "I love in Calabar" is a typo for "live", not a preference.
    "in", "at", "on", "from", "of", "for", "with", "by",
}
_LIKE_FORMS = {
    "like": "likes", "love": "loves", "enjoy": "enjoys", "prefer": "prefers",
    "hate": "hates", "dislike": "dislikes", "can't stand": "can't stand",
    "am into": "is into",
}

_NAME = re.compile(
    r"(?i:\b(?:my name is|my name's|call me|i am called|i'm called|"
    r"you can call me)\s+)([A-Za-z][A-Za-z'\-]{1,24})(?:\s+([A-Z][A-Za-z'\-]{1,24}))?"
)
_LIVES = re.compile(
    r"\bi(?:'m| am)?\s*(live in|stay in|come from|am from|'m from|am based in|'m based in|"
    r"was born in)\s+([A-Za-z][A-Za-z .'\-]{1,40}?)" + _END,
    re.IGNORECASE,
)
_WORKS = re.compile(
    r"\bi(?:'m| am)?\s*(?:work|am working|'m working|working)\s+(as|at|for|on)\s+"
    r"(.{2,60}?)" + _END,
    re.IGNORECASE,
)
_PROFESSION = re.compile(
    rf"\bi(?:'m| am)\s+an?\s+((?:\w+\s+){{0,2}}(?:{_PROFESSIONS}))\b",
    re.IGNORECASE,
)
_PREFERENCE = re.compile(
    r"\bi\s+(?:really\s+|absolutely\s+|do\s+)?(like|love|enjoy|prefer|hate|"
    r"dislike|can't stand|am into)\s+(.{2,60}?)" + _END,
    re.IGNORECASE,
)
_REMEMBER = re.compile(
    r"^\s*(?:please\s+)?(?:remember|don't forget|do not forget|keep in mind|"
    r"make a note|note)(?:\s+that)?\s*[:,]?\s+(.{3,300}?)\s*$",
    re.IGNORECASE | re.DOTALL,
)

_RECALL = (
    ("name", r"\b(?:what(?:'s| is)|do you (?:know|remember)) my name\b|\bwho am i\b"),
    ("location", r"\bwhere (?:do i live|am i from|do i stay)\b"),
    ("work", r"\bwhat do i do\b|\bwhere do i work\b|\bwhat(?:'s| is) my (?:job|work|profession)\b"),
    ("likes", r"\bwhat do i (?:like|love|enjoy|prefer|hate|dislike)\b|\bwhat(?:'s| is) my favou?rite\b"),
    (
        "about",
        r"\bwhat do you (?:know|remember) about me\b|\btell me (?:what you know )?about me\b|"
        r"\bwhat have i told you\b|\bwhat do you know about me\b",
    ),
)


@dataclass(frozen=True)
class MemoryNote:
    content: str
    memory_type: MemoryType
    importance: float
    kind: str  # name | place | work | preference | note
    detail: str = ""


def _clean(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip(" .,!?;:\"'")


def _titled(text: str) -> str:
    return " ".join(w.capitalize() if w.islower() else w for w in text.split(" "))


def _tidy_place(raw: str, rest: str) -> str:
    """'calabar' (+ ' located in Nigeria') -> 'Calabar, Nigeria'."""
    place = _titled(_clean(raw))
    extra = re.match(
        r"\s+(?:located\s+)?in\s+([A-Za-z][A-Za-z .'\-]{1,30}?)\s*(?:[.,!?;]|$)",
        rest,
    )

    if extra:
        place += ", " + _titled(_clean(extra.group(1)))

    return place


def _sentence(value: str) -> str:
    value = value.strip()
    return value[0].upper() + value[1:] if value else value


def extract_memory_notes(text: str) -> list[MemoryNote]:
    """Facts the user volunteered about themselves (never inferred)."""
    notes: list[MemoryNote] = []

    explicit = _REMEMBER.match(text)
    if explicit:
        body = _clean(explicit.group(1))
        if body:
            return [
                MemoryNote(
                    _sentence(body) + ".",
                    MemoryType.FACT,
                    0.8,
                    "note",
                    body,
                )
            ]

    match = _NAME.search(text)
    if match and match.group(1).lower() not in _NAME_STOP:
        first = match.group(1)
        first = first.capitalize() if first.islower() else first
        name = f"{first} {match.group(2)}" if match.group(2) else first
        notes.append(
            MemoryNote(
                f"The user's name is {name}.", MemoryType.PROFILE, 0.9, "name", name
            )
        )

    match = _LIVES.search(text)
    if match:
        place = _tidy_place(match.group(2), text[match.end():])
        phrase = {
            "live in": "lives in", "stay in": "lives in",
            "come from": "is from", "am from": "is from", "'m from": "is from",
            "am based in": "is based in", "'m based in": "is based in",
            "was born in": "was born in",
        }.get(match.group(1).lower(), "lives in")
        if place:
            notes.append(
                MemoryNote(
                    f"The user {phrase} {place}.", MemoryType.FACT, 0.7, "place", place
                )
            )

    match = _WORKS.search(text)
    if match:
        where = _clean(match.group(2))
        if where:
            notes.append(
                MemoryNote(
                    f"The user works {match.group(1).lower()} {where}.",
                    MemoryType.FACT, 0.7, "work", where,
                )
            )
    else:
        match = _PROFESSION.search(text)
        if match:
            role = _clean(match.group(1))
            notes.append(
                MemoryNote(
                    f"The user is a {role}.", MemoryType.FACT, 0.7, "work", role
                )
            )

    match = _PREFERENCE.search(text)
    if match:
        thing = _clean(match.group(2))
        verb = _LIKE_FORMS.get(match.group(1).lower(), match.group(1).lower())
        if thing and thing.lower() not in _OBJECT_STOP and "?" not in thing:
            first_word = thing.split()[0].lower()
            if first_word not in _OBJECT_STOP:
                notes.append(
                    MemoryNote(
                        f"The user {verb} {thing}.",
                        MemoryType.PREFERENCE, 0.6, "preference", thing,
                    )
                )

    return notes


def is_pure_disclosure(text: str) -> bool:
    """True when the message only tells SALLY something (no question)."""
    stripped = text.strip()

    return (
        "?" not in stripped
        and len(stripped) <= 240
        and not _QUESTION_START.match(stripped)
    )


def recall_kind(text: str) -> str | None:
    """Which question about the user this is, if any."""
    for kind, pattern in _RECALL:
        if re.search(pattern, text, re.IGNORECASE):
            return kind

    return None


def acknowledgement(notes: list[MemoryNote], replaced_name: bool = False) -> str:
    if len(notes) > 1:
        return "Got it, I'll remember all of that."

    note = notes[0]

    if note.kind == "name":
        if replaced_name:
            return f"Got it, I'll call you {note.detail} from now on."
        return f"Nice to meet you, {note.detail}! I'll remember that."

    if note.kind == "place":
        return f"Got it, {note.detail}. I'll keep that in mind."

    if note.kind == "preference":
        return "Noted, I'll remember that."

    if note.kind == "work":
        return "Thanks for telling me. I'll remember that."

    return "Done, I'll remember that."


_RECALL_FILTER = {
    "name": ("name is",),
    "location": ("lives in", "is from", "is based in", "was born in"),
    "work": ("works ", "is a "),
    "likes": ("likes", "loves", "enjoys", "prefers", "hates", "dislikes", "into"),
}


def answer_recall(kind: str, memories: list[Memory]) -> str:
    """Answer a question about the user from stored memories only."""
    if kind == "about":
        if not memories:
            return (
                "I don't know much about you yet. Tell me about yourself and "
                "I'll remember it."
            )

        top = sorted(memories, key=lambda item: item.importance, reverse=True)[:8]
        lines = "\n".join(f"- {item.content[:160]}" for item in top)

        return f"Here's what I remember about you:\n{lines}"

    needles = _RECALL_FILTER.get(kind, ())
    found = [m for m in memories if any(n in m.content.lower() for n in needles)]

    if kind == "name":
        if found:
            name = found[0].content.rsplit(" is ", 1)[-1].rstrip(".")
            return f"Your name is {name}."
        return "I don't know your name yet. What should I call you?"

    if not found:
        return "I don't know that about you yet. Tell me and I'll remember it."

    return "From what you've told me:\n" + "\n".join(f"- {m.content}" for m in found[:5])
