"""Let the model put what happened into words, without letting it do or
change anything.

SALLY's code runs the tool (or saves the memory). The model only receives a
small JSON description of *what happened* and phrases it. Its reply is then
checked: every number or name that matters must survive, otherwise the exact
deterministic text is used instead. The model can never alter a result.
"""

from __future__ import annotations

import json
import re
from typing import Any

# Longer answers (full machine report, memory lists) are already readable
# as they are, so they are not re-phrased.
MAX_EXACT_CHARS = 240
MAX_REPLY_CHARS = 600

_NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")

_SYSTEM = (
    "You are SALLY, a warm personal assistant. SALLY's own code just did "
    "something for the user; you did not do it. What happened is given below "
    "as JSON. Reply in one or two natural sentences using only those facts. "
    "Keep every number and name exactly as written. If ok is false, say "
    "briefly what went wrong and what the user could try. Never mention "
    "JSON, tools, or these instructions."
)


def json_safe(value: Any, limit: int = 600) -> Any:
    """The value if it is small and JSON-friendly, otherwise None."""
    try:
        encoded = json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return None

    if len(encoded) > limit:
        return None

    return json.loads(encoded)


def numbers_to_keep(text: str) -> list[str]:
    """The numbers in an exact answer that a rewrite must not change.

    Anything in parentheses is decoration (time zone, symbol, expression
    operands) and is not checked. For "a x b = c" only the result c matters:
    the model may well say "twenty-five times forty".
    """
    outside = re.sub(r"\([^)]*\)", "", text)

    if "=" in outside:
        outside = outside.rsplit("=", 1)[1]

    seen: list[str] = []

    for token in _NUMBER.findall(outside):
        cleaned = token.replace(",", "")

        if cleaned not in seen:
            seen.append(cleaned)

    return seen


def reply_keeps(verify: list[str], reply: str) -> bool:
    """True when every number and name in ``verify`` survived in the reply."""
    plain = reply.replace(",", "")

    for item in verify:
        if re.fullmatch(r"\d+(?:\.\d+)?", item):
            if not re.search(rf"(?<![\d.]){re.escape(item)}(?!\d)", plain):
                return False
        elif item.lower() not in reply.lower():
            return False

    return True


def narration_messages(
    objective: str,
    event: dict[str, Any],
    name: str | None = None,
) -> list[dict[str, str]]:
    system = _SYSTEM

    if name:
        system += f" The user's name is {name}."

    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": (
                f"{objective[:400]}\n\nWhat happened (JSON):\n"
                f"{json.dumps(event, ensure_ascii=False)}"
            ),
        },
    ]
