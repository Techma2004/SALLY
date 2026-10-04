"""Optional, experimental: let the local model classify a message.

Off by default (INTENT_LLM=false) because it costs one extra short model call
per message that the rules did not already understand. Output is constrained
to a fixed set of labels, and anything unexpected falls back to plain chat.
"""

from __future__ import annotations

import json

from core.agent.router import Route, RouteType

LABELS: dict[str, tuple[RouteType, str]] = {
    "time": (RouteType.TOOL, "datetime"),
    "math": (RouteType.TOOL, "calculator"),
    "convert": (RouteType.TOOL, "unit_convert"),
    "constant": (RouteType.TOOL, "scientific_constant"),
    "machine": (RouteType.TOOL, "machine_status"),
    "plan": (RouteType.AGENT, "planning"),
    "code": (RouteType.AGENT, "coding"),
    "research": (RouteType.AGENT, "research"),
}

_PROMPT = (
    "Classify the user's message. Reply with JSON only: "
    '{"intent": "<label>"}. Labels: chat (normal conversation, opinions, '
    "greetings, anything else), time (current time or date), math "
    "(arithmetic), convert (unit conversion), constant (physics constant), "
    "machine (this computer's CPU, RAM, disk, battery), plan (make a plan), "
    "code (programming help), research (compare or investigate)."
)


def classify(text: str, chat=None) -> Route | None:
    """Return a Route for a non-chat intent, or None (meaning: just chat)."""
    if chat is None:
        from core import llm

        chat = llm.chat

    try:
        raw = chat(
            [
                {"role": "system", "content": _PROMPT},
                {"role": "user", "content": text[:400]},
            ],
            max_tokens=16,
            temperature=0.0,
            json_mode=True,
            timeout_s=20,
        )
        label = str(json.loads(raw).get("intent", "")).strip().lower()
    except Exception:
        return None

    target = LABELS.get(label)

    if target is None:
        return None

    return Route(target[0], target[1], 0.6, "LLM intent classification.")
