from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class AgentAction:
    """
    Model-produced agent action.

    The action name is intentionally dynamic. SALLY does not maintain
    a hardcoded vocabulary of possible action names.
    """
    action: str
    tool: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)
    answer: str | None = None


def _extract_json(text: str) -> dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        lines = cleaned.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end <= start:
            raise ValueError(
                "Agent response did not contain a valid JSON object."
            )
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid agent JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Agent response must be a JSON object.")

    return data


def parse_action(text: str) -> AgentAction:
    data = _extract_json(text)

    action = str(data.get("action", "")).strip()
    tool = str(data.get("tool", "")).strip().lower()
    arguments = data.get("arguments", {})
    answer = str(data.get("answer", "")).strip()

    if not action:
        raise ValueError("Agent action requires a non-empty 'action'.")

    if not isinstance(arguments, dict):
        raise ValueError("Agent action 'arguments' must be an object.")

    if not tool and not answer:
        return AgentAction(
            action=action,
            arguments=arguments,
        )

    return AgentAction(
        action=action,
        tool=tool or None,
        arguments=arguments,
        answer=answer or None,
    )
