from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class ActionType(str, Enum):
    TOOL = "tool"
    FINAL = "final"


@dataclass(frozen=True)
class AgentAction:
    action: ActionType
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
            raise ValueError("Agent response did not contain a valid JSON object.")
        try:
            data = json.loads(cleaned[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid agent JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Agent response must be a JSON object.")

    return data


def parse_action(text: str) -> AgentAction:
    data = _extract_json(text)

    action = str(data.get("action", "")).strip().lower()

    if action == ActionType.FINAL.value:
        answer = str(data.get("answer", "")).strip()
        if not answer:
            raise ValueError("Final action requires a non-empty 'answer'.")
        return AgentAction(
            action=ActionType.FINAL,
            answer=answer,
        )

    if action == ActionType.TOOL.value:
        tool = str(data.get("tool", "")).strip().lower()
        arguments = data.get("arguments", {})

        if not tool:
            raise ValueError("Tool action requires a 'tool' name.")
        if not isinstance(arguments, dict):
            raise ValueError("Tool action 'arguments' must be an object.")

        return AgentAction(
            action=ActionType.TOOL,
            tool=tool,
            arguments=arguments,
        )

    raise ValueError(
        f"Unknown agent action: {action!r}. Expected 'tool' or 'final'."
    )
