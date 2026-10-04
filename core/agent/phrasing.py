"""Turn exact tool results into natural sentences, deterministically.

The numbers always come from the tool, never from the language model, so the
answer is instant and cannot be embellished or mis-stated.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any


def number(value: float | int) -> str:
    if isinstance(value, bool):
        return str(value)

    if isinstance(value, int):
        return f"{value:,}"

    if value == int(value) and abs(value) < 1e15:
        return f"{int(value):,}"

    if 1e-3 <= abs(value) < 1e15:
        return f"{value:,.6g}"

    return f"{value:.6g}"


def _pretty_expression(expression: str) -> str | None:
    """Readable form of the expression, or None if it is too tangled."""
    text = expression

    # The calculator wraps every operand: "(25) * (40)" -> "25 * 40".
    previous = None
    while previous != text:
        previous = text
        text = re.sub(r"\((-?\d+(?:\.\d+)?)\)", r"\1", text)

    if "(" in text and text.count("(") > 1:
        return None

    text = text.replace("**", "^").replace("*", " × ").replace("/", " ÷ ")
    text = re.sub(r"(?<=[\d)])\s*([+\-])\s*(?=[\d(])", r" \1 ", text)

    return re.sub(r"\s+", " ", text).strip()


def _calculator(arguments: dict[str, Any], output: Any) -> str | None:
    expression = str(arguments.get("expression", "")).strip()

    try:
        shown = number(float(output)) if "." in str(output) else number(int(output))
    except (TypeError, ValueError):
        return None

    pretty = _pretty_expression(expression) if expression else None

    if pretty is None:
        return f"That comes to {shown}."

    return f"{pretty} = {shown}"


def _datetime(objective: str, output: Any) -> str | None:
    match = re.match(r"(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?\((UTC[+-]\d{4})\)", str(output))

    if not match:
        return None

    moment = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M:%S")
    day = f"{moment:%A}, {moment:%B} {moment.day}, {moment.year}"
    clock = f"{moment.hour % 12 or 12}:{moment:%M} {'AM' if moment.hour < 12 else 'PM'}"

    wants_time = re.search(r"\btime\b|\bclock\b|\bo'clock\b", objective, re.IGNORECASE)
    wants_date = re.search(r"\bdate\b|\bday\b|\btoday\b|\bmonth\b|\byear\b", objective, re.IGNORECASE)

    if wants_date and not wants_time:
        return f"Today is {day}."

    return f"It's {clock} on {day} ({match.group(2)})."


def _unit(output: dict[str, Any]) -> str | None:
    try:
        source, target = output["input"], output["output"]

        return (
            f"{number(source['value'])} {source['unit']} is "
            f"{number(target['value'])} {target['unit']}."
        )
    except (KeyError, TypeError):
        return None


def _constant(output: dict[str, Any]) -> str | None:
    try:
        description = str(output["description"]).rstrip(".")
        value = output["value"]
        shown = number(value) if isinstance(value, int) else f"{value:g}"

        return f"{description}: {shown} {output['unit']} ({output['symbol']})."
    except (KeyError, TypeError, ValueError):
        return None


def _science(output: dict[str, Any]) -> str | None:
    try:
        return f"{output['expression']} = {number(output['result'])}"
    except (KeyError, TypeError):
        return None


def phrase_tool_answer(
    tool: str,
    objective: str,
    arguments: dict[str, Any],
    output: Any,
) -> str | None:
    """A natural sentence for a tool result, or None to use the default."""
    if tool == "calculator":
        return _calculator(arguments, output)

    if tool == "datetime":
        return _datetime(objective, output)

    if tool == "machine_status" and isinstance(output, dict):
        from core.native import describe_machine

        return describe_machine(output)

    if not isinstance(output, dict):
        return None

    if tool == "unit_convert":
        return _unit(output)

    if tool == "scientific_constant":
        return _constant(output)

    if tool == "science_calculate":
        return _science(output)

    return None
