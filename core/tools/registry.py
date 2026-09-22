from __future__ import annotations

from dataclasses import dataclass

from core.agent.tools import ToolRegistry


@dataclass(frozen=True)
class ToolInfo:
    name: str
    description: str
    safety: str = "safe"


def create_tool_registry() -> ToolRegistry:
    registry = ToolRegistry()

    registry.register(
        "calculator",
        _calculator,
    )

    registry.register(
        "datetime",
        _datetime,
    )

    return registry


def _calculator(expression: str) -> str:
    from .calculator import calculate
    return calculate(expression)


def _datetime() -> str:
    from .datetime_tool import current_datetime
    return current_datetime()


TOOL_INFO = (
    ToolInfo(
        name="calculator",
        description="Safely evaluates basic arithmetic expressions.",
        safety="safe",
    ),
    ToolInfo(
        name="datetime",
        description="Returns the current local date and time.",
        safety="safe",
    ),
)
