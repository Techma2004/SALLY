from __future__ import annotations

from dataclasses import dataclass

from core.agent.tools import ToolRegistry
from core.science import (
    convert_units,
    scientific_calculate,
    scientific_constant,
)


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
        description="Safely evaluates basic arithmetic expressions.",
    )

    registry.register(
        "datetime",
        _datetime,
        description="Returns the current local date and time.",
    )

    registry.register(
        "science_calculate",
        scientific_calculate,
        description="Performs verified scientific calculations.",
        requires_inference=True,
    )

    registry.register(
        "unit_convert",
        convert_units,
        description="Converts compatible scientific units.",
        requires_inference=True,
    )

    registry.register(
        "scientific_constant",
        scientific_constant,
        description="Returns verified scientific constants.",
        requires_inference=True,
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
    ),
    ToolInfo(
        name="datetime",
        description="Returns the current local date and time.",
    ),
    ToolInfo(
        name="science_calculate",
        description="Performs verified scientific calculations.",
    ),
    ToolInfo(
        name="unit_convert",
        description="Converts compatible scientific units.",
    ),
    ToolInfo(
        name="scientific_constant",
        description="Returns verified scientific constants.",
    ),
)
