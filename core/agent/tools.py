from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .types import ToolRequest, ToolResult


ToolFunction = Callable[..., Any]


class ToolRegistry:
    """Central registry for tools available to SALLY agents."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}

    def register(self, name: str, function: ToolFunction) -> None:
        name = name.strip().lower()

        if not name:
            raise ValueError("Tool name cannot be empty.")

        if name in self._tools:
            raise ValueError(f"Tool already registered: {name}")

        self._tools[name] = function

    def has(self, name: str) -> bool:
        return name.strip().lower() in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def execute(self, request: ToolRequest) -> ToolResult:
        name = request.name.strip().lower()

        if name not in self._tools:
            return ToolResult(
                name=name,
                success=False,
                error=f"Unknown tool: {name}",
            )

        try:
            result = self._tools[name](**request.arguments)

            return ToolResult(
                name=name,
                success=True,
                output=result,
            )

        except Exception as exc:
            return ToolResult(
                name=name,
                success=False,
                error=str(exc),
            )
