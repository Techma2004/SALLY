from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .types import ToolRequest, ToolResult


ToolFunction = Callable[..., Any]


@dataclass(frozen=True)
class ToolMetadata:
    name: str
    description: str = ""
    safety: str = "safe"
    requires_inference: bool = False


class ToolRegistry:
    """Central registry for tools available to SALLY agents."""

    def __init__(self) -> None:
        self._tools: dict[str, ToolFunction] = {}
        self._metadata: dict[str, ToolMetadata] = {}

    def register(
        self,
        name: str,
        function: ToolFunction,
        *,
        description: str = "",
        safety: str = "safe",
        requires_inference: bool = False,
    ) -> None:
        key = name.strip().lower()

        if not key:
            raise ValueError("Tool name cannot be empty.")

        if key in self._tools:
            raise ValueError(f"Tool already registered: {key}")

        self._tools[key] = function
        self._metadata[key] = ToolMetadata(
            name=key,
            description=description,
            safety=safety,
            requires_inference=requires_inference,
        )

    def has(self, name: str) -> bool:
        return name.strip().lower() in self._tools

    def names(self) -> list[str]:
        return list(self._tools.keys())

    def metadata(self, name: str) -> ToolMetadata:
        key = name.strip().lower()

        if key not in self._metadata:
            raise KeyError(f"Unknown tool: {key}")

        return self._metadata[key]

    def requires_inference(self, name: str) -> bool:
        return self.metadata(name).requires_inference

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
