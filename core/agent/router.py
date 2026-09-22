from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re


class RouteType(str, Enum):
    TOOL = "tool"
    AGENT = "agent"


@dataclass(frozen=True)
class Route:
    route_type: RouteType
    target: str
    confidence: float
    reason: str


class TaskRouter:
    """Lightweight deterministic router for obvious SALLY tasks."""

    _CALCULATOR_PATTERNS = (
        r"\bcalculate\b",
        r"\bcompute\b",
        r"\bsolve\b.*[0-9]",
        r"\bwhat is\b.*[0-9].*(?:\+|-|\*|/|%|\^|times|plus|minus|divided)",
        r"\bhow much is\b.*[0-9]",
        r"\b\d+\s*(?:\+|-|\*|/|%|\^)\s*\d+",
    )

    _DATETIME_PATTERNS = (
        r"\bwhat time\b",
        r"\bcurrent time\b",
        r"\btime is it\b",
        r"\bwhat date\b",
        r"\btoday'?s date\b",
        r"\bcurrent date\b",
        r"\bdate today\b",
    )

    _CODING_PATTERNS = (
        "code",
        "python",
        "javascript",
        "typescript",
        "react",
        "bug",
        "debug",
        "function",
        "api",
        "database",
        "program",
        "script",
    )

    _TESTING_PATTERNS = (
        "test",
        "testing",
        "review",
        "failure",
        "failing",
        "edge case",
        "coverage",
    )

    _RESEARCH_PATTERNS = (
        "research",
        "investigate",
        "compare",
        "look into",
        "find out",
    )

    def route(self, objective: str) -> Route:
        text = objective.strip().lower()

        if not text:
            return Route(
                route_type=RouteType.AGENT,
                target="planning",
                confidence=0.1,
                reason="Empty objective.",
            )

        if self._matches(self._CALCULATOR_PATTERNS, text):
            return Route(
                route_type=RouteType.TOOL,
                target="calculator",
                confidence=0.95,
                reason="Detected an arithmetic/calculation request.",
            )

        if self._matches(self._DATETIME_PATTERNS, text):
            return Route(
                route_type=RouteType.TOOL,
                target="datetime",
                confidence=0.95,
                reason="Detected a date/time request.",
            )

        if any(word in text for word in self._TESTING_PATTERNS):
            return Route(
                route_type=RouteType.AGENT,
                target="testing",
                confidence=0.85,
                reason="Detected testing/review intent.",
            )

        if any(word in text for word in self._CODING_PATTERNS):
            return Route(
                route_type=RouteType.AGENT,
                target="coding",
                confidence=0.85,
                reason="Detected software-development intent.",
            )

        if any(word in text for word in self._RESEARCH_PATTERNS):
            return Route(
                route_type=RouteType.AGENT,
                target="research",
                confidence=0.8,
                reason="Detected research/investigation intent.",
            )

        return Route(
            route_type=RouteType.AGENT,
            target="planning",
            confidence=0.5,
            reason="No specialized intent detected.",
        )

    @staticmethod
    def _matches(patterns: tuple[str, ...], text: str) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)


def create_router() -> TaskRouter:
    return TaskRouter()
