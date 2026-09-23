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
        r"\b(?:multiply|times)\b.*\d+.*\d+",
        r"\b\d+\s+(?:times|plus|minus)\s+\d+",
        r"\b\d+\s+divided\s+by\s+\d+",
        r"\b(?:add|subtract|divide|multiply)\b.*\d+.*\d+",
        r"^\s*(?:add|subtract|multiply|divide)\s+(?:by\s+)?-?\d+(?:\.\d+)?\s*$",
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

    _UNIT_CONVERSION_PATTERNS = (
        r"\bconvert\b.*\b(?:to|into|in)\b",
        r"\b(?:km/h|mph|m/s|kg|mg|g|ms|millisecond|milliseconds|"
        r"meters?|kilometers?|miles?|feet|ft|seconds?|minutes?|hours?)\b"
        r".*\b(?:to|into|in)\b",
        r"\bhow many\b.*\b(?:are\s+)?in\b.*\b(?:miles?|meters?|"
        r"kilometers?|feet|seconds?|minutes?|hours?|milliseconds?|ms)\b",
    )

    _CONSTANT_PATTERNS = (
        r"\bspeed of light\b",
        r"\bgravitational constant\b",
        r"\bplanck constant\b",
        r"\bboltzmann constant\b",
        r"\bavogadro (?:constant|number)\b",
    )

    _SCIENCE_CALC_PATTERNS = (
        r"\bsqrt\s*\(",
        r"\bsin\s*\(",
        r"\bcos\s*\(",
        r"\btan\s*\(",
        r"\blog\s*\(",
        r"\bln\s*\(",
        r"\bscientific calculation\b",
        r"\bscientific notation\b",
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
                RouteType.AGENT,
                "planning",
                0.1,
                "Empty objective.",
            )

        if self._matches(self._CALCULATOR_PATTERNS, text):
            return Route(
                RouteType.TOOL,
                "calculator",
                0.95,
                "Detected an arithmetic/calculation request.",
            )

        if self._matches(self._DATETIME_PATTERNS, text):
            return Route(
                RouteType.TOOL,
                "datetime",
                0.95,
                "Detected a date/time request.",
            )

        if self._matches(self._UNIT_CONVERSION_PATTERNS, text):
            return Route(
                RouteType.TOOL,
                "unit_convert",
                0.97,
                "Detected a unit conversion request.",
            )

        if self._matches(self._CONSTANT_PATTERNS, text):
            return Route(
                RouteType.TOOL,
                "scientific_constant",
                0.97,
                "Detected a scientific constant request.",
            )

        if self._matches(self._SCIENCE_CALC_PATTERNS, text):
            return Route(
                RouteType.TOOL,
                "science_calculate",
                0.93,
                "Detected a scientific calculation request.",
            )

        if any(word in text for word in self._TESTING_PATTERNS):
            return Route(
                RouteType.AGENT,
                "testing",
                0.85,
                "Detected testing/review intent.",
            )

        if any(word in text for word in self._CODING_PATTERNS):
            return Route(
                RouteType.AGENT,
                "coding",
                0.85,
                "Detected software-development intent.",
            )

        if any(word in text for word in self._RESEARCH_PATTERNS):
            return Route(
                RouteType.AGENT,
                "research",
                0.8,
                "Detected research/investigation intent.",
            )

        return Route(
            RouteType.AGENT,
            "planning",
            0.5,
            "No specialized intent detected.",
        )

    @staticmethod
    def _matches(patterns: tuple[str, ...], text: str) -> bool:
        return any(re.search(pattern, text) for pattern in patterns)


def create_router() -> TaskRouter:
    return TaskRouter()
