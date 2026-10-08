"""The shape of one conversation turn: its budget and its trace.

A turn is: route -> execute (tool / memory / self) -> optionally let the model
phrase what happened -> finish. ``TurnTrace`` records each phase so every
answer can show what actually happened, and ``TurnBudget`` bounds the work.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TurnBudget:
    max_seconds: float
    max_model_calls: int = 2

    @classmethod
    def from_settings(cls, settings: Any) -> TurnBudget:
        return cls(max_seconds=float(settings.agent.timeout))


class TurnTrace:
    def __init__(self, budget: TurnBudget) -> None:
        self.budget = budget
        self._started = time.monotonic()
        self.phases: list[dict[str, Any]] = []
        self.route: dict[str, str] | None = None
        self.tool: dict[str, Any] | None = None
        self.event: dict[str, Any] | None = None
        self.model_calls = 0
        self.narrated = False
        self.notes: list[str] = []
        self._recent: list[Any] | None = None

    def elapsed(self) -> float:
        return time.monotonic() - self._started

    def remaining(self) -> float:
        return max(1.0, self.budget.max_seconds - self.elapsed())

    def can_call_model(self) -> bool:
        return (
            self.model_calls < self.budget.max_model_calls
            and self.elapsed() < self.budget.max_seconds
        )

    @contextmanager
    def phase(self, name: str):
        started = time.perf_counter()

        try:
            yield
        finally:
            self.phases.append(
                {"name": name, "ms": round((time.perf_counter() - started) * 1000, 1)}
            )

    def recent_memories(self, memory: Any) -> list[Any]:
        """One database read per turn, however many helpers ask."""
        if self._recent is None:
            self._recent = memory.recent(limit=50)

        return self._recent

    def forget_cache(self) -> None:
        self._recent = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "route": self.route,
            "phases": self.phases,
            "tool": self.tool,
            "event": self.event,
            "narrated": self.narrated,
            "model_calls": self.model_calls,
            "notes": self.notes,
            "total_ms": round(self.elapsed() * 1000, 1),
        }
