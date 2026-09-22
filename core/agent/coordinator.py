from __future__ import annotations

from .registry import AgentRegistry, create_default_registry
from .runtime import AgentRuntime
from .types import AgentResult


class Coordinator:
    """
    SALLY's first coordinator.

    This version uses deterministic routing rather than asking the LLM
    to decide everything. LLM-based planning can be added later.
    """

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        runtime: AgentRuntime | None = None,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.runtime = runtime or AgentRuntime()

    def choose_agent(self, objective: str) -> str:
        text = objective.lower()

        coding_words = (
            "code",
            "python",
            "javascript",
            "react",
            "bug",
            "debug",
            "program",
            "function",
            "api",
            "database",
        )

        research_words = (
            "research",
            "compare",
            "investigate",
            "find out",
            "explain",
            "analyze",
        )

        testing_words = (
            "test",
            "testing",
            "review",
            "error",
            "failure",
            "edge case",
        )

        if any(word in text for word in coding_words):
            return "coding"

        if any(word in text for word in testing_words):
            return "testing"

        if any(word in text for word in research_words):
            return "research"

        return "planning"

    def run(self, objective: str, *, context: dict | None = None) -> AgentResult:
        agent_name = self.choose_agent(objective)
        spec = self.registry.get(agent_name)

        return self.runtime.run(
            spec,
            objective,
            context=context,
        )
