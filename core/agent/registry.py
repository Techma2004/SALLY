from __future__ import annotations

from .types import AgentSpec


class AgentRegistry:
    def __init__(self) -> None:
        self._agents: dict[str, AgentSpec] = {}

    def register(self, spec: AgentSpec) -> None:
        key = spec.name.strip().lower()

        if not key:
            raise ValueError("Agent name cannot be empty.")

        if spec.max_steps is not None and spec.max_steps < 1:
            raise ValueError("max_steps must be at least 1.")

        if spec.max_tokens is not None and spec.max_tokens < 1:
            raise ValueError("max_tokens must be at least 1.")

        self._agents[key] = spec

    def get(self, name: str) -> AgentSpec:
        key = name.strip().lower()

        try:
            return self._agents[key]
        except KeyError:
            raise KeyError(f"Unknown SALLY agent: {name}") from None

    def all(self) -> list[AgentSpec]:
        return list(self._agents.values())

    def names(self) -> list[str]:
        return list(self._agents.keys())

    def describe(self) -> str:
        return "\n".join(
            f"- {agent.name}: {agent.description}"
            for agent in self._agents.values()
        )


def create_default_registry() -> AgentRegistry:
    registry = AgentRegistry()

    registry.register(
        AgentSpec(
            name="research",
            role="Research Agent",
            description="Breaks down questions, compares information, and produces structured research.",
            system_prompt=(
                "You are SALLY's Research Agent. "
                "Analyze the objective carefully, separate facts from assumptions, "
                "and return concise structured findings. "
                "Do not pretend to have browsed the internet."
            ),
        )
    )

    registry.register(
        AgentSpec(
            name="coding",
            role="Coding Agent",
            description="Designs, explains, reviews, and troubleshoots software.",
            system_prompt=(
                "You are SALLY's Coding Agent. "
                "Think like a careful software engineer. "
                "Prefer simple, maintainable solutions and explain important tradeoffs."
            ),
        )
    )

    registry.register(
        AgentSpec(
            name="planning",
            role="Planning Agent",
            description="Turns broad objectives into practical implementation plans.",
            system_prompt=(
                "You are SALLY's Planning Agent. "
                "Turn objectives into ordered, realistic steps. "
                "Avoid unnecessary features and keep resource constraints in mind."
            ),
        )
    )

    registry.register(
        AgentSpec(
            name="testing",
            role="Testing Agent",
            description="Analyzes implementations for bugs, edge cases, and test coverage.",
            system_prompt=(
                "You are SALLY's Testing Agent. "
                "Look for concrete bugs, edge cases, regressions, and missing tests. "
                "Prioritize actionable findings."
            ),
        )
    )

    return registry
