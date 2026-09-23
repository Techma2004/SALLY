from __future__ import annotations

from core.memory import MemoryManager

from .registry import AgentRegistry, create_default_registry
from .router import RouteType, TaskRouter, create_router
from .runtime import AgentRuntime
from .types import AgentResult, AgentStatus, ToolRequest


class Coordinator:
    """
    Top-level request coordinator.

    Routes deterministic tasks directly to tools and
    sends reasoning-oriented tasks to SALLY agents.

    Relevant memories are retrieved before agent execution
    and passed as context.
    """

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        runtime: AgentRuntime | None = None,
        router: TaskRouter | None = None,
        memory: MemoryManager | None = None,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.runtime = runtime or AgentRuntime()
        self.router = router or create_router()
        self.memory = memory or MemoryManager()

    def choose_agent(self, objective: str) -> str:
        route = self.router.route(objective)
        if route.route_type is RouteType.AGENT:
            return route.target
        return "planning"

    def _memory_context(self, objective: str) -> dict[str, str]:
        memories = self.memory.search(objective, limit=5)

        if not memories:
            return {}

        lines = [
            f"- [{memory.memory_type.value}] {memory.content}"
            for memory in memories
        ]

        return {
            "relevant_memories": "\n".join(lines),
        }

    def run(
        self,
        objective: str,
        *,
        context: dict | None = None,
    ) -> AgentResult:
        route = self.router.route(objective)

        if route.route_type is RouteType.TOOL:
            result = self.runtime.tools.execute(
                ToolRequest(
                    name=route.target,
                    arguments=self._tool_arguments(
                        route.target,
                        objective,
                    ),
                )
            )

            if result.success:
                output = str(result.output)
                return AgentResult(
                    task_id="tool-" + route.target,
                    agent_name=route.target,
                    status=AgentStatus.COMPLETE,
                    output=output,
                    steps=1,
                    history=[],
                )

            return AgentResult(
                task_id="tool-" + route.target,
                agent_name=route.target,
                status=AgentStatus.FAILED,
                output="",
                steps=1,
                history=[],
                error=result.error,
            )

        spec = self.registry.get(route.target)

        combined_context = dict(context or {})
        memory_context = self._memory_context(objective)

        if memory_context:
            combined_context.update(memory_context)

        return self.runtime.run(
            spec,
            objective,
            context=combined_context,
        )

    @staticmethod
    def _tool_arguments(tool_name: str, objective: str) -> dict:
        if tool_name == "calculator":
            expression = Coordinator._extract_expression(objective)
            return {"expression": expression}
        return {}

    @staticmethod
    def _extract_expression(objective: str) -> str:
        text = objective.strip()

        prefixes = (
            "calculate ",
            "compute ",
            "what is ",
            "how much is ",
            "solve ",
        )

        lowered = text.lower()

        for prefix in prefixes:
            if lowered.startswith(prefix):
                return text[len(prefix):].strip().rstrip("?.!")

        return text.rstrip("?.!")

    def describe_route(self, objective: str) -> str:
        route = self.router.route(objective)

        return (
            f"{route.route_type.value} → {route.target} "
            f"({route.confidence:.2f}): {route.reason}"
        )
