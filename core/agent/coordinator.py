from __future__ import annotations

import re

from core.agent.inference import InferenceEngine
from core.memory import MemoryManager
from core.tools import create_tool_registry
from core.tools.calculator import natural_expression

from .registry import AgentRegistry, create_default_registry
from .router import RouteType, TaskRouter, create_router
from .runtime import AgentRuntime
from .types import (
    AgentResult,
    AgentStatus,
    AgentStep,
    StepType,
    ToolRequest,
)


class Coordinator:
    """
    Top-level request coordinator.

    Deterministic tools handle exact operations.
    Tools marked for inference pass their verified result through
    the hidden InferenceEngine before returning to the user.
    """

    def __init__(
        self,
        registry: AgentRegistry | None = None,
        runtime: AgentRuntime | None = None,
        router: TaskRouter | None = None,
        memory: MemoryManager | None = None,
        inference: InferenceEngine | None = None,
    ) -> None:
        self.registry = registry or create_default_registry()
        self.runtime = runtime or AgentRuntime(
            tools=create_tool_registry()
        )
        self.router = router or create_router()
        self.memory = memory or MemoryManager()
        self.inference = inference or InferenceEngine()

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

        return {"relevant_memories": "\n".join(lines)}

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
                        context,
                    ),
                )
            )

            if not result.success:
                return AgentResult(
                    task_id="tool-" + route.target,
                    agent_name=route.target,
                    status=AgentStatus.FAILED,
                    output="",
                    steps=1,
                    history=[],
                    error=result.error,
                )

            tool_step = AgentStep(
                number=1,
                step_type=StepType.TOOL,
                tool_request=ToolRequest(
                    name=route.target,
                    arguments=self._tool_arguments(
                        route.target,
                        objective,
                        context,
                    ),
                ),
            )

            observation = AgentStep(
                number=1,
                step_type=StepType.OBSERVE,
                tool_result=result,
                output=str(result.output),
            )

            if self.runtime.tools.requires_inference(route.target):
                combined_context = dict(context or {})
                memory_context = self._memory_context(objective)

                if memory_context:
                    combined_context.update(memory_context)

                try:
                    answer = self.inference.compose(
                        objective,
                        tool_name=route.target,
                        evidence=result.output,
                        context=combined_context,
                    )

                    final_step = AgentStep(
                        number=2,
                        step_type=StepType.FINAL,
                        output=answer,
                    )

                    return AgentResult(
                        task_id="inference-" + route.target,
                        agent_name="inference",
                        status=AgentStatus.COMPLETE,
                        output=answer,
                        steps=2,
                        history=[
                            tool_step,
                            observation,
                            final_step,
                        ],
                    )
                except Exception as exc:
                    return AgentResult(
                        task_id="tool-" + route.target,
                        agent_name=route.target,
                        status=AgentStatus.FAILED,
                        output="",
                        steps=1,
                        history=[
                            tool_step,
                            observation,
                        ],
                        error=f"Inference failed: {exc}",
                    )

            return AgentResult(
                task_id="tool-" + route.target,
                agent_name=route.target,
                status=AgentStatus.COMPLETE,
                output=str(result.output),
                steps=1,
                history=[
                    tool_step,
                    observation,
                ],
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
    def _tool_arguments(
        tool_name: str,
        objective: str,
        context: dict | None = None,
    ) -> dict:
        text = objective.strip().rstrip("?.!")
        lowered = text.lower()
        context = context or {}

        if tool_name == "calculator":
            expression = natural_expression(
                text,
                base_value=context.get("last_result"),
            )

            if expression is None:
                expression = Coordinator._extract_expression(text)

            return {"expression": expression}

        if tool_name == "science_calculate":
            for prefix in (
                "calculate ",
                "compute ",
                "evaluate ",
            ):
                if lowered.startswith(prefix):
                    return {
                        "expression": text[len(prefix):].strip()
                    }

            return {"expression": text}

        if tool_name == "unit_convert":
            import re

            match = re.search(
                r"how many\s+([a-zA-Z°/]+)\s+(?:are\s+)?in\s+"
                r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z°/]+)",
                text,
                re.IGNORECASE,
            )

            if match:
                target_unit, value, source_unit = match.groups()
                return {
                    "value": float(value),
                    "from_unit": source_unit,
                    "to_unit": target_unit,
                }

            match = re.search(
                r"(-?\d+(?:\.\d+)?)\s*([a-zA-Z°/]+)\s+"
                r"(?:to|into|in)\s+([a-zA-Z°/]+)",
                text,
                re.IGNORECASE,
            )

            if not match:
                match = re.search(
                    r"convert\s+(-?\d+(?:\.\d+)?)\s*"
                    r"([a-zA-Z°/]+)\s*(?:to|into|in)\s*"
                    r"([a-zA-Z°/]+)",
                    text,
                    re.IGNORECASE,
                )

            if not match:
                raise ValueError(
                    "Could not determine the value and units to convert."
                )

            return {
                "value": float(match.group(1)),
                "from_unit": match.group(2),
                "to_unit": match.group(3),
            }

        if tool_name == "scientific_constant":
            patterns = (
                "speed of light",
                "gravitational constant",
                "planck constant",
                "boltzmann constant",
                "avogadro constant",
                "avogadro number",
            )

            for phrase in patterns:
                if phrase in lowered:
                    return {
                        "name": phrase.replace(" ", "_")
                    }

            return {"name": text}

        return {}

    @staticmethod
    def _extract_expression(objective: str) -> str:
        text = objective.strip()
        lowered = text.lower()

        direct_prefixes = (
            "calculate ",
            "compute ",
            "what is ",
            "how much is ",
            "solve ",
        )

        for prefix in direct_prefixes:
            if lowered.startswith(prefix):
                return text[len(prefix):].strip().rstrip("?.!")

        expression = natural_expression(text)

        if expression:
            return expression

        return text.rstrip("?.!")

    def describe_route(self, objective: str) -> str:
        route = self.router.route(objective)

        return (
            f"{route.route_type.value} → {route.target} "
            f"({route.confidence:.2f}): {route.reason}"
        )
