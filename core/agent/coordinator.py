from __future__ import annotations

import re
from collections.abc import Iterator
from threading import Event
from uuid import uuid4

from core.agent.inference import InferenceEngine
from core.memory import MemoryManager, MemoryType
from core.tools import create_tool_registry
from core.tools.calculator import natural_expression

from .phrasing import phrase_tool_answer
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

    def _route(self, objective: str):
        route = self.router.route(objective)

        if route.route_type is RouteType.CHAT:
            from core.config import settings

            if settings.agent.intent_llm and len(objective) <= 400:
                from core.agent.intent_llm import classify

                return classify(objective) or route

        return route

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
        history: list[dict[str, str]] | None = None,
    ) -> AgentResult:
        route = self._route(objective)

        if route.route_type is RouteType.MEMORY:
            return self._memory_route(route.target, objective)

        if route.route_type is RouteType.SELF:
            return self._self_route(route.target, objective, history)

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

            if not result.success and route.reason.startswith("LLM"):
                return self._chat(
                    objective,
                    history=history,
                    memories=self._chat_memories(objective),
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

            phrased = phrase_tool_answer(
                route.target,
                objective,
                self._tool_arguments(route.target, objective, context),
                result.output,
            )

            if phrased is not None:
                return AgentResult(
                    task_id="tool-" + route.target,
                    agent_name=route.target,
                    status=AgentStatus.COMPLETE,
                    output=phrased,
                    steps=1,
                    history=[tool_step, observation],
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
                output=self._format_tool_output(route.target, result.output),
                steps=1,
                history=[
                    tool_step,
                    observation,
                ],
            )

        if route.route_type is RouteType.CHAT:
            return self._chat(
                objective,
                history=history,
                memories=self._chat_memories(objective),
            )

        spec = self.registry.get(route.target)

        if not spec.allowed_tools:
            # A text-only specialist: answer in one natural pass with its
            # persona. (Forcing a small model through JSON mangles code.)
            return self._chat(
                objective,
                history=history,
                memories=self._chat_memories(objective),
                persona=spec.system_prompt,
                agent_name=spec.name,
            )

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
    def _is_personal(memory) -> bool:
        """Things the user told SALLY, not logs like daily briefs."""
        if memory.memory_type in {
            MemoryType.PROFILE,
            MemoryType.FACT,
            MemoryType.PREFERENCE,
        }:
            return len(memory.content) <= 300

        return (
            memory.memory_type not in {MemoryType.DAILY, MemoryType.SESSION}
            and len(memory.content) <= 200
        )

    def _known_name(self) -> str | None:
        for memory in self.memory.recent(limit=50):
            if memory.content.startswith("The user's name is "):
                return memory.content.rsplit(" is ", 1)[-1].rstrip(".")

        return None

    def _chat_memories(self, objective: str) -> str:
        """Who the user is (always) plus anything relevant to this message."""
        recent = [m for m in self.memory.recent(limit=50) if self._is_personal(m)]
        profile = [m for m in recent if m.memory_type is MemoryType.PROFILE]
        relevant = [
            m for m in self.memory.search(objective, limit=5) if self._is_personal(m)
        ]

        lines: list[str] = []

        for memory in profile[:5] + relevant:
            line = f"- {memory.content}"

            if line not in lines:
                lines.append(line)

        return "\n".join(lines)

    def _self_route(
        self,
        target: str,
        objective: str,
        history: list[dict[str, str]] | None,
    ) -> AgentResult:
        from core.agent.selfknowledge import answer_self, variant_for

        answer = answer_self(
            target,
            name=self._known_name(),
            variant=variant_for(objective, len(history or [])),
        )

        return AgentResult(
            task_id=f"self-{uuid4().hex[:12]}",
            agent_name="self",
            status=AgentStatus.COMPLETE,
            output=answer,
            steps=1,
            history=[],
        )

    def _memory_route(self, target: str, objective: str) -> AgentResult:
        from core.agent.understanding import (
            acknowledgement,
            answer_recall,
            extract_memory_notes,
            recall_kind,
        )

        if target == "recall":
            answer = answer_recall(
                recall_kind(objective) or "about",
                [m for m in self.memory.recent(limit=50) if self._is_personal(m)],
            )

            return AgentResult(
                task_id=f"memory-{uuid4().hex[:12]}",
                agent_name="memory",
                status=AgentStatus.COMPLETE,
                output=answer,
                steps=1,
                history=[],
            )

        notes = extract_memory_notes(objective)
        existing = {m.content.lower(): m for m in self.memory.recent(limit=50)}
        replaced_name = False

        for note in notes:
            if note.kind == "name":
                for memory in existing.values():
                    if memory.content.startswith("The user's name is"):
                        replaced_name = (
                            memory.content.lower() != note.content.lower()
                        )
                        self.memory.forget(memory.id)

            if note.content.lower() in existing and note.kind != "name":
                continue

            self.memory.remember(
                note.content,
                memory_type=note.memory_type,
                importance=note.importance,
            )

        return AgentResult(
            task_id=f"memory-{uuid4().hex[:12]}",
            agent_name="memory",
            status=AgentStatus.COMPLETE,
            output=acknowledgement(notes, replaced_name),
            steps=1,
            history=[],
        )

    @staticmethod
    def _format_tool_output(name: str, output: object) -> str:
        if name == "machine_status" and isinstance(output, dict):
            from core.native import describe_machine

            return describe_machine(output)

        return str(output)

    _BASE_PROMPT = (
        "You are SALLY, a warm, sharp personal assistant running privately "
        "on the user's own machine. Talk like a thoughtful friend: natural, "
        "direct and brief (1 to 4 sentences unless asked for more), with no "
        "disclaimers and no lists unless useful. If you don't know "
        "something, say so plainly. Never claim to have done something you "
        "haven't, and never invent abilities: you can only do exact maths, "
        "tell the time and date, convert units, look up physics constants, "
        "report this machine's status, and remember what the user tells you."
    )

    # Keep the prompt small: it is re-read on every message, and on a small
    # CPU its length is most of the wait.
    _HISTORY_MESSAGES = 8
    _HISTORY_CHARS = 2400
    _MESSAGE_CHARS = 700

    def _chat_messages(
        self,
        objective: str,
        *,
        history: list[dict[str, str]] | None = None,
        memories: str = "",
        persona: str | None = None,
    ) -> list[dict[str, str]]:
        system_prompt = self._BASE_PROMPT

        if persona:
            system_prompt = (
                f"{persona} You are speaking as SALLY, a personal assistant: "
                "be practical and concise, and put code in code blocks."
            )

        if memories:
            system_prompt += (
                "\n\nWhat you know about the user:\n" + memories[:800]
            )

        kept: list[dict[str, str]] = []
        used = 0

        recent = [
            item
            for item in (history or [])[-self._HISTORY_MESSAGES:]
            if item.get("role") in {"user", "assistant"} and item.get("content")
        ]

        for item in reversed(recent):
            content = item["content"]

            if len(content) > self._MESSAGE_CHARS:
                content = content[: self._MESSAGE_CHARS].rstrip() + "…"

            if used + len(content) > self._HISTORY_CHARS and kept:
                break

            kept.append({"role": item["role"], "content": content})
            used += len(content)

        kept.reverse()

        return [
            {"role": "system", "content": system_prompt},
            *kept,
            {"role": "user", "content": objective},
        ]

    @staticmethod
    def _chat_result(
        status: AgentStatus,
        output: str = "",
        error: str | None = None,
        agent_name: str = "conversation",
    ) -> AgentResult:
        return AgentResult(
            task_id=f"chat-{uuid4().hex[:12]}",
            agent_name=agent_name,
            status=status,
            output=output,
            steps=1,
            history=[],
            error=error,
        )

    def _persona(self, route) -> tuple[str | None, str]:
        """(system prompt, result name) for routes answered conversationally."""
        if route.route_type is RouteType.CHAT:
            return None, "conversation"

        if route.route_type is RouteType.AGENT:
            spec = self.registry.get(route.target)

            if not spec.allowed_tools:
                return spec.system_prompt, spec.name

        return None, "conversation"

    def _chat(
        self,
        objective: str,
        *,
        history: list[dict[str, str]] | None = None,
        memories: str = "",
        persona: str | None = None,
        agent_name: str = "conversation",
    ) -> AgentResult:
        from core.config import settings
        from core.llm import chat as llm_chat

        messages = self._chat_messages(
            objective,
            history=history,
            memories=memories,
            persona=persona,
        )

        try:
            answer = llm_chat(
                messages,
                max_tokens=settings.agent.max_tokens,
                temperature=settings.agent.chat_temperature,
                timeout_s=settings.agent.timeout,
            ).strip()
        except Exception as exc:
            return self._chat_result(
                AgentStatus.FAILED,
                error=str(exc),
                agent_name=agent_name,
            )

        if not answer:
            return self._chat_result(
                AgentStatus.FAILED,
                error="Conversation model produced an empty response.",
                agent_name=agent_name,
            )

        return self._chat_result(
            AgentStatus.COMPLETE,
            answer,
            agent_name=agent_name,
        )

    def stream(
        self,
        objective: str,
        *,
        context: dict | None = None,
        history: list[dict[str, str]] | None = None,
        cancel: Event | None = None,
    ) -> Iterator[dict]:
        """
        Yield {"type": "token", "text": ...} events while a conversational
        reply is generated, then one {"type": "final", "result": AgentResult}.

        Tool and agent routes are not streamed; they yield only the final
        event.
        """
        route = self._route(objective)
        persona, agent_name = self._persona(route)

        if persona is None and route.route_type is not RouteType.CHAT:
            yield {
                "type": "final",
                "result": self.run(
                    objective,
                    context=context,
                    history=history,
                ),
            }
            return

        from core import llm as llm_module
        from core.config import settings

        messages = self._chat_messages(
            objective,
            history=history,
            memories=self._chat_memories(objective),
            persona=persona,
        )

        parts: list[str] = []

        try:
            for delta in llm_module.stream_chat(
                messages,
                max_tokens=settings.agent.max_tokens,
                temperature=settings.agent.chat_temperature,
                timeout_s=settings.agent.timeout,
                cancel=cancel,
            ):
                parts.append(delta)
                yield {"type": "token", "text": delta}
        except Exception as exc:
            yield {
                "type": "final",
                "result": self._chat_result(
                    AgentStatus.FAILED,
                    error=str(exc),
                    agent_name=agent_name,
                ),
            }
            return

        answer = "".join(parts).strip()

        if not answer:
            result = self._chat_result(
                AgentStatus.FAILED,
                error="Conversation model produced an empty response.",
                agent_name=agent_name,
            )
        else:
            result = self._chat_result(
                AgentStatus.COMPLETE,
                answer,
                agent_name=agent_name,
            )

        yield {"type": "final", "result": result}

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
                remainder = text[len(prefix):].strip().rstrip("?.!")

                # "What is 144 divided by 12?" -> "(144) / (12)"
                return natural_expression(remainder) or remainder

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
