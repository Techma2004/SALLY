from __future__ import annotations

import time
from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from threading import Event
from uuid import uuid4

from core.agent.inference import InferenceEngine
from core.memory import MemoryManager, MemoryType
from core.tools import create_tool_registry
from core.tools.calculator import natural_expression

from .narrator import (
    MAX_EXACT_CHARS,
    MAX_REPLY_CHARS,
    json_safe,
    narration_messages,
    numbers_to_keep,
    reply_keeps,
)
from .phrasing import phrase_tool_answer
from .registry import AgentRegistry, create_default_registry
from .router import RouteType, TaskRouter, create_router
from .runtime import AgentRuntime
from .turn import TurnBudget, TurnTrace
from .types import (
    AgentResult,
    AgentStatus,
    AgentStep,
    StepType,
    ToolRequest,
)


@dataclass
class Outcome:
    """The exact part of a turn, and what the model may be asked to phrase.

    ``event`` is the JSON-able record of what happened. ``verify`` lists the
    numbers/names a model rewrite must keep. ``recovers_failure`` marks a
    failed tool whose friendly explanation may replace the bare error.
    """

    result: AgentResult
    event: dict | None = None
    verify: list[str] = field(default_factory=list)
    narrate: bool = False
    recovers_failure: bool = False


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

    def _route(self, objective: str, trace: TurnTrace | None = None):
        route = self.router.route(objective)

        if route.route_type is RouteType.CHAT:
            from core.config import settings

            if (
                settings.agent.intent_llm
                and len(objective) <= 400
                and (trace is None or trace.can_call_model())
            ):
                from core.agent.intent_llm import classify

                if trace is not None:
                    trace.model_calls += 1

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
        cancel: Event | None = None,
    ) -> AgentResult:
        """One full turn, returned at once (drains the same loop as stream)."""
        final: AgentResult | None = None

        for event in self._turn(
            objective,
            context=context,
            history=history,
            cancel=cancel,
            streaming=False,
        ):
            if event["type"] == "final":
                final = event["result"]

        assert final is not None  # _turn always ends with a final event
        return final

    def stream(
        self,
        objective: str,
        *,
        context: dict | None = None,
        history: list[dict[str, str]] | None = None,
        cancel: Event | None = None,
    ) -> Iterator[dict]:
        """
        The same turn, with {"type": "token", "text": ...} events as words
        are generated, then one {"type": "final", "result": AgentResult}.

        Instant answers (small talk, tools without narration) yield only the
        final event.
        """
        yield from self._turn(
            objective,
            context=context,
            history=history,
            cancel=cancel,
            streaming=True,
        )

    # ------------------------------------------------------------------
    # The turn loop:  route -> execute -> (narrate | converse) -> finish
    # ------------------------------------------------------------------

    def _turn(
        self,
        objective: str,
        *,
        context: dict | None,
        history: list[dict[str, str]] | None,
        cancel: Event | None,
        streaming: bool,
    ) -> Iterator[dict]:
        from core.config import settings

        trace = TurnTrace(TurnBudget.from_settings(settings))

        with trace.phase("route"):
            route = self._route(objective, trace)

        trace.route = {
            "type": route.route_type.value,
            "target": route.target,
            "reason": route.reason,
        }

        persona, agent_name = self._persona(route)
        conversational = route.route_type is RouteType.CHAT or persona is not None

        if not conversational:
            with trace.phase("execute"):
                outcome = self._execute(route, objective, context, history, trace)

            if outcome is not None:
                trace.event = outcome.event

                if (
                    outcome.event is not None
                    and outcome.narrate
                    and settings.agent.narrate_tools
                    and trace.can_call_model()
                ):
                    yield from self._narrate(
                        objective, outcome, trace, cancel, streaming
                    )
                else:
                    yield {"type": "final", "result": self._finish(outcome.result, trace)}

                return

            # An LLM-chosen tool could not run: fall back to plain chat.
            trace.notes.append("tool unavailable, answered as conversation")

        yield from self._converse(
            objective, history, persona, agent_name, trace, cancel, streaming
        )

    @staticmethod
    def _finish(result: AgentResult, trace: TurnTrace) -> AgentResult:
        return replace(result, trace=trace.to_dict())

    def _recent(self, trace: TurnTrace | None = None) -> list:
        if trace is not None:
            return trace.recent_memories(self.memory)

        return self.memory.recent(limit=50)

    def _execute(
        self,
        route,
        objective: str,
        context: dict | None,
        history: list[dict[str, str]] | None,
        trace: TurnTrace,
    ) -> Outcome | None:
        """Run the exact part of a turn. None means: treat it as chat."""
        if route.route_type is RouteType.MEMORY:
            return self._memory_outcome(route.target, objective, trace)

        if route.route_type is RouteType.SELF:
            return Outcome(self._self_route(route.target, objective, history, trace))

        if route.route_type is RouteType.TOOL:
            return self._tool_outcome(route, objective, context, trace)

        spec = self.registry.get(route.target)
        combined_context = dict(context or {})
        memory_context = self._memory_context(objective)

        if memory_context:
            combined_context.update(memory_context)

        return Outcome(self.runtime.run(spec, objective, context=combined_context))

    def _tool_outcome(
        self,
        route,
        objective: str,
        context: dict | None,
        trace: TurnTrace,
    ) -> Outcome | None:
        arguments = self._tool_arguments(route.target, objective, context)
        request = ToolRequest(name=route.target, arguments=arguments)

        started = time.perf_counter()
        result = self.runtime.tools.execute(request)
        elapsed_ms = round((time.perf_counter() - started) * 1000, 1)

        trace.tool = {
            "name": route.target,
            "arguments": arguments,
            "ok": result.success,
            "ms": elapsed_ms,
        }

        if not result.success:
            if route.reason.startswith("LLM"):
                return None

            return Outcome(
                AgentResult(
                    task_id="tool-" + route.target,
                    agent_name=route.target,
                    status=AgentStatus.FAILED,
                    output="",
                    steps=1,
                    history=[],
                    error=result.error,
                ),
                event={
                    "source": "tool",
                    "tool": route.target,
                    "arguments": arguments,
                    "ok": False,
                    "error": str(result.error)[:300],
                },
                narrate=True,
                recovers_failure=True,
            )

        tool_step = AgentStep(
            number=1,
            step_type=StepType.TOOL,
            tool_request=request,
        )
        observation = AgentStep(
            number=1,
            step_type=StepType.OBSERVE,
            tool_result=result,
            output=str(result.output),
        )

        phrased = phrase_tool_answer(route.target, objective, arguments, result.output)

        if phrased is not None:
            return Outcome(
                AgentResult(
                    task_id="tool-" + route.target,
                    agent_name=route.target,
                    status=AgentStatus.COMPLETE,
                    output=phrased,
                    steps=1,
                    history=[tool_step, observation],
                ),
                event={
                    "source": "tool",
                    "tool": route.target,
                    "arguments": arguments,
                    "ok": True,
                    "summary": phrased,
                    "result": json_safe(result.output),
                    "elapsed_ms": elapsed_ms,
                },
                verify=numbers_to_keep(phrased),
                narrate=len(phrased) <= MAX_EXACT_CHARS,
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
            except Exception as exc:
                return Outcome(
                    AgentResult(
                        task_id="tool-" + route.target,
                        agent_name=route.target,
                        status=AgentStatus.FAILED,
                        output="",
                        steps=1,
                        history=[tool_step, observation],
                        error=f"Inference failed: {exc}",
                    )
                )

            return Outcome(
                AgentResult(
                    task_id="inference-" + route.target,
                    agent_name="inference",
                    status=AgentStatus.COMPLETE,
                    output=answer,
                    steps=2,
                    history=[
                        tool_step,
                        observation,
                        AgentStep(number=2, step_type=StepType.FINAL, output=answer),
                    ],
                )
            )

        return Outcome(
            AgentResult(
                task_id="tool-" + route.target,
                agent_name=route.target,
                status=AgentStatus.COMPLETE,
                output=self._format_tool_output(route.target, result.output),
                steps=1,
                history=[tool_step, observation],
            )
        )

    # ---------------- the model puts what happened into words -------------

    def _narrate(
        self,
        objective: str,
        outcome: Outcome,
        trace: TurnTrace,
        cancel: Event | None,
        streaming: bool,
    ) -> Iterator[dict]:
        from core import llm as llm_module

        messages = narration_messages(objective, outcome.event, self._known_name(trace))
        options = {
            "max_tokens": 120,
            "temperature": 0.3,
            "timeout_s": min(25.0, trace.remaining()),
        }

        trace.model_calls += 1
        parts: list[str] = []

        try:
            with trace.phase("narrate"):
                if streaming:
                    for delta in llm_module.stream_chat(messages, cancel=cancel, **options):
                        parts.append(delta)
                        yield {"type": "token", "text": delta}
                else:
                    text = llm_module.chat(messages, cancel=cancel, **options)
                    parts.append(text)
        except Exception as exc:
            trace.notes.append(f"narration failed: {type(exc).__name__}")
            parts = []

        reply = "".join(parts).strip()
        result = outcome.result

        if (
            reply
            and len(reply) <= MAX_REPLY_CHARS
            and reply_keeps(outcome.verify, reply)
        ):
            trace.narrated = True

            if outcome.recovers_failure:
                result = replace(result, status=AgentStatus.COMPLETE, error=None)

            result = replace(result, output=reply)
        elif reply:
            trace.notes.append("narration rejected: changed a fact; used exact text")

        if not result.output and result.error:
            trace.notes.append("narration unavailable")

        yield {"type": "final", "result": self._finish(result, trace)}

    # ---------------- plain conversation / text-only specialists ----------

    def _converse(
        self,
        objective: str,
        history: list[dict[str, str]] | None,
        persona: str | None,
        agent_name: str,
        trace: TurnTrace,
        cancel: Event | None,
        streaming: bool,
    ) -> Iterator[dict]:
        from core import llm as llm_module
        from core.config import settings

        messages = self._chat_messages(
            objective,
            history=history,
            memories=self._chat_memories(objective, trace),
            persona=persona,
        )
        options = {
            "max_tokens": settings.agent.max_tokens,
            "temperature": settings.agent.chat_temperature,
            "timeout_s": trace.remaining(),
            "cancel": cancel,
        }

        trace.model_calls += 1
        parts: list[str] = []

        try:
            with trace.phase("model"):
                if streaming:
                    for delta in llm_module.stream_chat(messages, **options):
                        parts.append(delta)
                        yield {"type": "token", "text": delta}
                else:
                    parts.append(llm_module.chat(messages, **options))
        except Exception as exc:
            yield {
                "type": "final",
                "result": self._finish(
                    self._chat_result(AgentStatus.FAILED, error=str(exc), agent_name=agent_name),
                    trace,
                ),
            }
            return

        answer = "".join(parts).strip()

        if answer:
            result = self._chat_result(AgentStatus.COMPLETE, answer, agent_name=agent_name)
        else:
            result = self._chat_result(
                AgentStatus.FAILED,
                error="Conversation model produced an empty response.",
                agent_name=agent_name,
            )

        yield {"type": "final", "result": self._finish(result, trace)}

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

    def _known_name(self, trace: TurnTrace | None = None) -> str | None:
        for memory in self._recent(trace):
            if memory.content.startswith("The user's name is "):
                return memory.content.rsplit(" is ", 1)[-1].rstrip(".")

        return None

    def _chat_memories(self, objective: str, trace: TurnTrace | None = None) -> str:
        """Who the user is (always) plus anything relevant to this message."""
        recent = [m for m in self._recent(trace) if self._is_personal(m)]
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
        trace: TurnTrace | None = None,
    ) -> AgentResult:
        from core.agent.selfknowledge import answer_self, variant_for

        answer = answer_self(
            target,
            name=self._known_name(trace),
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

    def _memory_outcome(
        self,
        target: str,
        objective: str,
        trace: TurnTrace,
    ) -> Outcome:
        from core.agent.understanding import (
            acknowledgement,
            answer_recall,
            extract_memory_notes,
            recall_kind,
        )

        def result(text: str) -> AgentResult:
            return AgentResult(
                task_id=f"memory-{uuid4().hex[:12]}",
                agent_name="memory",
                status=AgentStatus.COMPLETE,
                output=text,
                steps=1,
                history=[],
            )

        if target == "recall":
            kind = recall_kind(objective) or "about"
            personal = [m for m in self._recent(trace) if self._is_personal(m)]
            answer = answer_recall(kind, personal)
            outcome = Outcome(result(answer))

            if kind == "name":
                name = (
                    answer[len("Your name is "):].rstrip(".")
                    if answer.startswith("Your name is ")
                    else None
                )
                outcome.event = {
                    "source": "memory",
                    "action": "recalled",
                    "question": "the user's name",
                    "found": name is not None,
                    "name": name,
                }
                outcome.verify = [name] if name else []
                outcome.narrate = True

            return outcome

        notes = extract_memory_notes(objective)
        existing = {m.content.lower(): m for m in self._recent(trace)}
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

        trace.forget_cache()

        return Outcome(
            result(acknowledgement(notes, replaced_name)),
            event={
                "source": "memory",
                "action": "saved",
                "items": [note.content for note in notes],
                "replaced_previous_name": replaced_name,
            },
            verify=[note.detail for note in notes if note.kind in {"name", "place"}],
            narrate=True,
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
