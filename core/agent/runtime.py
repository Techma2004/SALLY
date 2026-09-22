from __future__ import annotations

import json
from uuid import uuid4
from collections.abc import Callable

from core.config import settings
from core.llm import chat as llm_chat

from .protocol import ActionType, parse_action
from .tools import ToolRegistry
from .types import (
    AgentResult,
    AgentSpec,
    AgentStatus,
    AgentStep,
    AgentTask,
    StepType,
    ToolRequest,
    ToolResult,
)


class AgentRuntime:
    """
    Executes SALLY agents through a real tool/observation loop.

    Flow:

        LLM decision
            ↓
        tool request
            ↓
        tool execution
            ↓
        observation
            ↓
        LLM decision
            ↓
        final answer

    Runtime limits are controlled centrally through SALLY configuration.
    """

    def __init__(
        self,
        tools: ToolRegistry | None = None,
        llm: Callable[..., str] | None = None,
    ) -> None:
        self.tools = tools or ToolRegistry()
        self.llm = llm or llm_chat
        self.active_tasks: dict[str, AgentTask] = {}

    def _available_tools(self, spec: AgentSpec) -> tuple[str, ...]:
        return tuple(
            name for name in spec.allowed_tools
            if self.tools.has(name)
        )

    def _system_prompt(self, spec: AgentSpec) -> str:
        available = self._available_tools(spec)

        if available:
            tools_text = ", ".join(available)
        else:
            tools_text = "none"

        return (
            f"{spec.system_prompt}\n\n"
            "You are operating inside SALLY's agent runtime.\n"
            "You MUST respond with exactly one JSON object and no surrounding explanation.\n\n"
            "To use a tool:\n"
            '{"action":"tool","tool":"<tool_name>","arguments":{...}}\n\n'
            "To finish:\n"
            '{"action":"final","answer":"<answer>"}\n\n'
            f"Available tools: {tools_text}\n"
            "Never invent a tool that is not listed above."
        )

    @staticmethod
    def _format_tool_result(result: ToolResult) -> str:
        if result.success:
            payload = {
                "success": True,
                "output": result.output,
            }
        else:
            payload = {
                "success": False,
                "error": result.error,
            }

        return json.dumps(payload, ensure_ascii=False, default=str)

    def run(
        self,
        spec: AgentSpec,
        objective: str,
        *,
        context: dict | None = None,
    ) -> AgentResult:
        task = AgentTask(
            task_id=uuid4().hex[:12],
            objective=objective,
            agent_name=spec.name,
            context=context or {},
            status=AgentStatus.RUNNING,
        )

        self.active_tasks[task.task_id] = task

        config = settings.agent
        max_steps = (
            spec.max_steps
            if spec.max_steps is not None
            else config.max_steps
        )
        max_tokens = (
            spec.max_tokens
            if spec.max_tokens is not None
            else config.max_tokens
        )
        temperature = (
            spec.temperature
            if spec.temperature is not None
            else config.temperature
        )

        context_text = ""
        if task.context:
            context_text = (
                "\n\nAdditional context:\n"
                + "\n".join(
                    f"{key}: {value}"
                    for key, value in task.context.items()
                )
            )

        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": self._system_prompt(spec),
            },
            {
                "role": "user",
                "content": (
                    f"Objective:\n{objective}"
                    f"{context_text}\n\n"
                    "Complete the objective."
                ),
            },
        ]

        try:
            for step_number in range(1, max_steps + 1):
                raw_output = self.llm(
                    messages,
                    max_tokens=max_tokens,
                    temperature=temperature,
                )

                task.steps.append(
                    AgentStep(
                        number=step_number,
                        step_type=StepType.THINK,
                        input=messages[-1]["content"],
                        output=raw_output,
                    )
                )

                try:
                    action = parse_action(raw_output)
                except ValueError as exc:
                    task.status = AgentStatus.FAILED
                    task.error = f"Invalid agent action: {exc}"

                    return AgentResult(
                        task_id=task.task_id,
                        agent_name=spec.name,
                        status=AgentStatus.FAILED,
                        output="",
                        steps=step_number,
                        history=task.steps.copy(),
                        error=task.error,
                    )

                if action.action is ActionType.FINAL:
                    task.steps.append(
                        AgentStep(
                            number=step_number,
                            step_type=StepType.FINAL,
                            output=action.answer or "",
                        )
                    )

                    task.status = AgentStatus.COMPLETE
                    task.result = action.answer or ""

                    return AgentResult(
                        task_id=task.task_id,
                        agent_name=spec.name,
                        status=AgentStatus.COMPLETE,
                        output=task.result,
                        steps=step_number,
                        history=task.steps.copy(),
                    )

                tool_name = action.tool or ""

                if tool_name not in spec.allowed_tools:
                    task.status = AgentStatus.FAILED
                    task.error = (
                        f"Agent '{spec.name}' is not allowed to use "
                        f"tool '{tool_name}'."
                    )

                    return AgentResult(
                        task_id=task.task_id,
                        agent_name=spec.name,
                        status=AgentStatus.FAILED,
                        output="",
                        steps=step_number,
                        history=task.steps.copy(),
                        error=task.error,
                    )

                if not self.tools.has(tool_name):
                    tool_result = ToolResult(
                        name=tool_name,
                        success=False,
                        error=f"Unknown tool: {tool_name}",
                    )
                else:
                    tool_request = ToolRequest(
                        name=tool_name,
                        arguments=action.arguments,
                    )

                    task.steps.append(
                        AgentStep(
                            number=step_number,
                            step_type=StepType.TOOL,
                            tool_request=tool_request,
                        )
                    )

                    tool_result = self.tools.execute(tool_request)

                task.steps.append(
                    AgentStep(
                        number=step_number,
                        step_type=StepType.OBSERVE,
                        tool_result=tool_result,
                        output=self._format_tool_result(tool_result),
                    )
                )

                messages.append(
                    {
                        "role": "assistant",
                        "content": raw_output,
                    }
                )

                messages.append(
                    {
                        "role": "user",
                        "content": (
                            f"Tool `{tool_name}` returned:\n"
                            f"{self._format_tool_result(tool_result)}\n\n"
                            "Continue the objective. Respond with exactly "
                            "one JSON object."
                        ),
                    }
                )

            task.status = AgentStatus.FAILED
            task.error = (
                f"Agent reached its maximum step limit ({max_steps}) "
                "without producing a final answer."
            )

            return AgentResult(
                task_id=task.task_id,
                agent_name=spec.name,
                status=AgentStatus.FAILED,
                output="",
                steps=max_steps,
                history=task.steps.copy(),
                error=task.error,
            )

        except Exception as exc:
            task.status = AgentStatus.FAILED
            task.error = str(exc)

            return AgentResult(
                task_id=task.task_id,
                agent_name=spec.name,
                status=AgentStatus.FAILED,
                output="",
                steps=len(
                    {
                        step.number
                        for step in task.steps
                        if step.step_type is StepType.THINK
                    }
                ),
                history=task.steps.copy(),
                error=str(exc),
            )
