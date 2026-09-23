from __future__ import annotations

import json
from collections.abc import Callable
from uuid import uuid4

from core.config import settings
from core.llm import chat as llm_chat

from .protocol import parse_action
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

    Action names are dynamic and model-defined. The runtime interprets
    the structure of an action rather than maintaining a hardcoded
    vocabulary of possible action names.
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
        tools_text = ", ".join(available) if available else "none"

        return (
            f"{spec.system_prompt}\n\n"
            "You are operating inside SALLY's agent runtime.\n"
            "You MUST respond with exactly one JSON object and no "
            "surrounding explanation.\n\n"
            "Action names are dynamic. Choose any concise action name "
            "that accurately describes what you are doing. "
            "Action names are not predefined.\n\n"
            "To provide a final response, include an 'answer' field.\n"
            '{"action":"<your_action>","answer":"<your_answer>"}\n\n'
            "To use an executable capability, include a registered "
            "tool name and its arguments.\n"
            '{"action":"<your_action>","tool":"<registered_tool>",'
            '"arguments":{...}}\n\n'
            f"Available tools: {tools_text}\n"
            "Never invent a tool name that is not listed above.\n"
            "You do not have web browsing, URL opening, messaging, "
            "filesystem access, or external-action capabilities unless "
            "a corresponding registered tool is explicitly listed.\n"
            "Never invent URLs, claim that you searched the web, or claim "
            "that you performed an external action when you did not.\n"
            "When a requested capability is unavailable, say so plainly "
            "and answer from the capabilities you actually have."
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

        return json.dumps(
            payload,
            ensure_ascii=False,
            default=str,
        )

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

                # Explicit tool takes precedence.
                # A dynamic action name may also directly identify a
                # registered tool without requiring a separate `tool`.
                tool_name = (
                    action.tool
                    or (
                        action.action.lower()
                        if self.tools.has(action.action.lower())
                        else ""
                    )
                )

                if tool_name:
                    tool_name = tool_name.strip().lower()

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
                            output=self._format_tool_result(
                                tool_result
                            ),
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
                                "Continue the objective. Respond with "
                                "exactly one JSON object."
                            ),
                        }
                    )

                    continue

                # Any dynamic action with an answer is a valid final
                # response, regardless of the action name.
                if action.answer:
                    task.steps.append(
                        AgentStep(
                            number=step_number,
                            step_type=StepType.FINAL,
                            output=action.answer,
                        )
                    )

                    task.status = AgentStatus.COMPLETE
                    task.result = action.answer

                    return AgentResult(
                        task_id=task.task_id,
                        agent_name=spec.name,
                        status=AgentStatus.COMPLETE,
                        output=task.result,
                        steps=step_number,
                        history=task.steps.copy(),
                    )

                # Unknown non-executable action: do not invent behavior
                # for it. Feed a structured observation back to the model
                # and let it decide what to do next.
                observation = (
                    f"Action `{action.action}` is not an executable "
                    "registered tool, so no tool was run. Continue the "
                    "objective by providing an answer or using a "
                    "registered tool."
                )

                task.steps.append(
                    AgentStep(
                        number=step_number,
                        step_type=StepType.OBSERVE,
                        output=observation,
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
                        "content": observation,
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

        finally:
            self.active_tasks.pop(task.task_id, None)
