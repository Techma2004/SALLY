from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"
    FAILED = "failed"
    TIMEOUT = "timeout"


class StepType(str, Enum):
    THINK = "think"
    TOOL = "tool"
    OBSERVE = "observe"
    FINAL = "final"


@dataclass(frozen=True)
class AgentSpec:
    name: str
    role: str
    description: str
    system_prompt: str
    allowed_tools: tuple[str, ...] = ()
    max_steps: int | None = None
    max_tokens: int | None = None
    temperature: float | None = None


@dataclass(frozen=True)
class ToolRequest:
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ToolResult:
    name: str
    success: bool
    output: Any = None
    error: str | None = None


@dataclass
class AgentStep:
    number: int
    step_type: StepType
    input: str = ""
    output: str = ""
    tool_request: ToolRequest | None = None
    tool_result: ToolResult | None = None


@dataclass
class AgentTask:
    task_id: str
    objective: str
    agent_name: str
    context: dict[str, Any] = field(default_factory=dict)
    status: AgentStatus = AgentStatus.IDLE
    steps: list[AgentStep] = field(default_factory=list)
    result: str | None = None
    error: str | None = None


@dataclass
class AgentResult:
    task_id: str
    agent_name: str
    status: AgentStatus
    output: str
    steps: int = 0
    history: list[AgentStep] = field(default_factory=list)
    error: str | None = None
