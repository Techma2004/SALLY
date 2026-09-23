from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from core.agent import Coordinator
from core.agent.types import AgentStatus


@dataclass(frozen=True)
class GatewayRequest:
    message: str
    user_id: str = "anonymous"
    source: str = "local"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GatewayResponse:
    answer: str
    status: AgentStatus
    task_id: str
    agent_name: str
    source: str
    error: str | None = None


class Gateway:
    """
    Unified front door for SALLY.

    Interfaces such as the web UI, Telegram, TUI, and future APIs
    should call this layer instead of talking directly to the
    Coordinator or agent runtime.
    """

    def __init__(self, coordinator: Coordinator | None = None) -> None:
        self.coordinator = coordinator or Coordinator()

    def handle(
        self,
        message: str,
        *,
        user_id: str = "anonymous",
        source: str = "local",
        metadata: dict[str, Any] | None = None,
    ) -> GatewayResponse:
        message = message.strip()

        if not message:
            raise ValueError("Message cannot be empty.")

        result = self.coordinator.run(
            message,
            context=metadata or {},
        )

        return GatewayResponse(
            answer=result.output,
            status=result.status,
            task_id=result.task_id,
            agent_name=result.agent_name,
            source=source,
            error=result.error,
        )

    def chat(
        self,
        message: str,
        *,
        user_id: str = "anonymous",
        source: str = "local",
    ) -> str:
        response = self.handle(
            message,
            user_id=user_id,
            source=source,
        )

        if response.status is AgentStatus.COMPLETE:
            return response.answer

        if response.error:
            return f"SALLY could not complete the request: {response.error}"

        return "SALLY could not complete the request."


def gateway_chat(
    user_id: str,
    message: str,
    source: str = "local",
) -> str:
    """
    Backward-compatible helper for existing interface adapters.
    """
    return Gateway().chat(
        message,
        user_id=user_id or "anonymous",
        source=source,
    )
