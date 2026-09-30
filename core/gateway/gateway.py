from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any

from core.agent import Coordinator
from core.agent.types import AgentStatus

# Caps how many distinct user_ids we remember a "last result" for. user_id
# is client-supplied and unauthenticated, so without a cap a caller could
# grow this dict without bound by sending many unique user_ids (DoS via
# memory exhaustion). Oldest entries are evicted first.
_MAX_TRACKED_USERS = 5000


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
        self._last_results: OrderedDict[str, str] = OrderedDict()

    def _remember_last_result(self, user_id: str, value: str) -> None:
        if user_id in self._last_results:
            self._last_results.move_to_end(user_id)
        elif len(self._last_results) >= _MAX_TRACKED_USERS:
            self._last_results.popitem(last=False)

        self._last_results[user_id] = value

    def handle(
        self,
        message: str,
        *,
        user_id: str = "anonymous",
        source: str = "local",
        metadata: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
    ) -> GatewayResponse:
        message = message.strip()

        if not message:
            raise ValueError("Message cannot be empty.")

        request_context = dict(metadata or {})

        if user_id in self._last_results:
            request_context.setdefault(
                "last_result",
                self._last_results[user_id],
            )

        result = self.coordinator.run(
            message,
            context=request_context,
            history=history,
        )

        if (
            result.status is AgentStatus.COMPLETE
            and result.agent_name == "calculator"
        ):
            try:
                float(result.output)
                self._remember_last_result(user_id, result.output)
            except (TypeError, ValueError):
                pass


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
