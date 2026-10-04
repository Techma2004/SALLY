from __future__ import annotations

from collections import OrderedDict
from collections.abc import Iterator
from threading import Event
from dataclasses import dataclass, field
from typing import Any

from core.agent import Coordinator
from core.agent.types import AgentResult, AgentStatus

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

    def _request_context(
        self,
        user_id: str,
        metadata: dict[str, Any] | None,
    ) -> dict[str, Any]:
        request_context = dict(metadata or {})

        if user_id in self._last_results:
            request_context.setdefault(
                "last_result",
                self._last_results[user_id],
            )

        return request_context

    def _to_response(
        self,
        result: AgentResult,
        user_id: str,
        source: str,
    ) -> GatewayResponse:
        if (
            result.status is AgentStatus.COMPLETE
            and result.agent_name == "calculator"
        ):
            # The answer is phrased for people; follow-ups need the raw number.
            raw = next(
                (
                    step.tool_result.output
                    for step in result.history
                    if getattr(step, "tool_result", None) is not None
                ),
                result.output,
            )

            try:
                float(raw)
                self._remember_last_result(user_id, str(raw))
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

        result = self.coordinator.run(
            message,
            context=self._request_context(user_id, metadata),
            history=history,
        )

        return self._to_response(result, user_id, source)

    def stream(
        self,
        message: str,
        *,
        user_id: str = "anonymous",
        source: str = "local",
        metadata: dict[str, Any] | None = None,
        history: list[dict[str, str]] | None = None,
        cancel: Event | None = None,
    ) -> Iterator[dict[str, Any]]:
        """
        Like handle(), but yields {"type": "token", "text": ...} events as a
        conversational reply is generated, then one
        {"type": "final", "response": GatewayResponse}.
        """
        message = message.strip()

        if not message:
            raise ValueError("Message cannot be empty.")

        for event in self.coordinator.stream(
            message,
            context=self._request_context(user_id, metadata),
            history=history,
            cancel=cancel,
        ):
            if event["type"] == "token":
                yield event
            else:
                yield {
                    "type": "final",
                    "response": self._to_response(
                        event["result"],
                        user_id,
                        source,
                    ),
                }

    def converse(
        self,
        message: str,
        *,
        user_id: str,
        source: str,
    ) -> GatewayResponse:
        """
        handle() with a persistent per-user thread, for messaging channels
        (Telegram, WhatsApp) that have no conversation id of their own.
        """
        memory = self.coordinator.memory
        owner = f"{source}:{user_id}"

        recent = memory.recent_conversations(owner, limit=1)
        conversation = (
            recent[0]
            if recent
            else memory.create_conversation(owner, title=message.strip()[:60])
        )

        history = [
            {"role": item.role, "content": item.content}
            for item in memory.conversation_messages(conversation.id, limit=8)
        ]

        response = self.handle(
            message,
            user_id=user_id,
            source=source,
            history=history,
        )

        memory.add_conversation_message(
            conversation.id,
            role="user",
            content=message.strip(),
        )

        if response.answer:
            memory.add_conversation_message(
                conversation.id,
                role="assistant",
                content=response.answer,
            )

        return response

    def chat(
        self,
        message: str,
        *,
        user_id: str = "anonymous",
        source: str = "local",
        history: list[dict[str, str]] | None = None,
    ) -> str:
        response = self.handle(
            message,
            user_id=user_id,
            source=source,
            history=history,
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
