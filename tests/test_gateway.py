from core.agent.types import AgentResult, AgentStatus
from core.gateway import Gateway


class FakeCoordinator:
    def run(self, objective, *, context=None):
        return AgentResult(
            task_id="test-task",
            agent_name="planning",
            status=AgentStatus.COMPLETE,
            output=f"Handled: {objective}",
            steps=1,
            history=[],
        )


def test_gateway_forwards_requests_to_coordinator():
    gateway = Gateway(coordinator=FakeCoordinator())

    response = gateway.handle(
        "Plan the next SALLY milestone.",
        user_id="test-user",
        source="web",
    )

    assert response.status is AgentStatus.COMPLETE
    assert response.answer == "Handled: Plan the next SALLY milestone."
    assert response.task_id == "test-task"
    assert response.agent_name == "planning"
    assert response.source == "web"


def test_gateway_chat_returns_answer():
    gateway = Gateway(coordinator=FakeCoordinator())

    answer = gateway.chat(
        "Explain the gateway.",
        user_id="test-user",
        source="tui",
    )

    assert answer == "Handled: Explain the gateway."


def test_gateway_rejects_empty_message():
    gateway = Gateway(coordinator=FakeCoordinator())

    try:
        gateway.handle("   ")
    except ValueError as exc:
        assert "empty" in str(exc).lower()
    else:
        raise AssertionError("Expected empty-message validation error.")


def test_gateway_uses_default_safe_tools():
    gateway = Gateway()

    response = gateway.handle(
        "Calculate 25 * 40",
        user_id="test-user",
        source="test",
    )

    assert response.status.value == "complete"
    assert response.answer == "1000"
