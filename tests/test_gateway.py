from core.agent.types import AgentResult, AgentStatus
from core.gateway import Gateway


class FakeCoordinator:
    def run(self, objective, *, context=None, history=None):
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


def test_gateway_chat_forwards_history():
    captured = {}

    class RecordingCoordinator(FakeCoordinator):
        def run(self, objective, *, context=None, history=None):
            captured["history"] = history
            return super().run(objective, context=context, history=history)

    gateway = Gateway(coordinator=RecordingCoordinator())
    history = [{"role": "user", "content": "hi"}]

    gateway.chat("Hello again.", user_id="test-user", history=history)

    assert captured["history"] == history


def test_converse_keeps_a_persistent_thread_per_channel_user(tmp_path):
    from core.memory import MemoryManager, MemoryStore

    memory = MemoryManager(
        store=MemoryStore(db_path=str(tmp_path / "converse.db"))
    )
    seen = []

    class Coordinator(FakeCoordinator):
        def __init__(self):
            self.memory = memory

        def run(self, objective, *, context=None, history=None):
            seen.append(history)
            return super().run(objective, context=context, history=history)

    gateway = Gateway(coordinator=Coordinator())

    gateway.converse("first", user_id="42", source="telegram")
    gateway.converse("second", user_id="42", source="telegram")
    gateway.converse("other user", user_id="99", source="telegram")

    assert seen[0] == []
    assert seen[1] == [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "Handled: first"},
    ]
    assert seen[2] == []  # a different user never sees that thread
