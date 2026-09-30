from fastapi.testclient import TestClient

from core.agent.types import AgentResult, AgentStatus
from core.gateway import web
from core.gateway.gateway import Gateway
from core.memory import MemoryManager, MemoryStore


class RecordingCoordinator:
    def __init__(self, memory):
        self.memory = memory
        self.histories = []

    def describe_route(self, objective):
        return "chat → conversation"

    def run(self, objective, *, context=None, history=None):
        self.histories.append(history)
        return AgentResult(
            task_id="t",
            agent_name="conversation",
            status=AgentStatus.COMPLETE,
            output=f"reply to: {objective}",
            steps=1,
            history=[],
        )


def test_web_chat_passes_prior_messages_as_history(tmp_path, monkeypatch):
    memory = MemoryManager(
        store=MemoryStore(db_path=str(tmp_path / "web.db"))
    )
    coordinator = RecordingCoordinator(memory)
    monkeypatch.setattr(web, "gateway", Gateway(coordinator=coordinator))

    client = TestClient(web.app, client=("127.0.0.1", 5000))

    first = client.post("/chat", json={"message": "My name is Edima."}).json()
    client.post(
        "/chat",
        json={
            "message": "What is my name?",
            "conversation_id": first["conversation_id"],
        },
    )

    assert coordinator.histories[0] == []
    assert coordinator.histories[1] == [
        {"role": "user", "content": "My name is Edima."},
        {"role": "assistant", "content": "reply to: My name is Edima."},
    ]


def test_conversation_messages_limit_returns_latest_in_order(tmp_path):
    memory = MemoryManager(
        store=MemoryStore(db_path=str(tmp_path / "limit.db"))
    )
    conversation = memory.create_conversation("u", title="t")

    for i in range(10):
        memory.add_conversation_message(
            conversation.id, role="user", content=f"m{i}"
        )

    latest = memory.conversation_messages(conversation.id, limit=3)

    assert [m.content for m in latest] == ["m7", "m8", "m9"]
    assert len(memory.conversation_messages(conversation.id)) == 10
