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


def _sse_events(response):
    import json

    return [
        json.loads(line[len("data: "):])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]


class StreamingCoordinator(RecordingCoordinator):
    def stream(self, objective, *, context=None, history=None, cancel=None):
        self.histories.append(history)
        for word in ("Hel", "lo"):
            yield {"type": "token", "text": word}
        yield {
            "type": "final",
            "result": AgentResult(
                task_id="s",
                agent_name="conversation",
                status=AgentStatus.COMPLETE,
                output="Hello",
                steps=1,
                history=[],
            ),
        }


def test_chat_stream_emits_tokens_then_done_and_saves_reply(
    tmp_path, monkeypatch
):
    memory = MemoryManager(store=MemoryStore(db_path=str(tmp_path / "s.db")))
    monkeypatch.setattr(
        web, "gateway", Gateway(coordinator=StreamingCoordinator(memory))
    )
    client = TestClient(web.app, client=("127.0.0.1", 5000))

    response = client.post("/chat/stream", json={"message": "Hi"})
    events = _sse_events(response)

    assert response.headers["content-type"].startswith("text/event-stream")
    assert [e["type"] for e in events] == ["start", "token", "token", "done"]
    assert "".join(e["text"] for e in events if e["type"] == "token") == "Hello"
    assert events[-1]["answer"] == "Hello"

    saved = memory.conversation_messages(events[0]["conversation_id"])
    assert [(m.role, m.content) for m in saved] == [
        ("user", "Hi"),
        ("assistant", "Hello"),
    ]


def test_chat_stream_rejects_blank_message(tmp_path, monkeypatch):
    memory = MemoryManager(store=MemoryStore(db_path=str(tmp_path / "b.db")))
    monkeypatch.setattr(
        web, "gateway", Gateway(coordinator=StreamingCoordinator(memory))
    )
    client = TestClient(web.app, client=("127.0.0.1", 5000))

    assert client.post("/chat/stream", json={"message": "   "}).status_code == 400


def test_delete_conversation_and_memory(tmp_path, monkeypatch):
    from core.memory import MemoryType

    memory = MemoryManager(store=MemoryStore(db_path=str(tmp_path / "d.db")))
    monkeypatch.setattr(
        web, "gateway", Gateway(coordinator=RecordingCoordinator(memory))
    )
    client = TestClient(web.app, client=("127.0.0.1", 5000))

    conversation = memory.create_conversation("u1", title="t")
    memory.add_conversation_message(conversation.id, role="user", content="x")

    # another user cannot delete it
    assert (
        client.delete(f"/conversations/{conversation.id}?user_id=u2").status_code
        == 404
    )
    assert (
        client.delete(f"/conversations/{conversation.id}?user_id=u1").status_code
        == 200
    )
    assert memory.conversation_messages(conversation.id) == []

    saved = memory.remember("Edima likes tea.", memory_type=MemoryType.PREFERENCE)
    assert client.delete(f"/memory/{saved.id}").status_code == 200
    assert client.delete(f"/memory/{saved.id}").status_code == 404
    assert memory.search("tea", limit=5) == []
