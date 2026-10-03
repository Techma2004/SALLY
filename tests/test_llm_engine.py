import threading

import pytest

from core import llm
from core.agent.runtime import AgentRuntime
from core.agent.tools import ToolRegistry
from core.agent.types import AgentSpec, AgentStatus


class FakeModel:
    """Mimics llama_cpp.Llama.create_chat_completion(stream=True)."""

    def __init__(self, words=("a", "b", "c", "d")):
        self.words = words
        self.calls = []
        self.closed = False

    def create_chat_completion(self, **kwargs):
        self.calls.append(kwargs)
        model = self

        def generate():
            try:
                for word in model.words:
                    yield {"choices": [{"delta": {"content": word}}]}
            finally:
                model.closed = True

        return generate()


@pytest.fixture
def fake_model(monkeypatch):
    model = FakeModel()
    monkeypatch.setattr("core.llm.get_llm", lambda: model)
    return model


def test_chat_joins_stream_and_passes_json_mode(fake_model):
    assert llm.chat([{"role": "user", "content": "hi"}]) == "abcd"
    assert "response_format" not in fake_model.calls[0]

    llm.chat([{"role": "user", "content": "hi"}], json_mode=True)
    assert fake_model.calls[1]["response_format"] == {"type": "json_object"}
    assert fake_model.calls[1]["stream"] is True


def test_stream_stops_on_cancel_and_releases_lock(fake_model):
    cancel = threading.Event()
    seen = []

    for token in llm.stream_chat([], cancel=cancel):
        seen.append(token)
        cancel.set()

    assert seen == ["a"]
    assert fake_model.closed
    assert llm._GENERATION_LOCK.acquire(blocking=False)
    llm._GENERATION_LOCK.release()


def test_stream_stops_at_deadline(fake_model):
    assert list(llm.stream_chat([], timeout_s=-1)) == ["a"]


def test_agent_runtime_default_llm_uses_json_mode_and_deadline(fake_model):
    fake_model.words = ('{"action": "final", ', '"answer": "done"}')
    runtime = AgentRuntime(tools=ToolRegistry())
    spec = AgentSpec(name="planning", role="planner", description="d", system_prompt="You plan.")

    result = runtime.run(spec, "plan something")

    assert result.status is AgentStatus.COMPLETE
    assert result.output == "done"
    assert fake_model.calls[0]["response_format"] == {"type": "json_object"}


def test_agent_runtime_repairs_one_invalid_reply():
    replies = iter(
        ["not json at all", '{"action": "final", "answer": "ok"}']
    )
    seen = []

    def fake(messages, *, max_tokens=None, temperature=None):
        seen.append(list(messages))
        return next(replies)

    runtime = AgentRuntime(tools=ToolRegistry(), llm=fake)
    spec = AgentSpec(name="planning", role="planner", description="d", system_prompt="You plan.")

    result = runtime.run(spec, "plan")

    assert result.status is AgentStatus.COMPLETE
    assert result.output == "ok"
    assert "not valid" in seen[1][-1]["content"]


def test_agent_runtime_fails_after_second_invalid_reply():
    runtime = AgentRuntime(
        tools=ToolRegistry(),
        llm=lambda messages, **kw: "still not json",
    )
    spec = AgentSpec(name="planning", role="planner", description="d", system_prompt="You plan.")

    assert runtime.run(spec, "plan").status is AgentStatus.FAILED
