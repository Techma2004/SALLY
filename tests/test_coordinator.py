from core.agent import AgentRuntime, Coordinator
from core.agent.router import RouteType
from core.agent.types import AgentStatus
from core.tools import create_tool_registry


def test_coordinator_routes_calculation_to_tool():
    tools = create_tool_registry()
    runtime = AgentRuntime(tools=tools)
    coordinator = Coordinator(runtime=runtime)

    result = coordinator.run("Calculate 123 * 456")

    assert result.status is AgentStatus.COMPLETE
    assert result.output == "56088"


def test_coordinator_routes_coding_to_agent():
    responses = iter([
        '{"action":"final","answer":"Coding task handled."}',
    ])

    def fake_llm(messages, *, max_tokens=None, temperature=None):
        return next(responses)

    tools = create_tool_registry()
    runtime = AgentRuntime(tools=tools, llm=fake_llm)
    coordinator = Coordinator(runtime=runtime)

    result = coordinator.run("Debug this Python function")

    assert result.status is AgentStatus.COMPLETE
    assert result.agent_name == "coding"
    assert result.output == "Coding task handled."


def test_coordinator_describes_tool_route():
    coordinator = Coordinator(
        runtime=AgentRuntime(tools=create_tool_registry())
    )

    description = coordinator.describe_route("Calculate 25 * 4")

    assert "tool" in description
    assert "calculator" in description


def test_coordinator_describes_agent_route():
    coordinator = Coordinator(
        runtime=AgentRuntime(tools=create_tool_registry())
    )

    description = coordinator.describe_route("Plan a new SALLY feature")

    assert "agent" in description
    assert "planning" in description


def test_coordinator_passes_relevant_memories_to_agent(tmp_path):
    from core.memory import MemoryManager, MemoryStore, MemoryType

    memory = MemoryManager(
        store=MemoryStore(
            db_path=str(tmp_path / "test_memory.db")
        )
    )

    memory.remember(
        "Edima prefers professional sleek interfaces.",
        memory_type=MemoryType.PREFERENCE,
    )

    captured = {}

    def fake_llm(messages, *, max_tokens=None, temperature=None):
        captured["messages"] = messages
        return '{"action":"final","answer":"Memory context received."}'

    tools = create_tool_registry()
    runtime = AgentRuntime(tools=tools, llm=fake_llm)

    coordinator = Coordinator(
        runtime=runtime,
        memory=memory,
    )

    result = coordinator.run(
        "Plan a professional interface for SALLY."
    )

    assert result.status is AgentStatus.COMPLETE
    assert result.output == "Memory context received."

    messages = captured["messages"]
    combined = "\n".join(message["content"] for message in messages)

    assert "Edima prefers professional sleek interfaces." in combined


def test_coordinator_calculates_natural_multiplication():
    coordinator = Coordinator()

    result = coordinator.run(
        "Could you multiply 25 by 40?"
    )

    assert result.status is AgentStatus.COMPLETE
    assert result.output == "1000"


def test_coordinator_passes_science_result_through_inference():
    from core.agent.inference import InferenceEngine

    captured = {}

    def fake_llm(messages, *, max_tokens=None, temperature=None):
        captured["messages"] = messages
        return "72 kilometres per hour is exactly 20 metres per second."

    coordinator = Coordinator(
        inference=InferenceEngine(llm=fake_llm),
    )

    result = coordinator.run(
        "Convert 72 km/h to m/s"
    )

    assert result.status is AgentStatus.COMPLETE
    assert result.agent_name == "inference"
    assert (
        result.output
        == "72 kilometres per hour is exactly 20 metres per second."
    )

    combined = "\n".join(
        message["content"]
        for message in captured["messages"]
    )

    assert "unit_conversion" in combined
    assert "20.0" in combined


def _chat_coordinator(tmp_path):
    from core.memory import MemoryManager, MemoryStore, MemoryType

    memory = MemoryManager(
        store=MemoryStore(db_path=str(tmp_path / "test_memory.db"))
    )
    memory.remember(
        "Edima prefers professional sleek interfaces.",
        memory_type=MemoryType.PREFERENCE,
    )

    return Coordinator(
        runtime=AgentRuntime(
            tools=create_tool_registry(),
            llm=lambda *args, **kwargs: "",
        ),
        memory=memory,
    )


def test_coordinator_chat_uses_memories_and_trimmed_history(
    tmp_path,
    monkeypatch,
):
    captured = {}

    def fake_chat(messages, *, max_tokens=None, temperature=None, **_):
        captured["messages"] = messages
        return "Sleek and professional."

    monkeypatch.setattr("core.llm.chat", fake_chat)

    history = [{"role": "user", "content": f"q{i}"} for i in range(12)]
    history.insert(11, {"role": "system", "content": "injected"})

    result = _chat_coordinator(tmp_path).run(
        "Which interfaces do I prefer?",
        history=history,
    )

    assert result.status is AgentStatus.COMPLETE
    assert result.agent_name == "conversation"
    assert result.output == "Sleek and professional."

    messages = captured["messages"]

    assert "professional sleek interfaces" in messages[0]["content"]
    assert all(m["content"] != "injected" for m in messages)
    # system prompt + last 8 history entries (1 dropped by role) + new message
    assert len(messages) == 1 + 7 + 1
    assert all(m["content"] not in {"q0", "q1", "q2"} for m in messages)


def test_coordinator_chat_reports_llm_failure(tmp_path):
    # conftest blocks the real LLM, so the chat call raises.
    result = _chat_coordinator(tmp_path).run("Hello there.")

    assert result.status is AgentStatus.FAILED
    assert result.error


def test_coordinator_chat_rejects_empty_reply(tmp_path, monkeypatch):
    monkeypatch.setattr("core.llm.chat", lambda *a, **k: "   ")

    result = _chat_coordinator(tmp_path).run("Hello there.")

    assert result.status is AgentStatus.FAILED
    assert "empty" in result.error.lower()


def test_coordinator_stream_yields_tokens_then_final(tmp_path, monkeypatch):
    def fake_stream(messages, *, max_tokens=None, temperature=None, **_):
        yield "Hi "
        yield "there"

    monkeypatch.setattr("core.llm.stream_chat", fake_stream)

    events = list(_chat_coordinator(tmp_path).stream("Hello there."))

    assert [e["type"] for e in events] == ["token", "token", "final"]
    assert events[-1]["result"].output == "Hi there"
    assert events[-1]["result"].status is AgentStatus.COMPLETE


def test_coordinator_stream_reports_failure_and_non_chat_routes(tmp_path):
    coordinator = _chat_coordinator(tmp_path)

    # conftest blocks the real LLM -> failure surfaces as a final event
    failed = list(coordinator.stream("Hello there."))
    assert failed[-1]["result"].status is AgentStatus.FAILED

    # tool routes are not streamed: a single final event
    tool = list(coordinator.stream("Calculate 25 * 40"))
    assert [e["type"] for e in tool] == ["final"]
    assert tool[0]["result"].output == "1000"
