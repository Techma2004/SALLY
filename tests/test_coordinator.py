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
