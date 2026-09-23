from core.agent import AgentRuntime
from core.agent.tools import ToolRegistry
from core.agent.types import AgentSpec, AgentStatus, StepType


def test_agent_runtime_tool_loop():
    tools = ToolRegistry()
    tools.register("echo", lambda text: f"Echo: {text}")

    responses = iter([
        '{"action":"tool","tool":"echo","arguments":{"text":"hello"}}',
        '{"action":"final","answer":"The tool returned successfully."}',
    ])

    def fake_llm(messages, *, max_tokens=None, temperature=None):
        return next(responses)

    runtime = AgentRuntime(tools=tools, llm=fake_llm)

    spec = AgentSpec(
        name="test-agent",
        role="Test Agent",
        description="Agent runtime test.",
        system_prompt="You are a test agent.",
        allowed_tools=("echo",),
    )

    result = runtime.run(spec, "Test the echo tool.")

    assert result.status is AgentStatus.COMPLETE
    assert result.output == "The tool returned successfully."
    assert result.steps == 2

    assert any(
        step.step_type is StepType.TOOL
        for step in result.history
    )

    assert any(
        step.step_type is StepType.OBSERVE
        for step in result.history
    )

    assert any(
        step.tool_result is not None
        and step.tool_result.success
        and step.tool_result.output == "Echo: hello"
        for step in result.history
    )


def test_agent_runtime_rejects_disallowed_tool():
    tools = ToolRegistry()

    def fake_llm(messages, *, max_tokens=None, temperature=None):
        return (
            '{"action":"tool","tool":"secret",'
            '"arguments":{}}'
        )

    runtime = AgentRuntime(tools=tools, llm=fake_llm)

    spec = AgentSpec(
        name="restricted-agent",
        role="Restricted Agent",
        description="Permission test.",
        system_prompt="You are a restricted test agent.",
        allowed_tools=(),
    )

    result = runtime.run(spec, "Try an unauthorized tool.")

    assert result.status is AgentStatus.FAILED
    assert result.output == ""
    assert result.error is not None
    assert "not allowed" in result.error


def test_agent_runtime_rejects_invalid_action():
    tools = ToolRegistry()

    def fake_llm(messages, *, max_tokens=None, temperature=None):
        return "This is not valid agent JSON."

    runtime = AgentRuntime(tools=tools, llm=fake_llm)

    spec = AgentSpec(
        name="invalid-agent",
        role="Invalid Agent",
        description="Invalid action test.",
        system_prompt="You are a test agent.",
    )

    result = runtime.run(spec, "Return an invalid action.")

    assert result.status is AgentStatus.FAILED
    assert result.output == ""
    assert result.error is not None
    assert "Invalid agent action" in result.error
