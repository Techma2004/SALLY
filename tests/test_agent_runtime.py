from core.agent import AgentRuntime
from core.agent.tools import ToolRegistry
from core.agent.types import AgentSpec, AgentStatus, StepType


def main() -> None:
    print("=== SALLY AGENT RUNTIME V3 ===")

    tools = ToolRegistry()
    tools.register("echo", lambda text: f"Echo: {text}")

    responses = iter(
        [
            '{"action":"tool","tool":"echo","arguments":{"text":"SALLY V3"}}',
            '{"action":"final","answer":"The tool worked and returned: Echo: SALLY V3"}',
        ]
    )

    def fake_llm(messages, *, max_tokens, temperature):
        return next(responses)

    runtime = AgentRuntime(
        tools=tools,
        llm=fake_llm,
    )

    spec = AgentSpec(
        name="v3-test",
        role="Runtime Test Agent",
        description="Tests the SALLY tool loop.",
        system_prompt=(
            "You are a runtime test agent. "
            "Use the available tool once, then finish."
        ),
        allowed_tools=("echo",),
        max_steps=3,
    )

    result = runtime.run(
        spec,
        "Use the echo tool with the text 'SALLY V3'.",
    )

    print("\nStatus :", result.status.value)
    print("Steps  :", result.steps)
    print("Agent  :", result.agent_name)

    print("\nHistory:")
    for step in result.history:
        print(f"- step={step.number} type={step.step_type.value}")
        if step.output:
            print(f"  output: {step.output}")

    print("\nResult:")
    print(result.output)

    assert result.status is AgentStatus.COMPLETE
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
        step.step_type is StepType.FINAL
        for step in result.history
    )
    assert "Echo: SALLY V3" in result.output

    print("\n✅ V3 TOOL LOOP TEST PASSED")


if __name__ == "__main__":
    main()
