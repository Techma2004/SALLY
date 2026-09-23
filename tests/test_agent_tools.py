from core.agent import AgentRuntime
from core.agent.types import AgentSpec, AgentStatus, StepType
from core.tools import create_tool_registry


def main() -> None:
    print("=== SALLY V3.2 REAL LLM TOOL INTEGRATION ===")

    tools = create_tool_registry()

    spec = AgentSpec(
        name="calculator-agent",
        role="Calculator Agent",
        description="Uses the calculator when arithmetic is required.",
        system_prompt=(
            "You are SALLY's Calculator Agent. "
            "When the objective requires arithmetic, you MUST use "
            "the calculator tool. After receiving the result, provide "
            "the final answer."
        ),
        allowed_tools=("calculator",),
        max_steps=3,
    )

    objective = "Calculate 123 * 456 and tell me the result."

    print("\nObjective:")
    print(objective)

    print("\nRunning real Llama agent...")
    runtime = AgentRuntime(tools=tools)

    result = runtime.run(spec, objective)

    print("\nStatus :", result.status.value)
    print("Steps  :", result.steps)
    print("Agent  :", result.agent_name)

    print("\nExecution history:")
    for step in result.history:
        print(
            f"- step={step.number} "
            f"type={step.step_type.value}"
        )

        if step.tool_request:
            print(
                f"  tool: {step.tool_request.name}"
            )

        if step.tool_result:
            print(
                f"  success: {step.tool_result.success}"
            )
            print(
                f"  output: {step.tool_result.output}"
            )

        if step.output:
            print(
                f"  output: {step.output}"
            )

    print("\nFinal result:")
    print(result.output)

    if result.error:
        print("\nError:")
        print(result.error)

    assert result.status is AgentStatus.COMPLETE
    assert any(
        step.step_type is StepType.TOOL
        for step in result.history
    ), "Agent never used the calculator."

    assert any(
        step.step_type is StepType.OBSERVE
        for step in result.history
    ), "Agent never observed the tool result."

    assert any(
        step.step_type is StepType.FINAL
        for step in result.history
    ), "Agent never produced a final answer."

    assert "56088" in result.output, (
        f"Expected 56088 in final answer, got: {result.output!r}"
    )

    print("\n✅ REAL LLM TOOL INTEGRATION PASSED")


if __name__ == "__main__":
    main()
