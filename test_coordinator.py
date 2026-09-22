from core.agent import AgentRuntime, Coordinator
from core.agent.router import RouteType
from core.agent.types import AgentStatus
from core.tools import create_tool_registry


def main() -> None:
    print("=== SALLY V3.4 COORDINATOR ===")

    tools = create_tool_registry()
    runtime = AgentRuntime(tools=tools)
    coordinator = Coordinator(runtime=runtime)

    print("\n--- Calculator request ---")

    objective = "Calculate 123 * 456."

    print("Request:", objective)
    print("Route  :", coordinator.describe_route(objective))

    result = coordinator.run(objective)

    print("Status :", result.status.value)
    print("Output :", result.output)

    assert result.status is AgentStatus.COMPLETE
    assert result.output == "56088"

    print("\n--- Datetime request ---")

    objective = "What time is it?"

    print("Request:", objective)
    print("Route  :", coordinator.describe_route(objective))

    result = coordinator.run(objective)

    print("Status :", result.status.value)
    print("Output :", result.output)

    assert result.status is AgentStatus.COMPLETE
    assert result.output

    print("\n--- Coding request ---")

    objective = "Help me debug this Python function."

    print("Request:", objective)
    print("Route  :", coordinator.describe_route(objective))

    assert (
        coordinator.router.route(objective).route_type
        is RouteType.AGENT
    )

    assert coordinator.choose_agent(objective) == "coding"

    print("Selected: coding")

    print("\n✅ COORDINATOR ROUTING TEST PASSED")


if __name__ == "__main__":
    main()
