from core.agent import AgentRuntime, Coordinator
from core.agent.types import AgentSpec, AgentStatus
from core.gateway import Gateway
from core.science import convert_units, scientific_calculate
from core.tools.calculator import natural_expression


def test_natural_chained_arithmetic():
    expression = natural_expression(
        "multiply 45 by 67 and add with 100 and divide by 100 "
        "and subtract by 45 * 67"
    )

    assert expression is not None

    coordinator = Coordinator()
    result = coordinator.run(
        "multiply 45 by 67 and add with 100 and divide by 100 "
        "and subtract by 45 * 67"
    )

    assert result.status is AgentStatus.COMPLETE
    assert result.output == "-2983.85"


def test_natural_addition():
    coordinator = Coordinator()

    result = coordinator.run("add 4 and 45")

    assert result.status is AgentStatus.COMPLETE
    assert result.output == "49"


def test_relative_arithmetic_follow_up():
    gateway = Gateway()

    first = gateway.handle(
        "multiply 45 by 67 and add with 100",
        user_id="test-user",
        source="test",
    )

    assert first.answer == "3115"

    second = gateway.handle(
        "divide by 56",
        user_id="test-user",
        source="test",
    )

    assert second.answer == "55.625"


def test_natural_unit_question_parsing():
    coordinator = Coordinator()

    args = coordinator._tool_arguments(
        "unit_convert",
        "How many meters are in 5 miles?",
    )

    assert args["value"] == 5.0
    assert args["from_unit"] == "miles"
    assert args["to_unit"] == "meters"


def test_milliseconds_conversion():
    result = convert_units(
        556,
        "ms",
        "minutes",
    )

    assert result["verified"] is True
    assert abs(
        result["output"]["value"] - (556 / 60000)
    ) < 1e-12


def test_runtime_cleans_completed_tasks():
    runtime = AgentRuntime(
        llm=lambda messages, **kwargs: (
            '{"action":"answer","answer":"done"}'
        )
    )

    spec = AgentSpec(
        name="cleanup-test",
        role="Cleanup Test",
        description="Runtime cleanup test.",
        system_prompt="You are a test agent.",
    )

    result = runtime.run(spec, "Finish this test.")

    assert result.status is AgentStatus.COMPLETE
    assert runtime.active_tasks == {}


def test_runtime_declares_capability_boundary():
    runtime = AgentRuntime()

    spec = AgentSpec(
        name="boundary-test",
        role="Boundary Test",
        description="Capability boundary test.",
        system_prompt="You are a test agent.",
    )

    prompt = runtime._system_prompt(spec)

    assert "Never invent URLs" in prompt
    assert "external action" in prompt
    assert "registered tool" in prompt
