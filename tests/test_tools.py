from core.agent.types import ToolRequest
from core.native import machine_status
from core.tools import calculate, current_datetime, create_tool_registry

def test_calculator_basic_expression():
    assert calculate("25 * 40") == "1000"


def test_calculator_parentheses():
    assert calculate("(100 + 50) / 5") == "30.0"


def test_calculator_power():
    assert calculate("2 ** 8") == "256"


def test_datetime_returns_string():
    result = current_datetime()

    assert isinstance(result, str)
    assert result.strip()


def test_tool_registry_contains_expected_tools():
    registry = create_tool_registry()

    assert registry.has("calculator")
    assert registry.has("datetime")


def test_machine_status_returns_live_snapshot():

    result = machine_status()

    assert isinstance(result, dict)
    assert result["time"]
    assert result["platform"]
    assert result["architecture"]

    assert result["cpu"]["cores"] >= 1
    assert result["memory"]["total_mb"] > 0
    assert 0 <= result["memory"]["used_mb"] <= result["memory"]["total_mb"]
    assert result["uptime_seconds"] >= 0

    battery = result["battery"]
    assert isinstance(battery["available"], bool)

    if battery["available"]:
        assert 0 <= battery["percent"] <= 100
        assert battery["status"]
        assert (
            battery["remaining_minutes"] is None
            or battery["remaining_minutes"] >= 0
        )


def test_machine_status_is_registered():
    registry = create_tool_registry()

    assert registry.has("machine_status")

    result = registry.execute(
        ToolRequest(
            name="machine_status",
            arguments={},
        )
    )

    assert result.success
    assert isinstance(result.output, dict)
