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
