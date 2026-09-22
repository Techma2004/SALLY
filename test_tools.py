from core.tools import calculate, current_datetime, create_tool_registry


def main() -> None:
    print("=== SALLY TOOL LAYER V3.1 ===")

    print("\nCalculator:")
    for expression in (
        "25 * 40",
        "(100 + 50) / 5",
        "2 ** 8",
    ):
        result = calculate(expression)
        print(f"  {expression} = {result}")

    print("\nDate/time:")
    print(f"  {current_datetime()}")

    print("\nRegistry:")
    registry = create_tool_registry()

    print("  Available:", ", ".join(registry.names()))

    calculator_result = registry.execute(
        __import__("core.agent.types", fromlist=["ToolRequest"]).ToolRequest(
            name="calculator",
            arguments={"expression": "25 * 40"},
        )
    )

    datetime_result = registry.execute(
        __import__("core.agent.types", fromlist=["ToolRequest"]).ToolRequest(
            name="datetime",
            arguments={},
        )
    )

    print(
        f"  calculator → success={calculator_result.success}, "
        f"output={calculator_result.output}"
    )

    print(
        f"  datetime → success={datetime_result.success}, "
        f"output={datetime_result.output}"
    )

    assert calculator_result.success
    assert calculator_result.output == "1000"

    assert datetime_result.success
    assert datetime_result.output

    print("\n✅ TOOL LAYER TEST PASSED")


if __name__ == "__main__":
    main()
