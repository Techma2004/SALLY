from core.agent.router import RouteType, create_router


def main() -> None:
    print("=== SALLY V3.3 TASK ROUTER ===")

    router = create_router()

    tests = [
        (
            "Calculate 123 * 456",
            RouteType.TOOL,
            "calculator",
        ),
        (
            "What time is it?",
            RouteType.TOOL,
            "datetime",
        ),
        (
            "Help me debug this Python function.",
            RouteType.AGENT,
            "coding",
        ),
        (
            "Review this code for edge cases.",
            RouteType.AGENT,
            "testing",
        ),
        (
            "Research lightweight local AI models.",
            RouteType.AGENT,
            "research",
        ),
        (
            "Plan my next SALLY development phase.",
            RouteType.AGENT,
            "planning",
        ),
    ]

    for objective, expected_type, expected_target in tests:
        route = router.route(objective)

        print(f"\nRequest: {objective}")
        print(f"  Type      : {route.route_type.value}")
        print(f"  Target    : {route.target}")
        print(f"  Confidence: {route.confidence}")
        print(f"  Reason    : {route.reason}")

        assert route.route_type is expected_type
        assert route.target == expected_target

    print("\n✅ ROUTER TEST PASSED")


if __name__ == "__main__":
    main()
