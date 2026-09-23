from core.agent.router import RouteType, create_router


def test_router_routes_calculation_to_calculator():
    router = create_router()

    route = router.route("Calculate 123 * 456")

    assert route.route_type is RouteType.TOOL
    assert route.target == "calculator"


def test_router_routes_datetime_to_datetime_tool():
    router = create_router()

    route = router.route("What time is it?")

    assert route.route_type is RouteType.TOOL
    assert route.target == "datetime"


def test_router_routes_coding_to_coding_agent():
    router = create_router()

    route = router.route("Debug this Python function")

    assert route.route_type is RouteType.AGENT
    assert route.target == "coding"


def test_router_routes_testing_to_testing_agent():
    router = create_router()

    route = router.route("Review the edge cases")

    assert route.route_type is RouteType.AGENT
    assert route.target == "testing"


def test_router_routes_research_to_research_agent():
    router = create_router()

    route = router.route("Research local AI")

    assert route.route_type is RouteType.AGENT
    assert route.target == "research"


def test_router_routes_general_requests_to_planning():
    router = create_router()

    route = router.route("Plan SALLY development")

    assert route.route_type is RouteType.AGENT
    assert route.target == "planning"
