from core.agent.router import TaskRouter, RouteType, create_router


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


def test_router_detects_natural_multiplication():
    router = TaskRouter()
    route = router.route("Could you multiply 25 by 40?")

    assert route.route_type is RouteType.TOOL
    assert route.target == "calculator"


def test_router_detects_word_arithmetic():
    router = TaskRouter()

    for prompt in (
        "25 times 40",
        "25 plus 40",
        "25 minus 40",
        "25 divided by 40",
    ):
        route = router.route(prompt)

        assert route.route_type is RouteType.TOOL
        assert route.target == "calculator"


def test_router_routes_unit_conversion_to_science():
    router = create_router()

    route = router.route("Convert 72 km/h to m/s")

    assert route.route_type is RouteType.TOOL
    assert route.target == "unit_convert"


def test_router_routes_scientific_constant():
    router = create_router()

    route = router.route("What is the speed of light?")

    assert route.route_type is RouteType.TOOL
    assert route.target == "scientific_constant"


def test_router_routes_scientific_calculation():
    router = create_router()

    route = router.route("Evaluate sqrt(144)")

    assert route.route_type is RouteType.TOOL
    assert route.target == "science_calculate"
