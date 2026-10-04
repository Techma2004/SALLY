import sys
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from core import help as help_module
from core.agent.coordinator import Coordinator
from core.agent.router import RouteType, create_router
from core.agent.types import ToolRequest
from core.gateway import web
from core.native import describe_machine
from core.tools import create_tool_registry


def _tool_examples():
    return [
        (name, example)
        for name, examples in help_module.TOOL_EXAMPLES.items()
        for example in examples
    ]


def _agent_examples():
    return [
        (agent["name"], example)
        for agent in help_module.AGENT_GUIDE
        for example in agent["examples"]
    ]


@pytest.mark.parametrize("tool,example", _tool_examples())
def test_every_tool_example_routes_to_its_tool(tool, example):
    route = create_router().route(example)

    assert route.route_type is RouteType.TOOL, example
    assert route.target == tool, example


@pytest.mark.parametrize("tool,example", _tool_examples())
def test_every_tool_example_actually_runs(tool, example):
    if tool == "machine_status":
        pytest.importorskip("sally_native")

    registry = create_tool_registry()
    arguments = Coordinator._tool_arguments(tool, example, {})

    result = registry.execute(ToolRequest(name=tool, arguments=arguments))

    assert result.success, (example, result.error)


@pytest.mark.parametrize("agent,example", _agent_examples())
def test_every_agent_example_routes_to_its_agent(agent, example):
    route = create_router().route(example)

    assert route.route_type is RouteType.AGENT, example
    assert route.target == agent, example


def test_every_registered_tool_has_examples():
    registry = create_tool_registry()

    assert set(registry.names()) == set(help_module.TOOL_EXAMPLES)


@pytest.mark.parametrize(
    "text",
    [
        "What is my name?",
        "How are you?",
        "Tell me about memory in humans",
        "What is the capital of France?",
        "Hello there",
    ],
)
def test_machine_patterns_do_not_hijack_normal_chat(text):
    assert create_router().route(text).route_type is RouteType.CHAT


def test_help_endpoint_lists_tools_agents_and_tips():
    client = TestClient(web.app, client=("127.0.0.1", 5000))

    data = client.get("/help").json()

    assert {t["name"] for t in data["tools"]} >= {"calculator", "machine_status"}
    assert all(t["examples"] for t in data["tools"])
    assert [a["name"] for a in data["agents"]] == [
        "planning", "coding", "testing", "research"
    ]
    assert data["tips"]


def test_describe_machine_is_readable_and_accurate():
    status = {
        "time": "10:00:00", "platform": "linux", "architecture": "x86_64",
        "cpu": {"cores": 4, "percent": 12.3},
        "memory": {"total_mb": 4096, "used_mb": 2048},
        "disks": [{"mount": "/", "filesystem": "ext4",
                   "total_gb": 100.0, "used_gb": 75.0, "percent": 75}],
        "network": {"received_mb": 5.0, "transmitted_mb": 2.0,
                    "interfaces": ["wlan0"]},
        "battery": {"available": True, "percent": 80,
                    "status": "Discharging", "remaining_minutes": 120},
        "uptime_seconds": 93784,
    }

    text = describe_machine(status)

    assert "CPU: 12.3% busy across 4 cores" in text
    assert "Memory: 2048 / 4096 MB (50%)" in text
    assert "Disk /: 75% of 100.0 GB used (25.0 GB free)" in text
    assert "5.0 MB received, 2.0 MB sent" in text
    assert "Battery: 80% (Discharging, about 120 min left)" in text
    assert "Uptime: 1 d 2 h 3 min" in text
    assert "{" not in text  # never a raw dict


def test_machine_status_tool_answer_is_formatted_text(monkeypatch, tmp_path):
    status = {
        "time": "10:00:00", "platform": "linux", "architecture": "x86_64",
        "cpu": {"cores": 4, "percent": 1.0},
        "memory": {"total_mb": 100, "used_mb": 50},
        "disks": [], "network": {"received_mb": 0.0, "transmitted_mb": 0.0,
                                 "interfaces": []},
        "battery": {"available": False}, "uptime_seconds": 60,
    }
    monkeypatch.setattr("core.native.machine_status", lambda: status)

    from core.agent.runtime import AgentRuntime
    from core.memory import MemoryManager, MemoryStore

    coordinator = Coordinator(
        runtime=AgentRuntime(tools=create_tool_registry()),
        memory=MemoryManager(store=MemoryStore(db_path=str(tmp_path / "m.db"))),
    )

    result = coordinator.run("How is my machine doing?")

    assert result.agent_name == "machine_status"
    assert result.output.startswith("Machine status at 10:00:00")
    assert "Battery: not available" in result.output
