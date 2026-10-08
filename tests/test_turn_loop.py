"""The turn loop: tools run in code, the model only puts the result in words."""

import json
import threading

import pytest

from core.agent.coordinator import Coordinator
from core.agent.narrator import numbers_to_keep, reply_keeps
from core.agent.runtime import AgentRuntime
from core.agent.types import AgentStatus
from core.memory import MemoryManager, MemoryStore
from core.tools import create_tool_registry


@pytest.fixture
def coordinator(tmp_path):
    return Coordinator(
        runtime=AgentRuntime(tools=create_tool_registry()),
        memory=MemoryManager(store=MemoryStore(db_path=str(tmp_path / "t.db"))),
    )


class Model:
    """Records every call and replies with a canned sentence."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    def chat(self, messages, **kwargs):
        self.calls.append(messages)
        return self.reply

    def stream(self, messages, **kwargs):
        self.calls.append(messages)
        for word in self.reply.split(" "):
            yield word + " "


@pytest.fixture
def model(monkeypatch):
    def install(reply):
        fake = Model(reply)
        monkeypatch.setattr("core.llm.chat", fake.chat)
        monkeypatch.setattr("core.llm.stream_chat", fake.stream)
        return fake

    return install


# ---------- the model only phrases what happened ----------

def test_the_model_receives_what_happened_as_json(coordinator, model, narration_on):
    fake = model("Twenty-five times forty comes to 1,000.")

    result = coordinator.run("Calculate 25 * 40")

    assert result.output == "Twenty-five times forty comes to 1,000."
    assert result.trace["narrated"] is True

    prompt = fake.calls[0][-1]["content"]
    event = json.loads(prompt.split("What happened (JSON):\n", 1)[1])

    assert event["source"] == "tool" and event["tool"] == "calculator"
    assert event["ok"] is True
    assert event["summary"] == "25 \u00d7 40 = 1,000"
    assert event["arguments"]["expression"]
    assert "you did not do it" in fake.calls[0][0]["content"].lower()


def test_a_reply_that_changes_the_number_is_rejected(coordinator, model, narration_on):
    model("That is about 999.")

    result = coordinator.run("Calculate 25 * 40")

    assert result.output == "25 \u00d7 40 = 1,000"  # exact text wins
    assert result.trace["narrated"] is False
    assert any("rejected" in note for note in result.trace["notes"])


def test_if_the_model_fails_the_exact_answer_is_still_given(coordinator, narration_on):
    # conftest blocks the model: narration fails, the answer must not.
    result = coordinator.run("Calculate 25 * 40")

    assert result.status is AgentStatus.COMPLETE
    assert result.output == "25 \u00d7 40 = 1,000"
    assert any("narration failed" in note for note in result.trace["notes"])


def test_narration_can_be_switched_off(coordinator, model):
    fake = model("should never be used")

    result = coordinator.run("Calculate 25 * 40")  # narrate_tools is off here

    assert result.output == "25 \u00d7 40 = 1,000"
    assert fake.calls == []


def test_long_exact_reports_are_not_rephrased(coordinator, model, narration_on, monkeypatch):
    fake = model("rewritten")
    monkeypatch.setattr(
        "core.native.machine_status",
        lambda: {
            "time": "10:00:00", "platform": "linux", "architecture": "x86_64",
            "cpu": {"cores": 4, "percent": 1.0},
            "memory": {"total_mb": 100, "used_mb": 50},
            "disks": [
                {"mount": "/", "filesystem": "ext4", "total_gb": 106.7, "used_gb": 83.7, "percent": 78},
                {"mount": "/boot/efi", "filesystem": "vfat", "total_gb": 1.0, "used_gb": 0.0, "percent": 1},
            ],
            "network": {"received_mb": 48.8, "transmitted_mb": 13.1, "interfaces": []},
            "battery": {"available": False}, "uptime_seconds": 60,
        },
    )

    result = coordinator.run("How is my machine doing?")

    assert result.output.startswith("Machine status at")
    assert fake.calls == []


def test_a_failed_tool_is_explained_not_just_errored(coordinator, model, narration_on):
    fake = model("I couldn't work that one out. Try writing it like 12 / 4.")

    result = coordinator.run("Calculate 10 / 0")

    assert result.status is AgentStatus.COMPLETE
    assert "couldn't work that one out" in result.output
    event = json.loads(fake.calls[0][-1]["content"].split("(JSON):\n", 1)[1])
    assert event["ok"] is False and event["error"]


def test_memory_saves_are_narrated_but_must_keep_the_name(coordinator, model, narration_on):
    model("Lovely to meet you, Edima, I'll keep that in mind.")
    assert coordinator.run("My name is Edima").output.startswith("Lovely to meet you, Edima")

    model("Nice to meet you!")  # drops the name -> rejected
    assert coordinator.run("My name is Eddy").output.startswith("Got it, I'll call you Eddy")
    assert [m.content for m in coordinator.memory.recent(limit=5)] == [
        "The user's name is Eddy."
    ]


def test_instant_routes_never_call_the_model(coordinator, model, narration_on):
    fake = model("nope")

    for text in ("thanks", "what can you do", "what version are you"):
        assert coordinator.run(text).trace["model_calls"] == 0

    assert fake.calls == []


# ---------- one loop, two ways to read it ----------

def test_run_and_stream_give_the_same_answer(coordinator, model, narration_on):
    model("It comes to 1,000.")

    streamed = list(coordinator.stream("Calculate 25 * 40"))
    final = coordinator.run("Calculate 25 * 40")

    assert [e["type"] for e in streamed] == ["token"] * 4 + ["final"]
    assert streamed[-1]["result"].output == final.output == "It comes to 1,000."


def test_the_trace_records_what_happened(coordinator, model, narration_on):
    model("It comes to 1,000.")

    trace = coordinator.run("Calculate 25 * 40").trace

    assert trace["route"]["type"] == "tool" and trace["route"]["target"] == "calculator"
    assert [p["name"] for p in trace["phases"]] == ["route", "execute", "narrate"]
    assert trace["tool"]["name"] == "calculator" and trace["tool"]["ok"] is True
    assert trace["event"]["summary"] == "25 \u00d7 40 = 1,000"
    assert trace["model_calls"] == 1 and trace["total_ms"] >= 0


def test_cancelling_stops_narration(coordinator, narration_on, monkeypatch):
    cancel = threading.Event()

    def stream(messages, **kwargs):
        assert kwargs["cancel"] is cancel
        yield "It "
        cancel.set()

    monkeypatch.setattr("core.llm.stream_chat", stream)

    events = list(coordinator.stream("Calculate 25 * 40", cancel=cancel))

    assert events[-1]["type"] == "final"
    assert events[-1]["result"].output == "25 \u00d7 40 = 1,000"  # partial reply rejected


def test_the_model_call_budget_is_enforced(coordinator, model, narration_on):
    from core.agent.turn import TurnBudget, TurnTrace

    fake = model("never")
    trace = TurnTrace(TurnBudget(max_seconds=60, max_model_calls=0))

    assert trace.can_call_model() is False
    assert fake.calls == []


# ---------- the guard itself ----------

def test_guard_numbers_and_names():
    assert numbers_to_keep("It's 7:05 PM on Sunday, October 4, 2026 (UTC+0100).") == [
        "7", "05", "4", "2026",
    ]
    assert numbers_to_keep("25 \u00d7 40 = 1,000") == ["1000"]
    assert numbers_to_keep("5 miles is 8.04672 kilometers.") == ["5", "8.04672"]
    assert reply_keeps(["1000"], "That's 1,000.")
    assert not reply_keeps(["1000"], "That's 10,000.")
    assert not reply_keeps(["8.04672"], "About 8.05 km.")
    assert reply_keeps(["Edima"], "hello edima!")


# ---------- one database read per turn, and connections are closed ----------

def test_a_turn_reads_recent_memories_once(coordinator, model, monkeypatch):
    coordinator.run("My name is Edima")
    model("Hey Edima, good to hear from you.")
    reads = []
    original = coordinator.memory.recent

    def counting(limit=10):
        reads.append(limit)
        return original(limit=limit)

    monkeypatch.setattr(coordinator.memory, "recent", counting)

    coordinator.run("Tell me something interesting")

    assert reads == [50]


def test_store_connections_are_closed(tmp_path):
    import sqlite3

    store = MemoryStore(db_path=str(tmp_path / "c.db"))

    with store._connect() as connection:
        connection.execute("SELECT 1")

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")  # closed after the block
