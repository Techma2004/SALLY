"""Regression tests taken from a real conversation with SALLY."""

import pytest

from core.agent.coordinator import Coordinator
from core.agent.phrasing import phrase_tool_answer
from core.agent.router import RouteType, create_router
from core.agent.runtime import AgentRuntime
from core.agent.types import AgentStatus
from core.agent.understanding import extract_memory_notes
from core.memory import MemoryManager, MemoryStore, MemoryType
from core.tools import create_tool_registry
from core.version import get_version


@pytest.fixture
def coordinator(tmp_path):
    return Coordinator(
        runtime=AgentRuntime(tools=create_tool_registry()),
        memory=MemoryManager(store=MemoryStore(db_path=str(tmp_path / "r.db"))),
    )


# ---- she must not invent abilities or identity ----

def test_what_can_you_do_is_answered_truthfully_without_the_model(coordinator):
    # conftest blocks the model: this passing proves no model call happened.
    out = coordinator.run("so what can you do").output

    assert "maths" in out and "memory" in out.lower()
    for invented in ("appointment", "reminder", "schedul", "tasks"):
        assert invented not in out.lower()


def test_identity_and_version_are_facts_not_guesses(coordinator):
    assert coordinator.run("what is your name").output.startswith("I'm SALLY")
    assert coordinator.run("what version are you").output == f"I'm SALLY v{get_version()}."


def test_small_talk_is_instant_and_uses_your_name(coordinator):
    coordinator.run("My name is Edima")

    for text in ("hi", "Thanks!", "bye sally", "i'll see you tomorrow"):
        result = coordinator.run(text)
        assert result.agent_name == "self", text
        assert result.status is AgentStatus.COMPLETE

    assert "Edima" in coordinator.run("bye sally").output


def test_small_talk_does_not_hijack_real_requests():
    router = create_router()

    for text in (
        "hello, can you write me a story",
        "thanks, now what time is it",
        "bye the way what is 2+2",
    ):
        assert router.route(text).route_type is not RouteType.SELF, text


# ---- latency: the prompt must stay small ----

def test_history_is_trimmed_to_a_budget(coordinator):
    history = [{"role": "user", "content": "x" * 5000} for _ in range(8)]
    history.append({"role": "assistant", "content": "the latest reply"})

    messages = coordinator._chat_messages("next question", history=history)
    body = [m for m in messages if m["role"] != "system"]

    assert body[-1] == {"role": "user", "content": "next question"}
    assert body[-2]["content"] == "the latest reply"  # newest is always kept
    assert sum(len(m["content"]) for m in body) <= 2400 + 700 + len("next question")
    assert all(len(m["content"]) <= 701 for m in body[:-1])


# ---- she must not dump logs as "what I know about you" ----

def test_recall_ignores_daily_brief_logs(coordinator):
    coordinator.memory.remember(
        "Daily brief 2026-08-29: # SALLY Daily Brief\n" + "weather " * 200,
        memory_type=MemoryType.DAILY,
    )
    coordinator.run("My name is Edima")

    out = coordinator.run("what do you know about me").output

    assert "Daily brief" not in out
    assert "The user's name is Edima." in out
    assert len(out) < 300


def test_chat_prompt_excludes_logs_but_knows_you(coordinator):
    coordinator.memory.remember("Daily brief " + "x" * 400, memory_type=MemoryType.DAILY)
    coordinator.run("My name is Edima")

    block = coordinator._chat_memories("weather")

    assert block == "- The user's name is Edima."


# ---- memory extraction edge cases from real typing ----

def test_typos_and_messy_places_do_not_pollute_memory():
    assert extract_memory_notes("i love in calabar located in Nigeria") == []

    notes = extract_memory_notes("i live in calabar located in Nigeria")
    assert [n.content for n in notes] == ["The user lives in Calabar, Nigeria."]

    assert [n.content for n in extract_memory_notes("i live in port harcourt")] == [
        "The user lives in Port Harcourt."
    ]


# ---- more natural ways to ask for the time ----

@pytest.mark.parametrize(
    "text",
    ["what is the time", "sally what is the time", "what's the time?",
     "got the time?", "tell me the time", "what day is it"],
)
def test_natural_time_questions_use_the_tool(text):
    assert create_router().route(text).target == "datetime"


def test_time_words_in_other_topics_stay_conversation():
    router = create_router()

    for text in ("what is the time complexity of quicksort",
                 "what do you think about time travel"):
        assert router.route(text).route_type is RouteType.CHAT, text


# ---- focused machine answers ----

STATUS = {
    "time": "10:00:00", "platform": "linux", "architecture": "x86_64",
    "cpu": {"cores": 4, "percent": 13.3},
    "memory": {"total_mb": 3829, "used_mb": 3062},
    "disks": [{"mount": "/", "filesystem": "ext4", "total_gb": 106.7,
               "used_gb": 83.7, "percent": 78}],
    "network": {"received_mb": 48.8, "transmitted_mb": 13.1, "interfaces": []},
    "battery": {"available": True, "percent": 100, "status": "Full"},
    "uptime_seconds": 95160,
}


@pytest.mark.parametrize(
    "question,expected",
    [
        ("sally disk space left", "You have 23.0 GB free on / (78% of 106.7 GB used)."),
        ("how much ram am I using", "You're using 3062 of 3829 MB of RAM (80%)."),
        ("what's my cpu doing", "The CPU is about 13.3% busy across 4 cores."),
        ("what is my battery", "Battery is at 100% (Full)."),
        ("what's the uptime", "This machine has been up for 1 d 2 h 26 min."),
        ("network usage", "Since boot: 48.8 MB received and 13.1 MB sent."),
    ],
)
def test_machine_questions_get_a_focused_answer(question, expected):
    assert phrase_tool_answer("machine_status", question, {}, STATUS) == expected


def test_a_general_machine_question_still_gets_the_full_report():
    out = phrase_tool_answer("machine_status", "how is my machine doing", {}, STATUS)

    assert out.startswith("Machine status at") and "Battery: 100% (Full)" in out


# ---- coding/planning/research answer naturally, and stream ----

def test_text_only_agents_stream_in_one_natural_pass(coordinator, monkeypatch):
    captured = {}

    def fake_stream(messages, **kwargs):
        captured["system"] = messages[0]["content"]
        captured["json_mode"] = kwargs.get("json_mode", False)
        yield "def reverse(s):\n"
        yield "    return s[::-1]"

    monkeypatch.setattr("core.llm.stream_chat", fake_stream)

    events = list(coordinator.stream("Write a Python function that reverses a string"))

    assert [e["type"] for e in events] == ["token", "token", "final"]
    final = events[-1]["result"]
    assert final.agent_name == "coding"
    assert final.output == "def reverse(s):\n    return s[::-1]"
    assert "Coding Agent" in captured["system"]
    assert captured["json_mode"] is False  # no JSON straitjacket for code


def test_instant_routes_do_not_stream(coordinator):
    events = list(coordinator.stream("thanks"))

    assert [e["type"] for e in events] == ["final"]
    assert events[0]["result"].agent_name == "self"
