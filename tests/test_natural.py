from dataclasses import replace

import pytest

from core.agent.coordinator import Coordinator
from core.agent.intent_llm import classify
from core.agent.phrasing import phrase_tool_answer
from core.agent.router import RouteType, create_router
from core.agent.runtime import AgentRuntime
from core.agent.types import AgentStatus
from core.agent.understanding import (
    extract_memory_notes,
    is_pure_disclosure,
    recall_kind,
)
from core.memory import MemoryManager, MemoryStore, MemoryType
from core.tools import create_tool_registry


@pytest.fixture
def coordinator(tmp_path):
    return Coordinator(
        runtime=AgentRuntime(tools=create_tool_registry()),
        memory=MemoryManager(store=MemoryStore(db_path=str(tmp_path / "n.db"))),
    )


# ---------- understanding what people say about themselves ----------

@pytest.mark.parametrize(
    "text,expected",
    [
        ("My name is Edima", ["The user's name is Edima."]),
        ("my name is edima", ["The user's name is Edima."]),
        ("Call me Eddy", ["The user's name is Eddy."]),
        ("I live in Calabar.", ["The user lives in Calabar."]),
        ("I'm from Lagos", ["The user is from Lagos."]),
        ("I work as a software engineer", ["The user works as a software engineer."]),
        ("I'm a developer", ["The user is a developer."]),
        ("I love jollof rice", ["The user loves jollof rice."]),
        (
            "remember that my sister's birthday is May 3",
            ["My sister's birthday is May 3."],
        ),
        (
            "my name is Edima and I live in Calabar",
            ["The user's name is Edima.", "The user lives in Calabar."],
        ),
    ],
)
def test_extracts_what_the_user_says_about_themselves(text, expected):
    assert [n.content for n in extract_memory_notes(text)] == expected


@pytest.mark.parametrize(
    "text",
    [
        "I like that answer",
        "I like how you talk",
        "I'm not sure about this",
        "My name is not important",
        "I love you",
        "What is the weather like?",
        "Hello there",
    ],
)
def test_does_not_invent_memories(text):
    assert extract_memory_notes(text) == []


def test_questions_are_not_pure_disclosures():
    assert is_pure_disclosure("My name is Edima")
    assert not is_pure_disclosure("My name is Edima, what time is it?")
    assert not is_pure_disclosure("Where do I live?")


@pytest.mark.parametrize(
    "text,kind",
    [
        ("What's my name?", "name"),
        ("what is my name", "name"),
        ("Who am I?", "name"),
        ("Where do I live?", "location"),
        ("What do I like?", "likes"),
        ("What do you know about me?", "about"),
        ("What is the capital of France?", None),
    ],
)
def test_recognises_questions_about_the_user(text, kind):
    assert recall_kind(text) == kind


def test_router_sends_disclosures_and_recall_to_memory():
    router = create_router()

    assert router.route("My name is Edima").route_type is RouteType.MEMORY
    assert router.route("What's my name?").target == "recall"
    # a question mixed in: handled normally, nothing saved silently
    assert router.route("My name is Edima, what time is it?").target == "datetime"
    assert router.route("I like that answer").route_type is RouteType.CHAT


# ---------- memory in a conversation ----------

def test_telling_sally_your_name_is_remembered_and_recalled(coordinator):
    reply = coordinator.run("My name is Edima")

    assert reply.agent_name == "memory"
    assert reply.output == "Nice to meet you, Edima! I'll remember that."

    saved = coordinator.memory.recent(limit=5)
    assert [(m.content, m.memory_type) for m in saved] == [
        ("The user's name is Edima.", MemoryType.PROFILE)
    ]

    assert coordinator.run("What's my name?").output == "Your name is Edima."


def test_unknown_name_is_asked_for_not_invented(coordinator):
    out = coordinator.run("What's my name?").output

    assert "don't know your name" in out


def test_changing_your_name_replaces_it_and_duplicates_are_not_stored(coordinator):
    coordinator.run("My name is Edima")
    reply = coordinator.run("Actually, call me Eddy")

    assert reply.output == "Got it, I'll call you Eddy from now on."
    names = [m.content for m in coordinator.memory.recent(limit=10)]
    assert names == ["The user's name is Eddy."]

    coordinator.run("I live in Calabar")
    coordinator.run("I live in Calabar")
    places = [m for m in coordinator.memory.recent(limit=10) if "lives in" in m.content]
    assert len(places) == 1


def test_what_do_you_know_about_me_lists_memories(coordinator):
    coordinator.run("My name is Edima")
    coordinator.run("I love jollof rice")

    out = coordinator.run("What do you know about me?").output

    assert out.startswith("Here's what I remember about you:")
    assert "The user's name is Edima." in out
    assert "The user loves jollof rice." in out


def test_chat_always_knows_who_you_are(coordinator, monkeypatch):
    coordinator.run("My name is Edima")
    captured = {}

    def fake_chat(messages, **kwargs):
        captured["system"] = messages[0]["content"]
        return "Hey Edima!"

    monkeypatch.setattr("core.llm.chat", fake_chat)

    result = coordinator.run("Good evening, how was your day")

    assert result.output == "Hey Edima!"
    assert "The user's name is Edima." in captured["system"]


# ---------- natural answers from exact tools ----------

def test_tool_answers_are_phrased_naturally_and_exactly():
    assert phrase_tool_answer("calculator", "", {"expression": "(25) * (40)"}, "1000") == "25 × 40 = 1,000"
    assert phrase_tool_answer("calculator", "", {"expression": "((1)+(2))*(3)+(4)"}, "13") == "(1 + 2) × 3 + 4 = 13"
    assert phrase_tool_answer("calculator", "", {"expression": "((((1)+(2))*(3))+(4))"}, "13") == "That comes to 13."
    assert phrase_tool_answer(
        "unit_convert", "",
        {}, {"input": {"value": 5.0, "unit": "miles"}, "output": {"value": 8.04672, "unit": "kilometers"}},
    ) == "5 miles is 8.04672 kilometers."
    assert phrase_tool_answer(
        "scientific_constant", "", {},
        {"description": "Speed of light in vacuum.", "value": 299792458, "unit": "m/s", "symbol": "c"},
    ) == "Speed of light in vacuum: 299,792,458 m/s (c)."
    assert phrase_tool_answer(
        "science_calculate", "", {}, {"expression": "sqrt(144)", "result": 12.0}
    ) == "sqrt(144) = 12"


def test_time_and_date_answers():
    stamp = "2026-10-04 19:05:09 UTC (UTC+0100)"

    assert phrase_tool_answer("datetime", "What time is it?", {}, stamp) == (
        "It's 7:05 PM on Sunday, October 4, 2026 (UTC+0100)."
    )
    assert phrase_tool_answer("datetime", "What's today's date?", {}, stamp) == (
        "Today is Sunday, October 4, 2026."
    )


def test_every_tool_example_gets_a_natural_answer(coordinator):
    from core import help as help_module

    for tool, examples in help_module.TOOL_EXAMPLES.items():
        if tool == "machine_status":
            continue
        for example in examples:
            result = coordinator.run(example)

            assert result.status is AgentStatus.COMPLETE, example
            assert not result.output.startswith("{"), (example, result.output)


# ---------- optional LLM intent classification ----------

def test_llm_intent_maps_labels_and_falls_back_to_chat():
    assert classify("got the time?", chat=lambda *a, **k: '{"intent": "time"}').target == "datetime"
    assert classify("x", chat=lambda *a, **k: '{"intent": "chat"}') is None
    assert classify("x", chat=lambda *a, **k: "not json") is None
    assert classify("x", chat=lambda *a, **k: '{"intent": "launch missiles"}') is None


def test_llm_intent_is_off_by_default_and_used_when_enabled(coordinator, monkeypatch):
    from core.config import settings

    def fake_chat(messages, **kwargs):
        if kwargs.get("json_mode"):
            return '{"intent": "time"}'
        return "Just chatting."

    monkeypatch.setattr("core.llm.chat", fake_chat)

    # off: a vague phrasing the rules don't catch is plain chat
    assert coordinator.run("got a sec to tell me when it is?").output == "Just chatting."

    enabled = replace(settings, agent=replace(settings.agent, intent_llm=True))
    monkeypatch.setattr("core.config.settings", enabled)

    on = coordinator.run("got a sec to tell me when it is?")
    assert on.agent_name == "datetime"
    assert on.output.startswith(("It's ", "Today is "))


def test_llm_intent_falls_back_to_chat_when_the_tool_cannot_run(coordinator, monkeypatch):
    from core.config import settings

    def fake_chat(messages, **kwargs):
        return '{"intent": "math"}' if kwargs.get("json_mode") else "Hello! 👋"

    monkeypatch.setattr("core.llm.chat", fake_chat)
    monkeypatch.setattr(
        "core.config.settings",
        replace(settings, agent=replace(settings.agent, intent_llm=True)),
    )

    result = coordinator.run("hello friend")

    assert result.status is AgentStatus.COMPLETE
    assert result.output == "Hello! 👋"
