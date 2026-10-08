from dataclasses import replace

import pytest


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    """Never load the real GGUF in tests.

    Only the model loader is blocked, so everything above it (chat,
    stream_chat, the agent runtime) still runs and can be exercised with a
    fake model via monkeypatch.setattr("core.llm.get_llm", ...).
    """

    def _blocked(*args, **kwargs):
        raise RuntimeError("Real LLM load blocked in tests; patch core.llm.get_llm")

    monkeypatch.setattr("core.llm.get_llm", _blocked)


@pytest.fixture(autouse=True)
def _exact_answers_by_default(monkeypatch):
    """Tool results are returned exactly unless a test turns narration on."""
    from core.config import settings

    monkeypatch.setattr(
        "core.config.settings",
        replace(settings, agent=replace(settings.agent, narrate_tools=False)),
    )


@pytest.fixture
def narration_on(monkeypatch):
    from core import config

    monkeypatch.setattr(
        "core.config.settings",
        replace(config.settings, agent=replace(config.settings.agent, narrate_tools=True)),
    )
