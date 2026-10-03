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
