import pytest


@pytest.fixture(autouse=True)
def _no_real_llm(monkeypatch):
    def _blocked(*args, **kwargs):
        raise RuntimeError("Real LLM call in a test; monkeypatch core.llm.chat")

    monkeypatch.setattr("core.llm.chat", _blocked)
