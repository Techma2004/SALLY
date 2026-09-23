import core.llm


def test_chat_with_mocked_llm(monkeypatch):
    calls = {}

    def fake_chat(messages, *, max_tokens=None, temperature=None):
        calls["messages"] = messages
        calls["max_tokens"] = max_tokens
        calls["temperature"] = temperature
        return "SALLY test response"

    monkeypatch.setattr(core.llm, "chat", fake_chat)

    result = core.llm.chat(
        [{"role": "user", "content": "Hello SALLY"}],
        max_tokens=32,
        temperature=0.2,
    )

    assert result == "SALLY test response"
    assert calls["messages"][0]["content"] == "Hello SALLY"
    assert calls["max_tokens"] == 32
    assert calls["temperature"] == 0.2
