import hashlib
import hmac
import json

from core.gateway.gateway import Gateway
from core.gateway.whatsapp import WhatsAppGateway


def test_whatsapp_signature_verification(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "test-secret")

    gateway = WhatsAppGateway(Gateway())
    body = b'{"hello":"world"}'

    digest = hmac.new(
        b"test-secret",
        body,
        hashlib.sha256,
    ).hexdigest()

    assert gateway.verify_signature(
        body,
        f"sha256={digest}",
    )

    assert not gateway.verify_signature(
        body,
        "sha256=invalid",
    )


def test_whatsapp_extracts_text_messages():
    gateway = WhatsAppGateway(Gateway())

    payload = {
        "entry": [
            {
                "changes": [
                    {
                        "value": {
                            "messages": [
                                {
                                    "id": "msg-123",
                                    "from": "2340000000000",
                                    "type": "text",
                                    "text": {
                                        "body": "Hello SALLY"
                                    },
                                }
                            ]
                        }
                    }
                ]
            }
        ]
    }

    messages = gateway.extract_messages(payload)

    assert len(messages) == 1
    assert messages[0].message_id == "msg-123"
    assert messages[0].sender == "2340000000000"
    assert messages[0].text == "Hello SALLY"


def test_whatsapp_send_text(monkeypatch):
    monkeypatch.setenv("WHATSAPP_ENABLED", "true")
    monkeypatch.setenv("WHATSAPP_ACCESS_TOKEN", "test-token")
    monkeypatch.setenv("WHATSAPP_PHONE_NUMBER_ID", "12345")
    monkeypatch.setenv("WHATSAPP_VERIFY_TOKEN", "verify")
    monkeypatch.setenv("WHATSAPP_APP_SECRET", "secret")
    monkeypatch.setenv("WHATSAPP_GRAPH_VERSION", "test-version")

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {"messages": [{"id": "out-123"}]}

    captured = {}

    def fake_post(url, **kwargs):
        captured["url"] = url
        captured["kwargs"] = kwargs
        return FakeResponse()

    monkeypatch.setattr(
        "core.gateway.whatsapp.requests.post",
        fake_post,
    )

    gateway = WhatsAppGateway(Gateway())

    result = gateway.send_text(
        "2340000000000",
        "Hello from SALLY",
    )

    assert result["messages"][0]["id"] == "out-123"
    assert captured["url"] == (
        "https://graph.facebook.com/test-version/"
        "12345/messages"
    )
    assert captured["kwargs"]["json"]["text"]["body"] == (
        "Hello from SALLY"
    )
