from __future__ import annotations

import hashlib
import hmac
import os
from dataclasses import dataclass
from typing import Any

import requests

from .gateway import Gateway


@dataclass(frozen=True)
class WhatsAppMessage:
    message_id: str
    sender: str
    text: str


class WhatsAppGateway:
    """
    WhatsApp transport adapter for SALLY.

    This layer handles WhatsApp webhook/API concerns only.
    SALLY reasoning remains inside the unified Gateway.
    """

    def __init__(self, sally_gateway: Gateway) -> None:
        self.sally_gateway = sally_gateway

    @property
    def enabled(self) -> bool:
        return os.getenv("WHATSAPP_ENABLED", "false").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    @property
    def access_token(self) -> str:
        return os.getenv("WHATSAPP_ACCESS_TOKEN", "").strip()

    @property
    def phone_number_id(self) -> str:
        return os.getenv("WHATSAPP_PHONE_NUMBER_ID", "").strip()

    @property
    def verify_token(self) -> str:
        return os.getenv("WHATSAPP_VERIFY_TOKEN", "").strip()

    @property
    def app_secret(self) -> str:
        return os.getenv("WHATSAPP_APP_SECRET", "").strip()

    @property
    def graph_version(self) -> str:
        return os.getenv("WHATSAPP_GRAPH_VERSION", "").strip()

    def _require_configuration(self) -> None:
        required = {
            "WHATSAPP_ACCESS_TOKEN": self.access_token,
            "WHATSAPP_PHONE_NUMBER_ID": self.phone_number_id,
            "WHATSAPP_VERIFY_TOKEN": self.verify_token,
            "WHATSAPP_APP_SECRET": self.app_secret,
            "WHATSAPP_GRAPH_VERSION": self.graph_version,
        }

        missing = [name for name, value in required.items() if not value]

        if missing:
            raise RuntimeError(
                "WhatsApp gateway is not fully configured. "
                f"Missing: {', '.join(missing)}"
            )

    def verify_webhook(
        self,
        mode: str | None,
        verify_token: str | None,
        challenge: str | None,
    ) -> str:
        if not self.enabled:
            raise RuntimeError("WhatsApp gateway is disabled.")

        if (
            mode != "subscribe"
            or not verify_token
            or not challenge
            or not hmac.compare_digest(
                verify_token,
                self.verify_token,
            )
        ):
            raise ValueError("Invalid WhatsApp webhook verification.")

        return challenge

    def verify_signature(
        self,
        body: bytes,
        signature: str | None,
    ) -> bool:
        if not self.enabled or not self.app_secret:
            return False

        if not signature or not signature.startswith("sha256="):
            return False

        digest = hmac.new(
            self.app_secret.encode("utf-8"),
            body,
            hashlib.sha256,
        ).hexdigest()

        expected = f"sha256={digest}"

        return hmac.compare_digest(signature, expected)

    @staticmethod
    def extract_messages(
        payload: dict[str, Any],
    ) -> list[WhatsAppMessage]:
        messages: list[WhatsAppMessage] = []

        for entry in payload.get("entry", []):
            for change in entry.get("changes", []):
                value = change.get("value", {})

                for message in value.get("messages", []):
                    if message.get("type") != "text":
                        continue

                    message_id = str(message.get("id", "")).strip()
                    sender = str(message.get("from", "")).strip()
                    text = str(
                        message.get("text", {}).get("body", "")
                    ).strip()

                    if not message_id or not sender or not text:
                        continue

                    messages.append(
                        WhatsAppMessage(
                            message_id=message_id,
                            sender=sender,
                            text=text,
                        )
                    )

        return messages

    def send_text(self, recipient: str, text: str) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("WhatsApp gateway is disabled.")

        self._require_configuration()

        url = (
            f"https://graph.facebook.com/"
            f"{self.graph_version}/"
            f"{self.phone_number_id}/messages"
        )

        response = requests.post(
            url,
            headers={
                "Authorization": f"Bearer {self.access_token}",
                "Content-Type": "application/json",
            },
            json={
                "messaging_product": "whatsapp",
                "recipient_type": "individual",
                "to": recipient,
                "type": "text",
                "text": {
                    "preview_url": False,
                    "body": text,
                },
            },
            timeout=15,
        )

        response.raise_for_status()
        return response.json()

    @staticmethod
    def _chunks(text: str, size: int = 3500) -> list[str]:
        text = text.strip()

        if not text:
            return []

        return [
            text[index:index + size]
            for index in range(0, len(text), size)
        ]

    def process_message(self, message: WhatsAppMessage) -> None:
        response = self.sally_gateway.handle(
            message.text,
            user_id=message.sender,
            source="whatsapp",
        )

        if response.answer:
            answer = response.answer
        elif response.error:
            answer = (
                "SALLY could not complete that request."
            )
        else:
            answer = (
                "SALLY could not complete that request."
            )

        for chunk in self._chunks(answer):
            self.send_text(message.sender, chunk)
