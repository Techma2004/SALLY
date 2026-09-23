from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

from core.config import settings
from core.llm import chat as llm_chat


class InferenceEngine:
    """
    Hidden result-to-language bridge.

    Converts verified tool evidence into a natural SALLY response.
    The interface never needs to know that this intermediate step exists.
    """

    def __init__(self, llm: Callable[..., str] | None = None) -> None:
        self.llm = llm or llm_chat

    def compose(
        self,
        objective: str,
        *,
        tool_name: str,
        evidence: Any,
        context: dict[str, Any] | None = None,
    ) -> str:
        context_text = ""

        if context:
            context_text = (
                "\nAdditional context:\n"
                + "\n".join(
                    f"{key}: {value}"
                    for key, value in context.items()
                )
            )

        evidence_text = json.dumps(
            evidence,
            ensure_ascii=False,
            indent=2,
            default=str,
        )

        messages = [
            {
                "role": "system",
                "content": (
                    "You are SALLY's response composer. "
                    "Use the verified evidence supplied below to "
                    "answer the user's request naturally and clearly. "
                    "Do not invent or alter numerical results. "
                    "Explain relevant units or meaning when useful. "
                    "Answer directly without greetings, filler, or "
                    "repeating the user's request unnecessarily. "
                    "Do not mention internal tools, inference engines, "
                    "routing, or hidden processing unless the user "
                    "explicitly asks about them."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User request:\n{objective}\n\n"
                    f"Verified result from `{tool_name}`:\n"
                    f"{evidence_text}"
                    f"{context_text}\n\n"
                    "Write the final answer for the user."
                ),
            },
        ]

        answer = self.llm(
            messages,
            max_tokens=min(settings.agent.max_tokens, 192),
            temperature=min(settings.agent.temperature, 0.3),
        ).strip()

        if not answer:
            raise ValueError("Inference engine produced an empty response.")

        return answer
