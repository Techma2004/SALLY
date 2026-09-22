from __future__ import annotations

from pathlib import Path
from threading import Lock
from typing import Any

from llama_cpp import Llama

from core.config import PROJECT_ROOT, settings

_MODEL: Llama | None = None
_MODEL_LOCK = Lock()


def model_path() -> Path:
    path = Path(settings.llm.model_path)

    if not path.is_absolute():
        path = PROJECT_ROOT / path

    if not path.exists():
        raise FileNotFoundError(f"LLM model not found: {path}")

    return path


def get_llm() -> Llama:
    global _MODEL

    if _MODEL is not None:
        return _MODEL

    with _MODEL_LOCK:
        if _MODEL is not None:
            return _MODEL

        config = settings.llm

        _MODEL = Llama(
            model_path=str(model_path()),
            n_ctx=config.n_ctx,
            n_threads=config.n_threads,
            n_batch=config.n_batch,
            n_gpu_layers=config.n_gpu_layers,
            verbose=config.verbose,
        )

    return _MODEL


def chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
) -> str:
    config = settings.llm

    llm = get_llm()

    response: Any = llm.create_chat_completion(
        messages=messages,
        max_tokens=max_tokens if max_tokens is not None else config.max_tokens,
        temperature=(
            temperature if temperature is not None else config.temperature
        ),
    )

    return response["choices"][0]["message"]["content"].strip()
