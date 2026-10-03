from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path
from threading import Event, Lock
from typing import TYPE_CHECKING, Any

from core.config import PROJECT_ROOT, settings

if TYPE_CHECKING:
    from llama_cpp import Llama

_MODEL: Llama | None = None
_MODEL_LOCK = Lock()
# llama.cpp contexts are not thread-safe: serialize generation so a second
# request (or an aborted stream) can never run inference concurrently.
_GENERATION_LOCK = Lock()


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

        # Imported lazily: llama-cpp-python is heavy, and the rest of SALLY
        # (tools, memory, web UI, tests) must work without loading it.
        from llama_cpp import Llama

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


def stream_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
    timeout_s: float | None = None,
    cancel: Event | None = None,
) -> Iterator[str]:
    """
    Yield the reply incrementally as the model generates it.

    json_mode  constrain decoding to valid JSON (llama.cpp grammar).
    timeout_s  stop generating after this many seconds (model load excluded).
    cancel     stop generating as soon as the event is set.

    Generation is serialized: llama.cpp contexts are not thread-safe.
    """
    config = settings.llm

    llm: Any = get_llm()

    options: dict[str, Any] = {}

    if json_mode:
        options["response_format"] = {"type": "json_object"}

    deadline = None if timeout_s is None else time.monotonic() + timeout_s

    with _GENERATION_LOCK:
        if cancel is not None and cancel.is_set():
            return

        stream = llm.create_chat_completion(
            messages=messages,
            max_tokens=(
                max_tokens if max_tokens is not None else config.max_tokens
            ),
            temperature=(
                temperature if temperature is not None else config.temperature
            ),
            stream=True,
            **options,
        )

        try:
            for chunk in stream:
                delta = chunk["choices"][0].get("delta", {}).get("content")

                if delta:
                    yield delta

                if cancel is not None and cancel.is_set():
                    break

                if deadline is not None and time.monotonic() > deadline:
                    break
        finally:
            close = getattr(stream, "close", None)

            if close is not None:
                close()


def chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int | None = None,
    temperature: float | None = None,
    json_mode: bool = False,
    timeout_s: float | None = None,
    cancel: Event | None = None,
) -> str:
    return "".join(
        stream_chat(
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
            timeout_s=timeout_s,
            cancel=cancel,
        )
    ).strip()
