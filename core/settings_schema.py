"""Settings that can be changed from the web UI, and the safe way to save them.

The web UI edits ``.env``. Only the keys listed here can be changed; secrets
are write-only (their values are never sent back); anything that decides who
can reach SALLY (host, port, API key) stays editable by hand only. Values are
validated, written atomically, and take effect on the next start.
"""

from __future__ import annotations

import os
import re
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import dotenv_values

from core.config import PROJECT_ROOT

ENV_PATH = PROJECT_ROOT / ".env"
MODELS_DIR = PROJECT_ROOT / "models"

# Changing these from a browser would change who can reach SALLY, so they
# are deliberately not editable here.
LOCKED_KEYS = ("SALLY_API_KEY", "HOST", "PORT", "MEMORY_DB_PATH")


class SettingsError(ValueError):
    """A value was rejected; ``errors`` maps setting key -> message."""

    def __init__(self, errors: dict[str, str]) -> None:
        super().__init__("; ".join(f"{k}: {v}" for k, v in errors.items()))
        self.errors = errors


@dataclass(frozen=True)
class Field:
    key: str
    label: str
    group: str
    kind: str  # int | float | bool | text | model | secret
    help: str = ""
    min: float | None = None
    max: float | None = None
    step: float | None = None
    running: Any = None  # callable(settings) -> value; None means read os.environ


def _attr(path: str):
    def getter(settings: Any) -> Any:
        value: Any = settings
        for part in path.split("."):
            value = getattr(value, part)
        return value

    return getter


FIELDS: tuple[Field, ...] = (
    Field("LLM_MODEL_PATH", "Model", "Model", "model",
          "The GGUF file SALLY thinks with. It must fit in your RAM.",
          running=_attr("llm.model_path")),
    Field("LLM_N_CTX", "Context window (tokens)", "Model", "int",
          "How much conversation the model can read at once. Bigger uses more RAM and is slower.",
          256, 32768, 256, _attr("llm.n_ctx")),
    Field("LLM_N_THREADS", "CPU threads", "Model", "int",
          "Best set to your number of physical CPU cores.",
          1, 64, 1, _attr("llm.n_threads")),
    Field("LLM_N_BATCH", "Batch size", "Model", "int",
          "How many prompt tokens are processed at once. Higher can read prompts faster but uses more RAM.",
          16, 2048, 16, _attr("llm.n_batch")),
    Field("LLM_MAX_TOKENS", "Longest reply (tokens)", "Model", "int",
          "Replies stop after this many tokens.",
          16, 4096, 16, _attr("llm.max_tokens")),
    Field("LLM_PRELOAD", "Load the model at startup", "Model", "bool",
          "Avoids a long wait on the first message, at the cost of keeping the model in RAM."),
    Field("LLM_N_GPU_LAYERS", "GPU layers", "Model", "int",
          "Leave at 0 for CPU-only machines.",
          0, 200, 1, _attr("llm.n_gpu_layers")),

    Field("CHAT_TEMPERATURE", "Chat warmth", "Conversation", "float",
          "Higher is more varied and chatty, lower is more predictable.",
          0, 2, 0.05, _attr("agent.chat_temperature")),
    Field("NARRATE_TOOLS", "Let the model phrase tool results", "Conversation", "bool",
          "Tools always run in code and the model only puts the result in words. If it changes a number, SALLY uses the exact text. Off is faster.",
          running=_attr("agent.narrate_tools")),
    Field("INTENT_LLM", "Smart intent detection (experimental)", "Conversation", "bool",
          "Ask the model to classify messages the rules did not understand. Costs one extra short model call.",
          running=_attr("agent.intent_llm")),
    Field("SALLY_NAME", "Assistant name", "Conversation", "text",
          "", running=_attr("sally.name")),

    Field("AGENT_TIMEOUT", "Time limit per turn (seconds)", "Limits", "int",
          "A turn that takes longer is stopped.", 10, 600, 5, _attr("agent.timeout")),
    Field("AGENT_MAX_TOKENS", "Agent reply length (tokens)", "Limits", "int",
          "", 16, 4096, 16, _attr("agent.max_tokens")),
    Field("AGENT_TEMPERATURE", "Agent temperature", "Limits", "float",
          "", 0, 2, 0.05, _attr("agent.temperature")),
    Field("AGENT_MAX_STEPS", "Agent steps", "Limits", "int",
          "", 1, 10, 1, _attr("agent.max_steps")),
    Field("RATE_LIMIT_PER_MINUTE", "Requests per minute", "Limits", "int",
          "Per client address.", 1, 1000, 1, _attr("server.rate_limit_per_minute")),

    Field("MEMORY_ENABLED", "Remember things", "Memory", "bool",
          "", running=_attr("memory.enabled")),
    Field("MEMORY_SEARCH_LIMIT", "Memories used per message", "Memory", "int",
          "", 1, 20, 1, _attr("memory.search_limit")),

    Field("DEFAULT_CITY", "Default city", "Location", "text",
          "", running=_attr("integrations.default_city")),
    Field("DEFAULT_COUNTRY", "Default country code", "Location", "text",
          "", running=_attr("integrations.default_country")),

    Field("OPENWEATHER_API_KEY", "OpenWeather API key", "Integrations", "secret",
          "For weather. Write-only: saved values are never shown.",
          running=_attr("integrations.openweather_api_key")),
    Field("NEWS_API_KEY", "NewsAPI key", "Integrations", "secret",
          "For news headlines.", running=_attr("integrations.news_api_key")),
    Field("TELEGRAM_BOT_TOKEN", "Telegram bot token", "Integrations", "secret", ""),
    Field("WHATSAPP_ENABLED", "WhatsApp enabled", "Integrations", "bool", ""),
    Field("WHATSAPP_ACCESS_TOKEN", "WhatsApp access token", "Integrations", "secret", ""),
    Field("WHATSAPP_PHONE_NUMBER_ID", "WhatsApp phone number ID", "Integrations", "text", ""),
    Field("WHATSAPP_VERIFY_TOKEN", "WhatsApp verify token", "Integrations", "secret", ""),
    Field("WHATSAPP_APP_SECRET", "WhatsApp app secret", "Integrations", "secret", ""),
)

BY_KEY = {field.key: field for field in FIELDS}
_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}
_SAFE_VALUE = re.compile(r"^[A-Za-z0-9_./:@+\-]*$")


# ------------------------------------------------------------------ files

def model_files() -> list[dict[str, Any]]:
    if not MODELS_DIR.is_dir():
        return []

    return [
        {
            "path": f"models/{path.name}",
            "name": path.name,
            "size_gb": round(path.stat().st_size / 1e9, 2),
        }
        for path in sorted(MODELS_DIR.glob("*.gguf"))
        if path.is_file()
    ]


def read_env(path: Path | None = None) -> dict[str, str]:
    target = path or ENV_PATH

    if not target.exists():
        return {}

    return {k: v for k, v in dotenv_values(target).items() if v is not None}


def _format_for_env(value: str) -> str:
    if _SAFE_VALUE.match(value):
        return value

    if "'" in value or "\\" in value:
        raise SettingsError({"": "Values may not contain quotes or backslashes."})

    return f"'{value}'"


def write_env(updates: dict[str, str], path: Path | None = None) -> None:
    """Update keys in place (keeping comments and order), atomically."""
    target = path or ENV_PATH
    lines = target.read_text(encoding="utf-8").splitlines() if target.exists() else []
    pending = dict(updates)
    output: list[str] = []

    for line in lines:
        match = re.match(r"^\s*([A-Z][A-Z0-9_]*)\s*=", line)

        if match and match.group(1) in pending:
            key = match.group(1)
            output.append(f"{key}={_format_for_env(pending.pop(key))}")
        else:
            output.append(line)

    if pending:
        if output and output[-1].strip():
            output.append("")
        output.append("# Saved from the web settings")
        output.extend(f"{k}={_format_for_env(v)}" for k, v in pending.items())

    handle, temp_name = tempfile.mkstemp(dir=target.parent, prefix=".env.", suffix=".tmp")

    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write("\n".join(output) + "\n")
        os.chmod(temp_name, 0o600)  # it can hold tokens
        os.replace(temp_name, target)
    except BaseException:
        Path(temp_name).unlink(missing_ok=True)
        raise


# ------------------------------------------------------------- validation

def _normalise(kind: str, value: Any) -> str:
    text = str(value).strip()

    if kind == "bool":
        return "true" if text.lower() in _TRUE else "false"

    if kind == "float":
        try:
            return repr(float(text))
        except ValueError:
            return text

    if kind == "int":
        try:
            return str(int(float(text)))
        except ValueError:
            return text

    return text


def validate(updates: dict[str, Any]) -> dict[str, str]:
    """Return {key: string to store}, or raise SettingsError."""
    errors: dict[str, str] = {}
    clean: dict[str, str] = {}

    for key, raw in updates.items():
        field = BY_KEY.get(key)

        if key in LOCKED_KEYS:
            errors[key] = "This setting can only be changed by editing .env."
            continue

        if field is None:
            errors[key] = "Unknown setting."
            continue

        text = "" if raw is None else str(raw).strip()

        if re.search(r"[\x00-\x1f\x7f]", text):
            errors[key] = "Control characters and line breaks are not allowed."
            continue

        if field.kind == "bool":
            if text.lower() in _TRUE:
                clean[key] = "true"
            elif text.lower() in _FALSE:
                clean[key] = "false"
            else:
                errors[key] = "Must be on or off."

        elif field.kind in {"int", "float"}:
            try:
                number = int(text) if field.kind == "int" else float(text)
            except ValueError:
                errors[key] = "Must be a number."
                continue

            if (field.min is not None and number < field.min) or (
                field.max is not None and number > field.max
            ):
                errors[key] = f"Must be between {field.min:g} and {field.max:g}."
                continue

            clean[key] = str(number)

        elif field.kind == "model":
            if text not in {item["path"] for item in model_files()}:
                errors[key] = "Pick a .gguf file from the models folder."
            else:
                clean[key] = text

        elif field.kind == "secret":
            if len(text) > 400:
                errors[key] = "Too long."
            else:
                clean[key] = text

        else:  # text
            if len(text) > 120:
                errors[key] = "Too long."
            elif "'" in text or "\\" in text:
                errors[key] = "Quotes and backslashes are not allowed."
            else:
                clean[key] = text

    if errors:
        raise SettingsError(errors)

    return clean


# ----------------------------------------------------------- description

def _running_value(field: Field, settings: Any) -> Any:
    if field.running is not None:
        return field.running(settings)

    return os.environ.get(field.key, "")


def describe(settings: Any, env: dict[str, str] | None = None) -> dict[str, Any]:
    """Everything the settings page needs. Secret values are never included."""
    saved = read_env() if env is None else env
    groups: dict[str, list[dict[str, Any]]] = {}
    any_pending = False

    for field in FIELDS:
        running = _running_value(field, settings)
        in_file = field.key in saved
        pending = in_file and _normalise(field.kind, saved[field.key]) != _normalise(
            field.kind, running
        )
        any_pending = any_pending or pending

        item: dict[str, Any] = {
            "key": field.key,
            "label": field.label,
            "kind": field.kind,
            "help": field.help,
            "pending": pending,
        }

        if field.min is not None:
            item.update(min=field.min, max=field.max, step=field.step)

        if field.kind == "secret":
            item["configured"] = bool(saved.get(field.key) if in_file else running)
        else:
            shown = saved[field.key] if in_file else running
            item["value"] = _normalise(field.kind, shown)
            item["running"] = _normalise(field.kind, running)

        groups.setdefault(field.group, []).append(item)

    return {
        "groups": [{"name": name, "fields": items} for name, items in groups.items()],
        "models": model_files(),
        "locked": list(LOCKED_KEYS),
        "restart_required": any_pending,
    }


# --------------------------------------------------------------- restart

def restart_process(delay: float = 0.8) -> None:
    """Replace this process with a fresh one, after the response is sent."""
    import sys

    def go() -> None:
        time.sleep(delay)
        os.execv(sys.executable, [sys.executable, *sys.argv])

    threading.Thread(target=go, daemon=True).start()
