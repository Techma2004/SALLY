from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from core.version import get_version

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _env(name: str, default: str) -> str:
    value = os.getenv(name)
    return value.strip() if value is not None else default


def _int(name: str, default: int) -> int:
    try:
        return int(_env(name, str(default)))
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    try:
        return float(_env(name, str(default)))
    except ValueError:
        return default


def _bool(name: str, default: bool) -> bool:
    value = _env(name, str(default)).lower()
    return value in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class LLMSettings:
    model_path: str
    n_ctx: int
    n_threads: int
    n_batch: int
    n_gpu_layers: int
    temperature: float
    max_tokens: int
    verbose: bool


@dataclass(frozen=True)
class AgentSettings:
    max_steps: int
    max_depth: int
    max_subagents: int
    timeout: int
    max_tokens: int
    temperature: float
    # Conversation is warmer and more varied than structured agent output.
    chat_temperature: float = 0.7
    # Experimental: let the local model classify messages the rules missed.
    intent_llm: bool = False
    # Let the model put exact tool/memory results into natural words.
    narrate_tools: bool = True


@dataclass(frozen=True)
class MemorySettings:
    enabled: bool
    db_path: str
    search_limit: int


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int
    api_key: str
    rate_limit_per_minute: int


@dataclass(frozen=True)
class SallySettings:
    name: str
    version: str


@dataclass(frozen=True)
class IntegrationSettings:
    openweather_api_key: str
    news_api_key: str
    default_city: str
    default_country: str


@dataclass(frozen=True)
class Settings:
    sally: SallySettings
    llm: LLMSettings
    agent: AgentSettings
    memory: MemorySettings
    server: ServerSettings
    integrations: IntegrationSettings


def get_settings() -> Settings:
    return Settings(
        sally=SallySettings(
            name=_env("SALLY_NAME", "SALLY"),
            version=get_version(),
        ),
        llm=LLMSettings(
            model_path=_env(
                "LLM_MODEL_PATH",
                "models/llama-3.2-1b-instruct-q4_k_m.gguf",
            ),
            n_ctx=_int("LLM_N_CTX", 2048),
            n_threads=_int("LLM_N_THREADS", 4),
            n_batch=_int("LLM_N_BATCH", 128),
            n_gpu_layers=_int("LLM_N_GPU_LAYERS", 0),
            temperature=_float("LLM_TEMPERATURE", 0.4),
            max_tokens=_int("LLM_MAX_TOKENS", 256),
            verbose=_bool("LLM_VERBOSE", False),
        ),
        agent=AgentSettings(
            max_steps=_int("AGENT_MAX_STEPS", 3),
            max_depth=_int("AGENT_MAX_DEPTH", 2),
            max_subagents=_int("AGENT_MAX_SUBAGENTS", 4),
            timeout=_int("AGENT_TIMEOUT", 120),
            max_tokens=_int("AGENT_MAX_TOKENS", 256),
            temperature=_float("AGENT_TEMPERATURE", 0.4),
            chat_temperature=_float("CHAT_TEMPERATURE", 0.7),
            intent_llm=_bool("INTENT_LLM", False),
            narrate_tools=_bool("NARRATE_TOOLS", True),
        ),
        memory=MemorySettings(
            enabled=_bool("MEMORY_ENABLED", True),
            db_path=_env("MEMORY_DB_PATH", "memory/memory.db"),
            search_limit=_int("MEMORY_SEARCH_LIMIT", 5),
        ),
        server=ServerSettings(
            host=_env("HOST", "127.0.0.1"),
            port=_int("PORT", 5678),
            # When unset, sensitive endpoints are restricted to loopback
            # callers only (see core/gateway/security.py). Set this to a
            # long random value if SALLY needs to be reachable remotely
            # (e.g. for the WhatsApp webhook) and callers must present it
            # via an `Authorization: Bearer <key>` or `X-API-Key` header.
            api_key=_env("SALLY_API_KEY", ""),
            rate_limit_per_minute=_int("RATE_LIMIT_PER_MINUTE", 60),
        ),
        integrations=IntegrationSettings(
            openweather_api_key=_env("OPENWEATHER_API_KEY", ""),
            news_api_key=_env("NEWS_API_KEY", ""),
            default_city=_env("DEFAULT_CITY", "Calabar"),
            default_country=_env("DEFAULT_COUNTRY", "NG"),
        ),
    )


settings = get_settings()


def load_user() -> dict:
    """Load persistent user profile data from human.json."""
    human_path = PROJECT_ROOT / "memory" / "core" / "human.json"

    try:
        if human_path.exists():
            data = json.loads(human_path.read_text(encoding="utf-8"))
            return {
                "user_name": data.get("user_name") or data.get("name") or "User",
                "user_handle": data.get("user_handle") or "User",
                "user_location": data.get("user_location") or "Unknown",
                "raw": data,
            }
    except (OSError, json.JSONDecodeError):
        pass

    return {
        "user_name": "User",
        "user_handle": "User",
        "user_location": "Unknown",
        "raw": {},
    }


def load_or_create_facts() -> dict:
    return load_user()
