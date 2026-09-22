from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

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


@dataclass(frozen=True)
class MemorySettings:
    enabled: bool
    db_path: str
    search_limit: int


@dataclass(frozen=True)
class ServerSettings:
    host: str
    port: int


@dataclass(frozen=True)
class SallySettings:
    name: str
    version: str


@dataclass(frozen=True)
class Settings:
    sally: SallySettings
    llm: LLMSettings
    agent: AgentSettings
    memory: MemorySettings
    server: ServerSettings


def get_settings() -> Settings:
    return Settings(
        sally=SallySettings(
            name=_env("SALLY_NAME", "SALLY"),
            version=_env("SALLY_VERSION", "0.1.0"),
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
        ),
        memory=MemorySettings(
            enabled=_bool("MEMORY_ENABLED", True),
            db_path=_env("MEMORY_DB_PATH", "memory/memory.db"),
            search_limit=_int("MEMORY_SEARCH_LIMIT", 5),
        ),
        server=ServerSettings(
            host=_env("HOST", "127.0.0.1"),
            port=_int("PORT", 8080),
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
                "user_name": data.get("user_name") or data.get("name") or "Edima",
                "user_handle": data.get("user_handle") or "Edima",
                "user_location": data.get("user_location") or "Calabar, NG",
                "raw": data,
            }
    except (OSError, json.JSONDecodeError):
        pass

    return {
        "user_name": "Edima",
        "user_handle": "Edima",
        "user_location": "Calabar, NG",
        "raw": {},
    }


def load_or_create_facts() -> dict:
    return load_user()
