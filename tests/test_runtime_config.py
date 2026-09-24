from setuptools import find_packages

from core.config import get_settings


def test_setuptools_discovers_all_runtime_packages():
    packages = set(find_packages(include=["core*"]))

    required = {
        "core",
        "core.agent",
        "core.gateway",
        "core.memory",
        "core.science",
        "core.tools",
    }

    assert required <= packages


def test_default_runtime_configuration_uses_port_5678(monkeypatch):
    monkeypatch.setenv("PORT", "5678")
    settings = get_settings()

    assert settings.server.port == 5678


def test_default_model_configuration_matches_current_runtime(monkeypatch):
    monkeypatch.setenv(
        "LLM_MODEL_PATH",
        "models/llama-3.2-1b-instruct-q4_k_m.gguf",
    )
    settings = get_settings()

    assert (
        settings.llm.model_path
        == "models/llama-3.2-1b-instruct-q4_k_m.gguf"
    )
