import os
import stat

import pytest
from fastapi.testclient import TestClient

from core import settings_schema as schema
from core.config import settings
from core.gateway import web


@pytest.fixture
def env_file(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "# SALLY runtime configuration\n"
        "LLM_N_CTX=2048\n"
        "\n"
        "# keep this comment\n"
        "LLM_TEMPERATURE=0.4\n"
        "OPENWEATHER_API_KEY=old-secret\n"
        "SALLY_API_KEY=locked-secret\n"
    )
    monkeypatch.setattr(schema, "ENV_PATH", path)
    return path


@pytest.fixture
def models(tmp_path, monkeypatch):
    folder = tmp_path / "models"
    folder.mkdir()
    (folder / "qwen.gguf").write_bytes(b"x" * 1000)
    (folder / "notes.txt").write_text("not a model")
    monkeypatch.setattr(schema, "MODELS_DIR", folder)
    return folder


@pytest.fixture
def client():
    return TestClient(web.app, client=("127.0.0.1", 5000))


# ---------------- writing .env safely ----------------

def test_write_keeps_comments_and_order_and_updates_in_place(env_file):
    schema.write_env({"LLM_N_CTX": "1024", "CHAT_TEMPERATURE": "0.9"})

    lines = env_file.read_text().splitlines()
    assert lines[:5] == [
        "# SALLY runtime configuration",
        "LLM_N_CTX=1024",
        "",
        "# keep this comment",
        "LLM_TEMPERATURE=0.4",
    ]
    assert "SALLY_API_KEY=locked-secret" in lines  # untouched
    assert lines[-2:] == ["# Saved from the web settings", "CHAT_TEMPERATURE=0.9"]


def test_env_file_is_private_and_written_atomically(env_file):
    schema.write_env({"LLM_N_CTX": "512"})

    assert stat.S_IMODE(os.stat(env_file).st_mode) == 0o600
    assert not [p for p in env_file.parent.iterdir() if p.name.endswith(".tmp")]


def test_values_with_spaces_are_quoted_and_round_trip(env_file):
    schema.write_env({"DEFAULT_CITY": "Port Harcourt"})

    assert "DEFAULT_CITY='Port Harcourt'" in env_file.read_text()
    assert schema.read_env()["DEFAULT_CITY"] == "Port Harcourt"


def test_a_new_env_file_is_created_if_missing(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    monkeypatch.setattr(schema, "ENV_PATH", path)

    schema.write_env({"LLM_N_CTX": "512"})

    assert path.read_text() == "# Saved from the web settings\nLLM_N_CTX=512\n"


# ---------------- validation ----------------

def test_valid_values_are_normalised():
    assert schema.validate(
        {"LLM_N_CTX": "1024", "CHAT_TEMPERATURE": 0.5, "NARRATE_TOOLS": "On", "DEFAULT_CITY": " Lagos "}
    ) == {"LLM_N_CTX": "1024", "CHAT_TEMPERATURE": "0.5", "NARRATE_TOOLS": "true", "DEFAULT_CITY": "Lagos"}


@pytest.mark.parametrize(
    "updates",
    [
        {"LLM_N_CTX": "9999999"},          # above the maximum
        {"LLM_N_CTX": "lots"},             # not a number
        {"LLM_N_THREADS": "0"},            # below the minimum
        {"NARRATE_TOOLS": "maybe"},        # not a boolean
        {"NOT_A_SETTING": "x"},            # unknown
        {"DEFAULT_CITY": "a" * 500},       # too long
        {"DEFAULT_CITY": "Lagos\nHOST=0.0.0.0"},  # line injection
        {"OPENWEATHER_API_KEY": "abc\rdef"},
        {"DEFAULT_CITY": "O'Brien"},       # quote
    ],
)
def test_bad_values_are_rejected(updates):
    with pytest.raises(schema.SettingsError):
        schema.validate(updates)


@pytest.mark.parametrize("key", ["SALLY_API_KEY", "HOST", "PORT", "MEMORY_DB_PATH"])
def test_access_control_settings_cannot_be_changed_from_the_web(key):
    with pytest.raises(schema.SettingsError) as error:
        schema.validate({key: "0.0.0.0"})

    assert "editing .env" in error.value.errors[key]


def test_model_must_be_a_gguf_in_the_models_folder(models):
    assert schema.validate({"LLM_MODEL_PATH": "models/qwen.gguf"}) == {
        "LLM_MODEL_PATH": "models/qwen.gguf"
    }

    for bad in ("models/notes.txt", "../../etc/passwd", "/etc/passwd", "models/missing.gguf"):
        with pytest.raises(schema.SettingsError):
            schema.validate({"LLM_MODEL_PATH": bad})


# ---------------- describing ----------------

def test_describe_never_includes_secret_values(env_file, models):
    data = schema.describe(settings)
    flat = [f for g in data["groups"] for f in g["fields"]]
    weather = next(f for f in flat if f["key"] == "OPENWEATHER_API_KEY")

    assert weather["configured"] is True and "value" not in weather
    assert "old-secret" not in str(data) and "locked-secret" not in str(data)


def test_describe_reports_pending_changes_and_models(env_file, models):
    env_file.write_text(env_file.read_text() + "LLM_N_CTX=1024\n")

    data = schema.describe(settings, schema.read_env())
    ctx = next(f for g in data["groups"] for f in g["fields"] if f["key"] == "LLM_N_CTX")

    assert ctx["value"] == "1024" and ctx["pending"] is (settings.llm.n_ctx != 1024)
    assert data["models"] == [{"path": "models/qwen.gguf", "name": "qwen.gguf", "size_gb": 0.0}]
    assert "SALLY_API_KEY" in data["locked"]


# ---------------- the API ----------------

def test_api_saves_valid_settings_and_reports_restart(env_file, models, client):
    response = client.put(
        "/settings",
        json={"values": {"LLM_N_CTX": 1024, "LLM_MODEL_PATH": "models/qwen.gguf"}},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["saved"] == ["LLM_MODEL_PATH", "LLM_N_CTX"]
    assert "LLM_MODEL_PATH=models/qwen.gguf" in env_file.read_text()
    assert body["restart_required"] is True


def test_api_rejects_everything_if_any_value_is_bad(env_file, models, client):
    before = env_file.read_text()

    response = client.put(
        "/settings", json={"values": {"LLM_N_CTX": 1024, "HOST": "0.0.0.0"}}
    )

    assert response.status_code == 422
    assert "HOST" in response.json()["detail"]["errors"]
    assert env_file.read_text() == before  # nothing half-saved


def test_api_get_and_restart(env_file, models, client, monkeypatch):
    assert client.get("/settings").json()["groups"]

    called = []
    monkeypatch.setattr(schema, "restart_process", lambda: called.append(True))

    assert client.post("/settings/restart").json() == {"status": "restarting"}
    assert called == [True]


def test_settings_are_not_reachable_from_other_machines(env_file, models):
    remote = TestClient(web.app, client=("203.0.113.9", 5000))

    assert remote.get("/settings").status_code == 403
    assert remote.put("/settings", json={"values": {}}).status_code == 403
    assert remote.post("/settings/restart").status_code == 403
