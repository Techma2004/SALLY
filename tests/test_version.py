import re
import subprocess
import sys
import tomllib
from pathlib import Path

from core.config import settings
from core.version import ROOT, get_version

PYPROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["version"]


def test_version_comes_from_pyproject():
    assert get_version() == PYPROJECT
    assert settings.sally.version == PYPROJECT


def test_health_and_api_docs_report_the_same_version():
    from fastapi.testclient import TestClient

    from core.gateway import web

    client = TestClient(web.app, client=("127.0.0.1", 5000))

    assert client.get("/health").json()["version"] == PYPROJECT
    assert web.app.version == PYPROJECT


def test_no_version_environment_override_remains():
    for name in (".env.example", "README.md"):
        text = (ROOT / name).read_text()
        assert not re.search(r"^SALLY_VERSION=", text, re.MULTILINE), name


def test_documents_that_spell_the_version_are_in_sync():
    # Fails with instructions when pyproject.toml changed but the docs did not.
    result = subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "sync_version.py"), "--check"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stdout


def test_module_prints_the_version():
    result = subprocess.run(
        [sys.executable, "-m", "core.version"],
        capture_output=True, text=True, cwd=ROOT,
    )

    assert result.stdout.strip() == PYPROJECT
