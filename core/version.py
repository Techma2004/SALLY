"""The one place SALLY's version is read from: ``pyproject.toml`` in the root.

Everything that shows a version (the /health endpoint, the API docs, the web
UI, ``python -m core.version``) goes through ``get_version()``. Documentation
that has to spell the number out (README badge, upgrade plan) is stamped from
here by ``scripts/sync_version.py`` and checked by ``tests/test_version.py``.

The native crate (``native/``) is a separately versioned component and keeps
its own number.
"""

from __future__ import annotations

import tomllib
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@lru_cache(maxsize=1)
def get_version() -> str:
    pyproject = ROOT / "pyproject.toml"

    try:
        with pyproject.open("rb") as handle:
            return str(tomllib.load(handle)["project"]["version"])
    except (OSError, KeyError, tomllib.TOMLDecodeError):
        pass

    # Installed without the source tree (e.g. a built wheel).
    try:
        from importlib.metadata import version

        return version("sally")
    except Exception:
        return "0.0.0+unknown"


__version__ = get_version()


if __name__ == "__main__":
    print(get_version())
