#!/usr/bin/env python3
"""Stamp the version from pyproject.toml into the docs that must spell it out.

    python scripts/sync_version.py          # rewrite the files
    python scripts/sync_version.py --check  # exit 1 if anything is stale

pyproject.toml is the single source of truth; nothing else is edited by hand.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.version import get_version  # noqa: E402

VERSION = get_version()

# file -> [(pattern, replacement)]
STAMPS: dict[str, list[tuple[str, str]]] = {
    "README.md": [
        (r"Version-v[\w.+-]+-blue", f"Version-v{VERSION}-blue"),
        (r"\*\*v[\w.+-]+ Current Runtime:\*\*", f"**v{VERSION} Current Runtime:**"),
        (
            r"The active package version is `[^`]*`",
            f"The active package version is `{VERSION}`",
        ),
    ],
    "SALLY_HERMES_UPGRADE_PLAN.md": [
        (r"\*\*Current Version:\*\* `[^`]*`", f"**Current Version:** `{VERSION}`"),
    ],
}

# Lines that must not exist: the version is no longer an environment setting.
REMOVE_LINES = {
    "README.md": r"^SALLY_VERSION=.*\n",
    ".env.example": r"^SALLY_VERSION=.*\n",
}


def process(write: bool) -> list[str]:
    stale: list[str] = []

    for name in sorted(set(STAMPS) | set(REMOVE_LINES)):
        path = ROOT / name

        if not path.exists():
            continue

        original = path.read_text(encoding="utf-8")
        text = original

        for pattern, replacement in STAMPS.get(name, []):
            text = re.sub(pattern, replacement, text)

        if name in REMOVE_LINES:
            text = re.sub(REMOVE_LINES[name], "", text, flags=re.MULTILINE)

        if text != original:
            stale.append(name)

            if write:
                path.write_text(text, encoding="utf-8")

    return stale


def main() -> int:
    check = "--check" in sys.argv[1:]
    stale = process(write=not check)

    if check:
        if stale:
            print(f"Out of date with pyproject.toml ({VERSION}): {', '.join(stale)}")
            print("Run: python scripts/sync_version.py")
            return 1

        print(f"All versions match pyproject.toml ({VERSION}).")
        return 0

    print(f"Version {VERSION}: " + (f"updated {', '.join(stale)}" if stale else "nothing to change"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
