from __future__ import annotations

from datetime import datetime


def current_datetime() -> str:
    now = datetime.now().astimezone()

    return now.strftime(
        "%Y-%m-%d %H:%M:%S %Z (UTC%z)"
    )
