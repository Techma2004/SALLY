from __future__ import annotations

import argparse

from core.config import settings


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SALLY interface launcher"
    )
    parser.add_argument(
        "interface",
        nargs="?",
        choices=("tui", "web", "telegram"),
        default="tui",
        help="interface to launch",
    )

    args = parser.parse_args()

    if args.interface == "tui":
        from core.tui import run_tui

        run_tui()
        return

    if args.interface == "web":
        import uvicorn

        uvicorn.run(
            "core.gateway.web:app",
            host=settings.server.host,
            port=settings.server.port,
            reload=False,
        )
        return

    if args.interface == "telegram":
        from core.gateway.telegram import run_telegram

        run_telegram()


if __name__ == "__main__":
    main()
