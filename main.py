from __future__ import annotations

import argparse

from core.config import settings


WEB_PORT = 8080


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SALLY interface launcher"
    )
    parser.add_argument(
        "interface",
        nargs="?",
        choices=("web", "telegram"),
        default="web",
        help="interface to launch",
    )

    args = parser.parse_args()

    if args.interface == "web":
        import uvicorn

        uvicorn.run(
            "core.gateway.web:app",
            host=settings.server.host,
            port=WEB_PORT,
            reload=False,
        )
        return

    if args.interface == "telegram":
        from core.gateway.telegram import run_telegram

        run_telegram()


if __name__ == "__main__":
    main()
