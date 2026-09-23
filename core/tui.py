from __future__ import annotations

import os
from pathlib import Path

from core.config import PROJECT_ROOT, settings
from core.gateway import Gateway

try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.history import FileHistory

    HAS_PROMPT_TOOLKIT = True
except ImportError:
    HAS_PROMPT_TOOLKIT = False


HISTORY_FILE = PROJECT_ROOT / "memory" / ".prompt_history"

COMMANDS = {
    "/new": "clear the current prompt history",
    "/memory": "search SALLY's memory",
    "/skills": "list registered tools",
    "/model": "show the configured model",
    "/usage": "show local runtime information",
    "/help": "show commands",
    "/exit": "exit SALLY",
    "/quit": "exit SALLY",
}


def _command_help() -> str:
    return "\n".join(
        f"{name} - {description}"
        for name, description in COMMANDS.items()
    )


def _handle_command(
    command: str,
    gateway: Gateway,
) -> str | None:
    stripped = command.strip()
    lowered = stripped.lower()

    if lowered in {"/exit", "/quit"}:
        return "EXIT"

    if lowered in {"/help", "/h"}:
        return _command_help()

    if lowered == "/new":
        if HISTORY_FILE.exists():
            HISTORY_FILE.write_text("", encoding="utf-8")
        return "Prompt history cleared."

    if lowered.startswith("/memory") or lowered.startswith("recall "):
        query = (
            stripped.split(" ", 1)[1]
            if " " in stripped
            else ""
        ).strip()

        if not query:
            return "Usage: /memory <query>"

        memories = gateway.coordinator.memory.search(
            query,
            limit=5,
        )

        if not memories:
            return "No matching memories found."

        return "\n".join(
            f"- [{memory.memory_type.value}] {memory.content}"
            for memory in memories
        )

    if lowered == "/skills":
        tools = gateway.coordinator.runtime.tools.names()

        if not tools:
            return "No tools registered."

        return "Registered tools:\n" + "\n".join(
            f"- {name}"
            for name in tools
        )

    if lowered == "/model":
        model_path = Path(settings.llm.model_path)

        if not model_path.is_absolute():
            model_path = PROJECT_ROOT / model_path

        return (
            f"Model: {settings.llm.model_path}\n"
            f"Exists: {model_path.exists()}\n"
            f"Context: {settings.llm.n_ctx}\n"
            f"Threads: {settings.llm.n_threads}"
        )

    if lowered == "/usage":
        db_path = PROJECT_ROOT / "memory" / "memory.db"
        size_kb = (
            db_path.stat().st_size / 1024
            if db_path.exists()
            else 0
        )

        return (
            f"Memory DB: {size_kb:.1f} KB\n"
            f"Active tasks: "
            f"{len(gateway.coordinator.runtime.active_tasks)}"
        )

    return None


def run_tui() -> None:
    gateway = Gateway()

    if HAS_PROMPT_TOOLKIT:
        session = PromptSession(
            history=FileHistory(str(HISTORY_FILE)),
            completer=WordCompleter(
                list(COMMANDS),
                ignore_case=True,
            ),
            auto_suggest=AutoSuggestFromHistory(),
        )
    else:
        session = None

    print(
        f"=== {settings.sally.name} "
        f"v{settings.sally.version} ==="
    )
    print("Type /help for commands.")

    while True:
        try:
            message = (
                session.prompt("\nYou: ")
                if session is not None
                else input("\nYou: ")
            )
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        message = message.strip()

        if not message:
            continue

        if message.startswith("/") or message.lower().startswith("recall "):
            command_result = _handle_command(
                message,
                gateway,
            )

            if command_result == "EXIT":
                print("Goodbye.")
                break

            if command_result is not None:
                print(f"\n{command_result}")
                continue

        try:
            response = gateway.handle(
                message,
                user_id="local",
                source="tui",
            )

            if response.answer:
                print(f"\nSALLY: {response.answer}")
            elif response.error:
                print(f"\nSALLY: {response.error}")
            else:
                print("\nSALLY: No response.")

        except Exception as exc:
            print(f"\nSALLY error: {exc}")
