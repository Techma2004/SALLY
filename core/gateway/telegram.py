from __future__ import annotations

import asyncio
import os

from .gateway import Gateway


def _allowed_ids() -> set[int]:
    raw = os.getenv("TELEGRAM_ALLOWED_IDS", "")

    return {
        int(value.strip())
        for value in raw.split(",")
        if value.strip().isdigit()
    }


def _is_allowed(user_id: int) -> bool:
    allowed = _allowed_ids()

    return not allowed or user_id in allowed


def run_telegram(gateway: Gateway | None = None) -> None:
    try:
        from telegram import Update
        from telegram.ext import (
            Application,
            CommandHandler,
            ContextTypes,
            MessageHandler,
            filters,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Telegram support is optional. Install it with "
            "`uv add python-telegram-bot` before using the "
            "telegram interface."
        ) from exc

    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()

    if not token:
        raise RuntimeError(
            "TELEGRAM_BOT_TOKEN is not configured."
        )

    gateway = gateway or Gateway()

    async def start(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if update.effective_user is None or update.message is None:
            return

        if not _is_allowed(update.effective_user.id):
            return

        await update.message.reply_text(
            "Hello! I'm SALLY. Send me a message and I'll help."
        )

    async def handle_message(
        update: Update,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        if (
            update.effective_user is None
            or update.message is None
            or not update.message.text
        ):
            return

        user_id = update.effective_user.id

        if not _is_allowed(user_id):
            return

        await update.message.chat.send_action("typing")

        response = await asyncio.to_thread(
            gateway.handle,
            update.message.text,
            user_id=str(user_id),
            source="telegram",
        )

        answer = response.answer or response.error or (
            "SALLY could not complete that request."
        )

        for start_index in range(0, len(answer), 4000):
            await update.message.reply_text(
                answer[start_index:start_index + 4000]
            )

    app = Application.builder().token(token).build()

    app.add_handler(CommandHandler("start", start))
    app.add_handler(
        MessageHandler(
            filters.TEXT & ~filters.COMMAND,
            handle_message,
        )
    )

    print("SALLY Telegram gateway running.")
    app.run_polling()
