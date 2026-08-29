import os, asyncio, json
from pathlib import Path
from dotenv import load_dotenv
PROJECT_ROOT = Path(__file__).parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

def get_token():
    # try env
    t = os.getenv("TELEGRAM_BOT_TOKEN", "")
    if t and "PUT_YOUR" not in t and len(t) > 20 and ":" in t:
        return t.strip()
    # try.env file directly
    envf = PROJECT_ROOT / ".env"
    if envf.exists():
        for line in envf.read_text().splitlines():
            if line.startswith("TELEGRAM_BOT_TOKEN="):
                v = line.split("=",1)[1].strip().strip('"').strip("'").strip()
                # handle malformed PUT_YOUR_TOKEN_HERE=XXXX
                if "=" in v and "PUT" in v:
                    v = v.split("=")[-1]
                if v and "PUT" not in v and ":" in v and len(v) > 20:
                    return v
    # try settings.json
    sf = PROJECT_ROOT / "settings.json"
    if sf.exists():
        try:
            d = json.loads(sf.read_text())
            tok = d.get("TELEGRAM_BOT_TOKEN","")
            if tok and ":" in tok:
                return tok
        except: pass
    return None

try:
    from telegram import Update
    from telegram.ext import Application, CommandHandler, MessageHandler, filters, ContextTypes
    HAS_TG = True
except ImportError as e:
    HAS_TG = False
    print(f"Missing dep: {e}")

from.gateway import gateway_chat
from pathlib import Path
CHAT_IDS_FILE = Path(__file__).parent.parent.parent / "memory" / "telegram_chat_ids.json"

ALLOWED_IDS = os.getenv("TELEGRAM_ALLOWED_IDS","")
ALLOWED_SET = set([int(x.strip()) for x in ALLOWED_IDS.split(",") if x.strip().isdigit()]) if ALLOWED_IDS else set()

def is_allowed(uid):
    return True if not ALLOWED_SET else uid in ALLOWED_SET

async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not is_allowed(update.effective_user.id):
        await update.message.reply_text("Private SALLY.")
        return
    await update.message.reply_text(f"Hello {update.effective_user.first_name}! I'm SALLY v0.45\n\n/skills\n/memory <q>\n/learn\n/new\n\nTry: what time is it")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    user_text = update.message.text or ""
    print(f"[TG] {update.effective_user.username} ({uid}): {user_text[:80]}")
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="typing")
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, lambda: gateway_chat(str(uid), user_text, "telegram"))
    await update.message.reply_text(answer[:4000])

async def handle_commands(update: Update, context: ContextTypes.DEFAULT_TYPE):
    uid = update.effective_user.id
    if not is_allowed(uid):
        return
    user_text = update.message.text or ""
    loop = asyncio.get_event_loop()
    answer = await loop.run_in_executor(None, lambda: gateway_chat(str(uid), user_text, "telegram"))
    await update.message.reply_text(answer[:4000])

def main():
    if not HAS_TG:
        print("pip install python-telegram-bot==21.6 python-dotenv")
        return
    token = get_token()
    if not token:
        print("No real TELEGRAM_BOT_TOKEN in.env!")
        print("Add: TELEGRAM_BOT_TOKEN=123456:AAH...")
        return
    print(f"[GATEWAY] Starting @Sally_12345_bot with token {token[:10]}...{token[-6:]}")
    app = Application.builder().token(token).build()
    app.add_handler(CommandHandler("start", start_command))
    app.add_handler(CommandHandler("help", start_command))
    app.add_handler(CommandHandler("skills", handle_commands))
    app.add_handler(CommandHandler("memory", handle_commands))
    app.add_handler(CommandHandler("learn", handle_commands))
    app.add_handler(CommandHandler("model", handle_commands))
    app.add_handler(CommandHandler("usage", handle_commands))
    app.add_handler(CommandHandler("new", handle_commands))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    print("[GATEWAY] Polling @Sally_12345_bot - message it now!")
    app.run_polling()

if __name__ == "__main__":
    main()
