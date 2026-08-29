# SALLY Hermes Upgrade Tracker v0.45

Phase 0 Fixes DONE — mmproj crash fix,.env split
Phase 1 Memory DONE — SQLite FTS5 44KB 10ms search
Phase 2 Tools DONE — 7 tools, forced intent for 1.5B model
Phase 3 TUI DONE — prompt_toolkit + rich, /new /skills /memory /model /usage /help
Phase 4 Learning Loop DONE — tool_usage table, auto skill after 5 uses, /learn
Phase 5 Gateway DONE — Telegram @Sally_12345_bot working
    - core/gateway/gateway.py shared router (TUI/Telegram/Discord)
    - core/gateway/telegram.py v21.6 polling, Python 3.13 compatible
    - Shares memory.db + history_telegram.json + learner
    - Commands: /start /skills /memory /learn /new
    - Tested: same Edima memory in Telegram as TUI

Current:
- Model: Qwen2.5-Coder-1.5B Q3_K_L 1.3GB
- DB: memory.db 44KB + tool_usage + auto skills
- Frontends: TUI (main.py) + Telegram (@Sally_12345_bot)
- Gateway token loader handles PUT_YOUR_TOKEN malformed case

Next (tomorrow):
Phase 6 Cron — APScheduler daily brief 7am weather + memory + news
Phase 5b Optional — WhatsApp Cloud API gateway (Flask + ngrok)
