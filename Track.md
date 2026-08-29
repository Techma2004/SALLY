# SALLY Hermes Upgrade Tracker v0.46 - COMPLETE

Phase 0 Fixes DONE
Phase 1 Memory DONE — SQLite FTS5 44KB 10ms
Phase 2 Tools DONE — 7 tools + forced intent for 1.5B
Phase 3 TUI DONE — prompt_toolkit + rich
Phase 4 Learning Loop DONE — tool_usage + auto skills + /learn
Phase 5 Gateway DONE — Telegram @Sally_12345_bot v21.6, Python 3.13 fix
Phase 6 Cron DONE — APScheduler daily 7am WAT brief
    - core/cron.py self-contained (no handle_tool_call dep)
    - Weather Calabar + News NG + Memory recap + Learner stats
    - Saves memory/daily/YYYY-MM-DD.md + saves to memory.db
    - Pushes to Telegram via telegram_chat_ids.json
    - Run: python3 -m core.cron --now (test) / python3 -m core.cron (daemon)

Hermes v0.46 COMPLETE:
- TUI: python3 main.py
- Telegram: python3 -m core.gateway.telegram
- Cron: python3 -m core.cron (separate terminal / tmux)

Stack: Qwen2.5-Coder-1.5B Q3_K_L, SQLite, PTK TUI, python-telegram-bot 21.6, APScheduler
