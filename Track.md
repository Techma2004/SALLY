# SALLY Hermes Upgrade Tracker v0.44

Phase 0 Fixes DONE — mmproj crash, .env
Phase 1 Memory DONE — SQLite FTS5 44KB 10ms
Phase 2 Tools DONE — 7 tools OpenAI format, forced intent for 1.5B
Phase 3 TUI DONE — prompt_toolkit + rich, /new /skills /memory /model /usage /help
Phase 4 Learning Loop DONE — tool_usage table, auto skill after 5 uses, /learn

Current:
- Model Qwen2.5-Coder-1.5B Q3_K_L
- DB 44KB, get_time x1, calc x2
- TUI v0.44 knows Edima
- Auto skills: calc_auto.py after 5x

Next:
Phase 5 Gateway — Telegram bot + Discord
Phase 6 Cron — APScheduler daily brief
