# SALLY Progress Tracker — Hermes Upgrade

## Phase 0 — Fixes (DONE)
- [x] mmproj crash fix,.env split, tool-forced brain

## Phase 1 — Hermes 3-Layer Memory (DONE) v0.42
- [x] SQLite + FTS5 ~10ms, memory.db 44KB, migration USER.md/MEMORY.md

## Phase 2 — Real Function Calling (DONE) v0.43
- [x] tools.py 7 tools OpenAI format
- [x] brain.py v4 forced intent detection for 1.5B model
- [x] chat() loop: LLM -> tool -> LLM
- Tested: time, weather, calc working with Qwen2.5-Coder-1.5B

## Phase 3 — Real TUI (DONE) v0.43
- [x] core/tui.py prompt_toolkit + rich
- [x] Commands: /new /skills /memory /model /usage /help /exit
- [x] FileHistory, autocomplete, AutoSuggest
- [x] main.py launches run_tui()
- Tested: /usage DB 44KB, /skills lists 7, weather 23.5C 96% rain, time, calc

## Phase 4 — Learning Loop (NEXT)
- [ ] core/learner.py auto-create skill after 5+ tool calls
- [ ] nudges every 15 turns

## Phase 5 — Gateway
- [ ] Telegram/Discord

## Phase 6 — Cron
- [ ] APScheduler

## Current:
- Model: Qwen2.5-Coder-1.5B Q3_K_L 1.3GB
- DB: 44KB, <1GB/year target
- Tools forced working
