# SALLY Progress Tracker — Hermes Upgrade
### Keep track from beginning to end

## Phase 0 — Fixes (DONE)
- [x] Fixed mmproj crash — config.py rejects vision projection files
- [x] Split config: .env (API keys) vs settings.json (model params)
- [x] Removed wired defaults — no hardcoded model names
- [x] Brain tool-forced — no more hallucination, must use tools for weather/news
- [x] Cleaned memory junk — removed fake conversations from USER.md
- [x] Created track.md

## Phase 1 — Hermes 3-Layer Memory (DONE) v0.42
- [x] core/memory.py v2
  - SQLite + FTS5 (~10ms search over 10K docs) — Hermes benchmark
  - Tables: sessions, memories, memories_fts, user_model
  - Functions: init_db(), save_memory(), search_memory(), save_user_model(), get_user_model()
  - Migration: USER.md + MEMORY.md -> memory.db
  - Backward compat: load_user(), load_memory() still work
  - Nudge: should_nudge() every 15 turns
  - compact_daily()
- [x] Storage: Core 200MB + DB <1GB/year + Skills <50MB
- [x] Fixed .gitignore: ignore memory.db/daily/cron/output, track USER.md/MEMORY.md/skills
- [x] Tested: search_memory("fruits") works

## Phase 2 — Real Hermes Function Calling (IN PROGRESS) v0.43
- [x] core/tools.py v2
  - TOOLS list in OpenAI format (7 tools)
  - TOOL_MAP + execute_tool(name, args)
  - get_tool_descriptions_for_prompt() for ChatML
  - Tools: get_weather, get_news, get_time, calc, remember_fact, recall_memory, save_user_fact_tool
- [x] core/brain.py v2
  - Hermes ChatML: <|im_start|>system with PERSONALITY + [USER CONTEXT] + [MEMORY CONTEXT] + [TOOLS]
  - parse_tool_call() handles 3 formats:
    1. {"tool": "name", "args": {}}
    2. {"name": "name", "arguments": {}}
    3. <tool_call>{...}</tool_call>
  - chat() = tool loop: LLM -> tool -> LLM -> final answer
  - filter_identity(): JARVIS -> SALLY
  - get_llm() with mmproj safety
  - should_nudge() integration
  - Backward compat: respond() wrapper

## Phase 3 — Real TUI (NEXT) v0.43
- [ ] core/tui.py with prompt_toolkit + rich
- [ ] Slash commands: /new /skills /memory /model /compress /usage
- [ ] Autocomplete, history, streaming, Ctrl+C interrupt

## Phase 4 — Learning Loop (PLANNED) v0.44
- [ ] core/learner.py
- [ ] core/skill_creator.py — auto-create skill after 5+ tool calls
- [ ] Skills self-improve after 3 uses
- [ ] Nudges every 20 turns

## Phase 5 — Gateway (PLANNED) v0.44
- [ ] core/gateway.py — Telegram/Discord/Slack/WhatsApp
- [ ] Voice memo transcription
- [ ] Cross-platform continuity

## Phase 6 — Cron (PLANNED) v0.45
- [ ] core/cron.py — APScheduler + natural language
- [ ] Disk cleanup for cron/output/

## Phase 7 — Subagents (PLANNED) v0.46
- [ ] core/subagents.py — parallel delegates

## Phase 8 — Backends (PLANNED) v0.50
- [ ] core/backends.py — local/docker/ssh/modal

---

## Current File Tree
```
SALLY/
├── core/
│   ├── config.py (Phase 0)
│   ├── memory.py (Phase 1 DONE)
│   ├── tools.py (Phase 2 DONE)
│   ├── brain.py (Phase 2 DONE)
│   ├── skills.py (old, to be migrated to tools/)
│   └── voice.py
├── memory/
│   ├── USER.md (tracked)
│   ├── MEMORY.md (tracked)
│   ├── memory.db (gitignored, FTS5)
│   ├── daily/ (gitignored)
│   └── skills/
├── track.md
├── SALLY_HERMES_UPGRADE_PLAN.md
└── main.py
```

## How to test Phase 2
```bash
python3 -c "from core.brain import parse_tool_call; print(parse_tool_call('{\"tool\": \"get_weather\", \"args\": {\"city\": \"Calabar\"}}'))"
python3 -c "from core.tools import execute_tool; print(execute_tool('get_time', {}))"
python3 -c "from core.tools import execute_tool; print(execute_tool('calc', {'expression': '2+2*3'}))"
```

## Git log so far
```
Phase 1: Hermes 3-layer memory...
Phase 0: fix mmproj crash, tool-forced brain...
```

