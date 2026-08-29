# SALLY -> Hermes Agent Upgrade Plan
### Original Python Path, All Hermes Features

**Goal:** Keep `core/` + `memory/skills/` but get every Hermes feature: TUI, FTS5 memory, learning loop, gateway, cron, subagents.

**Current SALLY:** ~250MB + 2GB model, 2 skills, simple REPL, append-only memory
**Target SALLY v2:** ~300MB + <1GB memory DB + 50MB skills, 118 skills capability

---

### Phase 0 — DONE
- [x] Fixed mmproj crash, config split, brain tool-forced, memory junk cleaned
- [x] track.md created
- Structure:
```
SALLY/
├── main.py
├── core/
│   ├── config.py
│   ├── brain.py
│   ├── memory.py
│   └── skills.py
├── memory/
│   ├── skills/
│   ├── USER.md
│   ├── MEMORY.md
│   └── daily/
```

---

### Phase 1 — Hermes 3-Layer Memory (START HERE)
**Why first:** Everything else needs it. Hermes core is FTS5 search.

**Files:**
- `core/memory.py` -> upgrade to sqlite
- `memory/memory.db` (new, FTS5)

**Schema:**
```sql
CREATE TABLE sessions (id, started_at, summary)
CREATE TABLE memories (id, type, content, importance, created_at)
CREATE VIRTUAL TABLE memories_fts USING fts5(content, content='memories', content_rowid='id')
CREATE TABLE user_model (key, value, confidence) -- Honcho-style
```

**Features:**
1. Short-term: current context (already in LLM ctx)
2. Long-term: FTS5 search `search_memory(query)` ~10ms over 10K docs
3. User model: auto-builds from USER.md — coding style, timezone, prefs

**API:**
```python
search_memory("weather calabar") -> top 5 relevant memories
save_memory("user loves fruits", importance=0.8)
get_user_model() -> {"city": "Calabar", "loves": ["fruits"], "stack": ["python"]}
compact_daily() -> summarize daily/*.md into MEMORY.md when >1000 lines
```

**Storage:** <1GB for year (Hermes benchmark)

**Tasks:**
- [ ] Create memory.db
- [ ] Migrate USER.md + MEMORY.md into db
- [ ] Add FTS5 search
- [ ] Add periodic nudge: every 10 turns ask "save this?"

---

### Phase 2 — Real Hermes Function Calling
**Why:** Replace TRIGGERS hack. This stops hallucination.

**Current:**
```python
if "weather" in text: run weather
```

**Hermes:**
```python
# brain outputs:
{"tool": "weather", "args": {"city": "calabar"}}
```

**Files:**
- `core/brain.py` -> rewrite to ChatML + tool parser
- `core/tools.py` (new) — tool definitions in OpenAI format

**Prompt format:**
```
<|im_start|>system
You are SALLY. Tools: weather(city), news(topic), save_memory(fact), time()
<|im_end|>
<|im_start|>user
what is weather in calabar
<|im_end|>
<|im_start|>assistant to=self -> need to parse
```

**Parser:**
- If LLM outputs JSON with "tool", execute
- Else return text answer

**Model note:** Llama-3.2-1B is weak at JSON. Recommend upgrade to:
- `Hermes-3-Llama-3.1-8B` (8GB) — best for SALLY offline
- Or keep 1B but add fallback regex parser

**Tasks:**
- [ ] Define tools in `core/tools.py` (40+ eventually)
- [ ] Rewrite brain.py with ChatML + streaming tool output
- [ ] Add tool approval for shell tools

---

### Phase 3 — Real TUI (Hermes Terminal)
**Why:** Hermes TUI is 50% of UX.

**File:** `core/tui.py` (new)

**Library:** `prompt_toolkit` + `rich`

**Features:**
- Multiline editing (Shift+Enter)
- Slash commands: `/new`, `/model`, `/skills`, `/memory`, `/compress`, `/usage`
- Autocomplete for slash + skills
- Conversation history (up/down)
- Streaming tool output (like Hermes)
- Interrupt: Ctrl+C redirects

**Commands:**
```
/new -> new session
/skills -> list memory/skills/
/memory -> search_memory UI
/model -> switch models
/time -> call time skill
```

**Tasks:**
- [ ] Replace main.py input() with prompt_toolkit
- [ ] Add slash-command handler
- [ ] Add rich streaming

---

### Phase 4 — Closed Learning Loop (The Magic)
**Why:** This is why Hermes beats OpenClaw.

**Files:**
- `core/learner.py` (new)
- `core/skill_creator.py` (new)

**Loop:**
1. Task finishes with 5+ tool calls -> learner triggers
2. LLM summarizes task -> creates SKILL.md
```
# weekly-report-from-git
## When to use: user asks weekly report
## Steps: git log -> classify -> report
## Code: ...
```
3. Save to `memory/skills/auto/weekly-report.md`
4. Index into FTS5
5. Next time similar task -> auto-call skill (40% faster per Hermes benchmark)

**Skill self-improve:**
- Each skill tracks usage count + success
- After 3 uses, LLM rewrites skill to be better

**Nudges:**
- Every 20 turns: "You learned 3 facts, save to MEMORY.md? [y/n]"

**Tasks:**
- [ ] Add skill creator trigger
- [ ] Add skill improvement loop
- [ ] Implement nudge system

---

### Phase 5 — Gateway (Lives Where You Do)
**Why:** Talk to SALLY from Telegram while it runs on VPS.

**File:** `core/gateway.py`

**Single process handles:**
- CLI (TUI)
- Telegram
- Discord
- WhatsApp (via whatsapp-web)
- Voice memo transcription (whisper)

**Flow:**
```
Telegram -> gateway.py -> brain.py -> memory.db -> reply to Telegram
CLI -> same brain.py -> same memory.db -> reply to CLI
= cross-platform continuity
```

**Library:** `python-telegram-bot`, `discord.py`

**Tasks:**
- [ ] Create gateway.py with platform adapters
- [ ] Add voice transcription
- [ ] Config: `settings.json` -> `gateway.telegram_token`

---

### Phase 6 — Cron Scheduler
**Why:** Daily reports, nightly backups.

**File:** `core/cron.py`

**Library:** `APScheduler`

**Natural language:**
```
You: remind me daily 8am weather in Calabar on telegram
SALLY: creates cron job
```

**Storage:** `cron/output/` — monitor disk (Hermes had bug where this filled disk)

**Tasks:**
- [ ] Add cron with natural language parser
- [ ] Add delivery to any platform
- [ ] Add disk cleanup for cron/output

---

### Phase 7 — Subagents / Parallel
**Why:** Complex tasks.

**File:** `core/subagents.py`

**Feature:**
- Spawn isolated python scripts that call tools via RPC
- Zero-context-cost turns

**Example:**
```
You: research 3 competitors in parallel
SALLY: spawns 3 subagents -> each researches one -> aggregates
```

---

### Phase 8 — Backends / Runs Anywhere
**File:** `core/backends.py`

**Backends:**
- local (now)
- docker
- ssh
- modal / daytona (serverless, hibernates when idle)

**Config:** `settings.json` -> `backend: "local"`

---

### Implementation Order (recommended)

**Week 1:**
- Phase 1: FTS5 memory (2 days) — biggest win
- Phase 2: Real tool calling (2 days) — fixes hallucination
- Phase 3: TUI basics (1 day)

**Week 2:**
- Phase 4: Learning loop (3 days) — makes SALLY self-improving
- Phase 6: Cron (1 day)

**Week 3:**
- Phase 5: Telegram gateway (2 days)
- Phase 7: Subagents (2 days)

**Total storage target:** Core 200MB + DB <1GB + Skills <50MB = <1.3GB (API mode) or +8GB model = ~10GB offline

---

### Next Immediate File

`core/memory.py` v2 — Hermes 3-layer with FTS5.

Want me to write it now? It will migrate your existing USER.md/MEMORY.md into memory.db and give you `search_memory()` like Hermes.

