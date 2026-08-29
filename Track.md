# SALLY Project — track.md
### From beginning to end

**Project:** SALLY — offline, private, blank-slate AI assistant
**Model:** Llama-3.2-1B-Instruct (via llama_cpp)
**Goal:** Learn from user, not from internet. Memory = Obsidian + Hermes style.

---

### 1. Initial State (where we started)

```
SALLY/
├── main.py
├── core/
│   ├── config.py (had wired defaults)
│   ├── brain.py (hallucinated)
│   ├── memory.py (created junk)
│   └── skills.py
├── memory/
│   ├── skills/
│   │   ├── weather.py (OpenWeatherMap)
│   │   └── news.py (NewsAPI)
│   ├── events/ (junk — 1 file per message)
│   ├── facts/ (junk)
│   ├── USER.md
│   └── MEMORY.md
├── models/
│   ├── Llama-3.2-1B-Instruct...
│   └── mmproj-Ornith... (vision projector, NOT an LLM)
├── settings.json
└── .env
```

Personality: Blank slate. Should learn `USER.md` and `MEMORY.md` from conversation.

---

### 2. Bugs Found

**Bug 1: mmproj crash**
- `.env` had `LLM_MODEL_PATH=mmproj-Ornith-Llama...` 
- mmproj is eyes only, not a LLM. `llama_cpp` crashed trying to load it as LLM.
- Error: model load failed / no context.

**Bug 2: Config had wired defaults**
- `config.py` had hardcoded `DEFAULT_CITY = "Calabar"` etc inside code.
- Should be in `settings.json` / `.env` separation.
- `example.settings.json` existed but not used as fallback.

**Bug 3: Brain hallucinated instead of using tools**
- You asked `what is weather in calabar`
- Skill was loaded `[SKILL] weather loaded` but brain replied `I don't have real-time access...`
- Cause: `brain.py` called LLM before checking tools, LLM hallucinated.

**Bug 4: Memory junk files**
- Saying `i love eating fruits` created `memory/events/2026-05-13_i_love_eating_fruits.md`
- Every chat = new file. Folder exploded.
- Should be Hermes + Obsidian: one `MEMORY.md`, one `USER.md`, and `daily/YYYY-MM-DD.md`

**Bug 5: Using `cat` to edit files**
- Bad practice — overwrites, no validation.

---

### 3. Fixes Applied

**Fix 1: Config split — `core/config.py`**
- Now reads `settings.json` first, falls back to `example.settings.json`
- Secrets only from `.env` via `python-dotenv`
- Zero hardcoded defaults in code
- Validates model path exists and rejects mmproj:
```python
if "mmproj" in LLM_MODEL_PATH.name: raise ValueError(...)
```

**Fix 2: Brain — `core/brain.py`**
- Load skills first
- `run_skill_if_needed(text)` checks triggers BEFORE LLM call
- If skill matched, return tool output directly — no LLM hallucination
- Only if no skill matched, call LLM with memory context

**Fix 3: Memory — `core/memory.py` (Hermes + Obsidian)**
- Removed `memory/events/` and `memory/facts/`
- Now:
  - `memory/USER.md` = who user is
  - `memory/MEMORY.md` = facts loves/hates
  - `memory/daily/YYYY-MM-DD.md` = only real daily log
- Functions: `load_user()`, `load_memory()`, `save_user_fact()`, `save_memory_fact()`, `log_daily()`
- `i love fruits` now appends `- i love fruits` to `MEMORY.md`, not new file

**Fix 4: Skills loader — `core/skills.py`**
- Auto-discovers `memory/skills/*.py`
- Each skill defines `TRIGGERS = [...]`
- No need to edit core to add skill

**Fix 5: Clean junk**
```bash
rm -rf memory/events memory/facts
rm -rf memory/daily/*.md
```

**Fix 6: Weather / News skills**
- Added `TRIGGERS` to each
- `weather.py` uses `OPENWEATHER_API_KEY` from .env
- `news.py` uses `NEWS_API_KEY`
- Returns `[TOOL] No key...` if key missing instead of crashing

---

### 4. Weft Experiment (considered and reverted)

You asked: can we rewrite SALLY in weft?

We explored two meanings:
1. **Weft architecture** = warp (core) + weft (skills woven) — good, it's just plugin architecture
2. **Weft language** = custom DSL with `manifest.weft` — not good for now, adds parser maintenance, no ecosystem

Decision: Keep weft *idea* (modular folders) but stay in **original Python**. No custom language. Python = warp, Python skills = weft.

We cleaned experiment:
```bash
rm -rf warp weft weft_lang
```

---

### 5. Final Working Structure (original Python)

```
SALLY/
├── main.py
├── core/
│   ├── config.py # no wired defaults, settings.json + .env
│   ├── brain.py # tool-forced, no hallucination
│   ├── memory.py # USER.md + MEMORY.md + daily/
│   └── skills.py # auto-discovers memory/skills/*.py
├── memory/
│   ├── skills/
│   │   ├── weather.py # TRIGGERS = ["weather", "temperature"...]
│   │   └── news.py # TRIGGERS = ["news", "headline"...]
│   ├── USER.md
│   ├── MEMORY.md
│   └── daily/
│       └── 2026-05-13.md
├── models/
│   └── Llama-3.2-1B-Instruct... (NOT mmproj)
├── settings.json
├── example.settings.json
├── .env
└── .env.example
```

**settings.json** = threads, ctx, temp, city
**.env** = LLM_MODEL_PATH, API keys, voice model

---

### 6. How to Run

```bash
python3 main.py
You: my name is Edima
You: i love eating fruits
You: what is weather in calabar
You: exit

ls memory/
cat memory/USER.md
cat memory/MEMORY.md
cat memory/daily/2026-05-13.md
```

Expected: No new files in `memory/`, only updates to USER.md / MEMORY.md.

---

### 7. How to Add a New Skill

1. Create `memory/skills/time.py`:
```python
TRIGGERS = ["time", "what time", "clock"]
import datetime
def run(query: str, city=None):
    return f"Time now: {datetime.datetime.now().strftime('%H:%M:%S')}"
```
2. Restart `main.py` — auto-loaded as `[SKILL] loaded time`

---

### 8. Current Status

- [x] Model loads (Llama-3.2-1B, not mmproj)
- [x] Config separation done
- [x] Brain tool-forced fixed
- [x] Memory junk cleaned, Hermes+Obsidian working
- [x] Weather/news skills load but need real API keys in .env
- [ ] Voice in/out (whisper + TTS)
- [ ] More skills: time, calc, calendar
- [ ] Test real weather with key

---

### 9. Next Steps

1. Add real keys to `.env`: `OPENWEATHER_API_KEY=...`
2. Build `time` skill (2 min win)
3. Build `calc` skill
4. Then voice

Track last updated: 2026-05-13 — everything works, original Python path restored.
