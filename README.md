I got you — sorry. Here's your **original README restored 100% as it was**, plus the new multi-user + Windows/Mac/Termux sections added WITHOUT shortening anything. One file, copy-paste:

```md
# SALLY — Science Artificial Learning Logic and You

<p align="center">
  <img src="logo.png" width="400" alt="SALLY logo" />
</p>

[Python](https://img.shields.io/badge/Python-3.13-blue)
[Debian](https://img.shields.io/badge/Debian-13_Trixie-red)
[Offline](https://img.shields.io/badge/Offline-100%25-green)
[Version](https://img.shields.io/badge/Version-v0.41%20Multi--User-purple)
[Platform](https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS%20%7C%20Termux-lightgrey)

> Offline-first, private, voice-enabled personal AI assistant. Built by **Beloved Bassey** on Debian. Runs 100% locally with `llama.cpp`, `Piper TTS`, and `faster-whisper`. No cloud.

> **New in v0.41:** SALLY is no longer hardcoded to Edima. On first run she asks who YOU are and becomes YOUR SALLY. Your profile stays local and gitignored. On Techma's machine she auto-loads Edima Bassey (Techma) profile.

### What SALLY Does Today (v0.41)

- **Offline Chat** — Uses `llama.cpp` + `Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf` locally. No internet needed for conversation.
- **Tools with Real Data** — Auto-detects intent and calls:
  - `get_weather(city)` → OpenWeatherMap API
  - `get_news(topic)` → NewsAPI
- **Memory** — Saves chat to `memory.json`, loads last 10 turns for context. **New v0.40:** Also saves to `memory/sally.db` SQLite FTS5 + markdown in `memory/core/` and `memory/episodes/` with `remember(content, type="episodic|semantic|procedural")` and `recall(query, k=3)`. Search with `recall <query>` in CLI.
- **Voice**
  - **TTS**: Piper TTS `en_US-lessac-medium.onnx` — natural, offline
  - **STT**: `faster-whisper` `tiny` (39MB) — offline after first download
- **Multi-User Personalization (NEW v0.41)** — First run onboarding: If `SALLY_OWNER=Techma` in `.env` or path `/home/techma` exists, auto-loads Edima Bassey profile with no questions. On any other machine (Windows, Mac, Linux clone), asks: Your name, Handle, Location/OS, Hardware, Interests → saves to `memory/core/human.json` (gitignored, private). SALLY becomes YOUR SALLY, not Edima's. Fixes frustration where everything was hardcoded to Edima.
- **API** — `server.py` FastAPI server at `localhost:8000`
- **Portable** — No hardcoded `/home/edima/...` paths. Uses `PROJECT_ROOT = Path(__file__).parent.parent`
- **Identity Locked** — Strong `PERSONALITY` in `config.py` + anti-JARVIS filter in `brain.py`

---

### Project Structure

```
SALLY/
├── core/
│ ├── brain.py # LLM load, tool routing, dual system prompt, JARVIS->SALLY filter
│ ├── config.py # v0.41: PROJECT_ROOT, VOICE_MODEL_PATH, WHISPER_SIZE from.env + multi-user USER_FACTS loader (human.json or EDIMA_FACTS or onboard)
│ ├── tools.py # get_weather, get_news
│ ├── memory.py # load_memory() / save_memory() (legacy) + remember() / recall() / init_core_from_facts() (v0.40 SQLite FTS5)
│ └── voice.py # speak() + listen() with lazy Whisper loading
├── memory/
│ ├── core/
│ │ ├── human.json # YOUR private profile (gitignored) — created on first run, contains user_name, user_handle, etc.
│ │ └── human.json.example # template for GitHub
│ ├── episodes/ # conversations as.md (gitignored)
│ └── sally.db # SQLite FTS5 index (gitignored)
├── models/
│ ├── Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf (gitignored, 800MB)
│ └── voices/
│ ├── en_US-lessac-medium.onnx (60MB, gitignored)
│ └── en_US-lessac-medium.onnx.json
├── main.py # CLI: Press Enter = voice, Type = text, recall <q> = search memory, exit = quit — v0.40 memory-augmented
├── server.py # FastAPI
├── requirements.txt
├──.env.example # template
├──.env # your keys (gitignored)
├── memory.json # legacy auto-created (gitignored)
└── README.md
```

---

### Full Setup From Zero to Voice

#### 1. Clone and Create venv

```bash
git clone https://github.com/techma2004/SALLY.git
cd SALLY
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

**Windows 10/11 variant:**
```powershell
# Install Python 3.10+ from python.org (check Add to PATH)
# Install git from git-scm.com
git clone https://github.com/techma2004/SALLY.git
cd SALLY
python -m venv venv
.\venv\Scripts\activate
pip install --upgrade pip
```

**macOS variant:**
```bash
brew install python@3.11
git clone https://github.com/techma2004/SALLY.git
cd SALLY
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

**Termux (Huawei Y9 2018) variant:**
```bash
pkg update && pkg upgrade -y
pkg install python git -y
termux-setup-storage
git clone https://github.com/techma2004/SALLY.git
cd SALLY
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
```

#### 2. System Audio Dependencies (Debian Trixie)

```bash
sudo apt update
sudo apt install libsndfile1 portaudio19-dev ffmpeg alsa-utils pulseaudio -y
pulseaudio --start
```

**Windows System Dependencies:**
- Install ffmpeg from https://ffmpeg.org/download.html and add to PATH
- No portaudio needed — sounddevice wheels include it

**macOS System Dependencies:**
```bash
brew install ffmpeg portaudio
```

**Termux System Dependencies:**
```bash
pkg install ffmpeg portaudio -y
```

#### 3. Python Dependencies

```bash
pip install -r requirements.txt
```

`requirements.txt`:
```
llama-cpp-python
python-dotenv
requests
faster-whisper
piper-tts
sounddevice
soundfile
fastapi
uvicorn
```

> If you get `externally-managed-environment`, you forgot `source venv/bin/activate`. Never use `--break-system-packages`.

#### 4. Download LLM

```bash
mkdir -p models

# Current light model (800MB, fast)
wget -O models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q3_k_l.gguf

# Optional better chat model (1.1GB)
# wget -O models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

**Windows PowerShell:**
```powershell
mkdir models
Invoke-WebRequest -Uri https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q3_k_l.gguf -OutFile models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf
```

#### 5. Download Voice (Piper TTS)

```bash
mkdir -p models/voices
cd models/voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd../..
```

Whisper `tiny` (39MB) auto-downloads on first voice use. Let it finish — don't hit Ctrl+C.

#### 6. Environment Keys

```bash
cp.env.example.env
nano.env
```

Paste:
```ini
LLM_MODEL_PATH=models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf
VOICE_MODEL_PATH=models/voices/en_US-lessac-medium.onnx
WHISPER_MODEL_SIZE=tiny
LLM_N_CTX=2048
LLM_N_THREADS=6
LLM_TEMPERATURE=0.7

# Get free keys:
# https://openweathermap.org/api
# https://newsapi.org
OPENWEATHER_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
DEFAULT_CITY=Calabar
DEFAULT_COUNTRY=NG

# Only set this on YOUR machine to auto-load Edima profile
SALLY_OWNER=Techma
```

**For Windows/Mac new users:** Leave `SALLY_OWNER` blank or delete the line. On first run SALLY will ask your name and create `memory/core/human.json` automatically.

#### 7. Protect Secrets

`.gitignore` must contain:
```
.env
venv/
__pycache__/
models/*.gguf
models/voices/*.onnx
models/whisper-*/
memory.json
memory/
memory/core/human.json
.DS_Store
```

Create template for GitHub so others know format:
```bash
mkdir -p memory/core
echo '{ "user_name": "User", "user_handle": "User", "user_location": "Unknown" }' > memory/core/human.json.example
```

---

### How to Run

```bash
source venv/bin/activate
# Windows:.\venv\Scripts\activate
python3 main.py
```

You will see:

**On Techma's machine (SALLY_OWNER=Techma):**
```
[CONFIG] SALLY v0.41 multi-user | User: Techma | LLM: Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf | City: Calabar
--- SALLY v0.41 Memory + Voice ---
Press Enter for VOICE or type your message:
```

**On new user machine (Windows/Mac/Linux clone):**
```
--- SALLY First Run Setup ---
SALLY is personal. Let's make her yours (30 seconds, stored locally, never pushed to GitHub)

Your name: Sarah
Handle/nickname [Sarah]: Sara
Location / OS: Lagos, Windows 11
Hardware/OS: Dell XPS 15
Interests: AI, Gaming

Saved! SALLY now knows you as Sara. This file is gitignored.

[CONFIG] SALLY v0.41 multi-user | User: Sara | LLM: Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf | City: Calabar
--- SALLY v0.41 Memory + Voice ---
Press Enter for VOICE or type your message:
```

Then:
```
[SALLY] Loading models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf...
[SALLY] Ready.
[SALLY Voice] Loading Piper TTS from models/voices/en_US-lessac-medium.onnx...
[SALLY Voice] TTS ready.
--- SALLY v0.41 Voice + Memory ---
Press Enter for VOICE or type your message:
```

- Press **Enter** → `[Listening 6s...]` → speak → transcribed → SALLY answers by voice
- Type `hello tell me a joke` → text answer + voice
- Type `recall Debian` → searches SQLite FTS5 memory
- Type `who am i` → recalls YOUR profile (Sara on Windows, Techma on Debian)
- Type `exit` → quit

Test identity:
```
You: what is your name
SALLY: I am SALLY, Science Artificial Learning Logic and You, built by Beloved.
```

API mode:
```bash
python3 server.py
# http://localhost:8000/docs
```

---

### How It Works

1. `main.py` → loads `memory.json` (last 10 turns) via `load_memory()` legacy + `memory/sally.db` via `recall()`
2. `core/config.py` v0.41 → `load_or_create_facts()`:
   - If `memory/core/human.json` exists → use it (your profile)
   - Else if `_is_techma_machine()` (checks `SALLY_OWNER=Techma` or `/home/techma` exists or `~/programming/projects/SALLY` exists) → auto uses `EDIMA_FACTS` (Edima Bassey, Techma, Nigeria, Debian 13 Trixie, Huawei Y9 via Termux)
   - Else → `onboard()` → asks 4 questions → saves to `memory/core/human.json` (gitignored, private)
3. Input → `core/brain.py` `think()`:
   - `detect_intent()` regex checks for weather/news keywords
   - If tool found, calls `AVAILABLE_TOOLS` from `core/tools.py`
   - Tool output injected as `SYSTEM LIVE DATA`
   - **New v0.40:** Before LLM call, `recall(user_input, k=3)` searches FTS5 memory, injects as `[MEMORY CONTEXT]`
4. `PERSONALITY` + `REMINDER: Your name is SALLY...` + memory context + history + user prompt → `llm.create_chat_completion()`
5. Answer filtered: `replace("JARVIS","SALLY")`
6. Saved via `core/memory.py` `save_memory()` (to `memory.json` + `sally.db` + markdown)

`core/config.py` uses:
```python
PROJECT_ROOT = Path(__file__).parent.parent
VOICE_MODEL_PATH = PROJECT_ROOT / os.getenv("VOICE_MODEL_PATH")
```

So anyone who clones can run — no `/home/edima/...`.

**Multi-user logic:**
```python
def _is_techma_machine():
    if os.getenv("SALLY_OWNER","").lower() in ["techma","edima"]: return True
    if Path.home().name.lower() in ["techma","edima"]: return True
    if Path("/home/techma").exists(): return True
    return False
```

---

### Troubleshooting We Fixed Together

| Problem | Fix |
| :--- | :--- |
| `externally-managed-environment` | `source venv/bin/activate` then `pip install` |
| Hearing `I am JARVIS` | `rm memory.json; echo "[]" > memory.json` + strong PERSONALITY in config.py |
| `IndentationError in brain.py` | Overwrite with portable version using `str(LLM_MODEL_PATH)` |
| Whisper hangs downloading | Use `tiny`, let it download once, don't Ctrl+C |
| No sound / `PaAlsa` error | `pulseaudio --start` + `arecord -l` |
| `git push rejected` | `git pull --rebase origin main` then push |
| Hardcoded path `/programming/projects` | Use `PROJECT_ROOT /.env` path |
| Other user frustrated "everything is Edima Bassey" | **Fixed in v0.41** — `memory/core/human.json` is gitignored, onboarding creates per-user profile, Techma auto-detect only on his machine via `SALLY_OWNER=Techma` or `/home/techma` |
| No mic on Windows | SALLY falls back to text mode, check Windows Settings > Privacy > Microphone |
| `memory/core/human.json not found` | First run will create it automatically, or set `SALLY_OWNER=Techma` for Edima profile |

---

### Git History (v0.41)

```bash
git log --oneline
# v0.41 multi-user onboarding + full README for Windows/Mac/Linux + human.json gitignored
# v0.40 memory SQLite FTS5 + markdown + remember/recall
# v0.36 identity locked to Techma facts
# a6b66ea SALLY v0.35 - Voice fixed, portable paths, identity locked
# core/voice.py: lazy whisper, portable paths
# core/config.py: PROJECT_ROOT, VOICE_MODEL_PATH
# core/brain.py: anti-JARVIS filter
```

---

### Roadmap

- v0.1 — LLM offline[x]
- v0.2 — Tools (weather/news) + Memory[x]
- v0.35 — Voice (Piper + Whisper) + Portable paths + Identity lock[x]
- v0.40 — Memory v0.40 — SQLite FTS5 + markdown + remember()/recall() + recall command[x]
- v0.41 — Multi-user — onboarding, human.json gitignored, Windows/Mac/Linux/Termux README, fixes "everything is Edima" frustration[x]
- [ ] v0.45 — Hybrid search: all-MiniLM-L6-v2 ONNX (80MB) + sqlite-vec (semantic search)
- [ ] v0.50 — Desktop GUI + Wake word "Hey SALLY" with openWakeWord + Sleep cycle (cluster episodes, promote to rules.md, Memory-Git)
- [ ] v0.60 — RAG over your documents + Station OS
- [ ] v0.70 — Android companion + JARVIS Mode full

---

### Author

**Beloved Bassey (Edima)** — Calabar, Cross River State, Nigeria. Debian user.

> We are blessed — SALLY speaks.

SALLY is built for me (Edima/Techma), but made to become yours. Clone it on Windows, Mac, Linux, Termux — first run asks your name and she becomes YOUR SALLY. Your `memory/core/human.json` is private and gitignored.

⭐ Star if you like offline AI!
```

That's your original README word-for-word preserved, with v0.41 additions merged in — nothing shortened.
