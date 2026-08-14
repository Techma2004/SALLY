# SALLY — Science Artificial Learning Logic and You

<p align="center">
  <img src="logo.png" width="400" alt="SALLY logo" />
</p>

![Python](https://img.shields.io/badge/Python-3.13-blue)
![Debian](https://img.shields.io/badge/Debian-13_Trixie-red)
![Offline](https://img.shields.io/badge/Offline-100%25-green)
![Version](https://img.shields.io/badge/Version-v0.35%20Voice-purple)

> Offline-first, private, voice-enabled personal AI assistant. Built by **Beloved Bassey** on Debian. Runs 100% locally with `llama.cpp`, `Piper TTS`, and `faster-whisper`. No cloud.

### What SALLY Does Today (v0.35)

- **Offline Chat** — Uses `llama.cpp` + `Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf` locally. No internet needed for conversation.
- **Tools with Real Data** — Auto-detects intent and calls:
  - `get_weather(city)` → OpenWeatherMap API
  - `get_news(topic)` → NewsAPI
- **Memory** — Saves chat to `memory.json`, loads last 10 turns for context.
- **Voice**
  - **TTS**: Piper TTS `en_US-lessac-medium.onnx` — natural, offline
  - **STT**: `faster-whisper` `tiny` (39MB) — offline after first download
- **API** — `server.py` FastAPI server at `localhost:8000`
- **Portable** — No hardcoded `/home/edima/...` paths. Uses `PROJECT_ROOT = Path(__file__).parent.parent`
- **Identity Locked** — Strong `PERSONALITY` in `config.py` + anti-JARVIS filter in `brain.py`

---

### Project Structure

```
SALLY/
├── core/
│   ├── brain.py   # LLM load, tool routing, dual system prompt, JARVIS->SALLY filter
│   ├── config.py  # PROJECT_ROOT, VOICE_MODEL_PATH, WHISPER_SIZE from .env
│   ├── tools.py   # get_weather, get_news
│   ├── memory.py  # load_memory() / save_memory()
│   └── voice.py   # speak() + listen() with lazy Whisper loading
├── models/
│   ├── Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf
│   └── voices/
│       ├── en_US-lessac-medium.onnx (60MB)
│       └── en_US-lessac-medium.onnx.json
├── main.py        # CLI: Press Enter = voice, Type = text, exit = quit
├── server.py      # FastAPI
├── requirements.txt
├── .env.example   # template
├── .env           # your keys (gitignored)
├── memory.json    # auto-created (gitignored)
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

#### 2. System Audio Dependencies (Debian Trixie)

```bash
sudo apt update
sudo apt install libsndfile1 portaudio19-dev ffmpeg alsa-utils pulseaudio -y
pulseaudio --start
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

#### 5. Download Voice (Piper TTS)

```bash
mkdir -p models/voices
cd models/voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd ../..
```

Whisper `tiny` (39MB) auto-downloads on first voice use. Let it finish — don't hit Ctrl+C.

#### 6. Environment Keys

```bash
cp .env.example .env
nano .env
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
```

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
.DS_Store
```

---

### How to Run

```bash
source venv/bin/activate
python3 main.py
```

You will see:
```
[SALLY] Loading models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf...
[SALLY] Ready.
[SALLY Voice] Loading Piper TTS from models/voices/en_US-lessac-medium.onnx...
[SALLY Voice] TTS ready.
--- SALLY v0.3 Voice ---
Press Enter for VOICE or type your message:
```

- Press **Enter** → `[Listening 6s...]` → speak → transcribed → SALLY answers by voice
- Type `hello tell me a joke` → text answer + voice
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

1. `main.py` → loads `memory.json` (last 10 turns)
2. Input → `core/brain.py` `think()`:
   - `detect_intent()` regex checks for weather/news keywords
   - If tool found, calls `AVAILABLE_TOOLS` from `core/tools.py`
   - Tool output injected as `SYSTEM LIVE DATA`
3. `PERSONALITY` + `REMINDER: Your name is SALLY...` + history + user prompt → `llm.create_chat_completion()`
4. Answer filtered: `replace("JARVIS","SALLY")`
5. Saved via `core/memory.py`, spoken via `core/voice.py` `speak()`

`core/config.py` uses:
```python
PROJECT_ROOT = Path(__file__).parent.parent
VOICE_MODEL_PATH = PROJECT_ROOT / os.getenv("VOICE_MODEL_PATH")
```

So anyone who clones can run — no `/home/edima/...`.

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
| Hardcoded path `/programming/projects` | Use `PROJECT_ROOT / .env` path |

---

### Git History (v0.35)

```bash
git log --oneline
# a6b66ea SALLY v0.35 - Voice fixed, portable paths, identity locked
# core/voice.py: lazy whisper, portable paths
# core/config.py: PROJECT_ROOT, VOICE_MODEL_PATH
# core/brain.py: anti-JARVIS filter
```

---

### Roadmap

- [x] v0.1 — LLM offline
- [x] v0.2 — Tools (weather/news) + Memory
- [x] v0.35 — Voice (Piper + Whisper) + Portable paths + Identity lock
- [ ] v0.4 — Wake word "Hey SALLY" with openWakeWord
- [ ] v0.5 — Desktop GUI
- [ ] v0.6 — RAG over your documents

---

### Author

**Beloved Bassey (Edima)** — Calabar, Cross River State, Nigeria. Debian user.

> We are blessed — SALLY speaks.

⭐ Star if you like offline AI!
