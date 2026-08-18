# SALLY — Science Artificial Learning Logic and You

<p align="center">
  <img src="logo.png" width="400" alt="SALLY Logo" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS%20%7C%20Termux-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/Offline-100%25-2ECC71?style=flat" alt="Offline">
  <img src="https://img.shields.io/badge/Version-v0.41-blue?style=flat" alt="Version">
  <img src="https://img.shields.io/badge/License-MIT-green?style=flat" alt="License">
</p>

**SALLY is an offline-first, private, voice-enabled personal AI assistant.** It runs entirely locally using `llama.cpp`, `Piper TTS`, and `faster-whisper`. No cloud dependencies, no telemetry.

> Built on Debian 13 (Trixie) and designed to be fully portable across Linux, Windows, macOS, and Android via Termux.

## Table of Contents
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Architecture](#architecture)
- [Memory System](#memory-system)
- [Personalization](#personalization)
- [Troubleshooting](#troubleshooting)
- [Version History](#version-history)
- [Roadmap](#roadmap)
- [Contributing](#contributing)

## Features

**v0.41 Current Capabilities:**

- **Offline Chat:** Powered by `llama.cpp` with `Qwen2.5-Coder-1.5B-Instruct` (GGUF). No internet required for conversation.
- **Tool Integration:** Automatic intent detection for:
  - `get_weather(city)` - OpenWeatherMap API
  - `get_news(topic)` - NewsAPI
- **Advanced Memory (v0.40):**
  - SQLite FTS5 + Markdown hybrid storage
  - `memory.json` - Legacy short-term history (last 10 turns)
  - `memory/sally.db` - Persistent FTS5 index
  - `memory/core/human.json` - User profile (private, gitignored)
  - `memory/episodes/` - Long-term conversational episodes as markdown
  - API: `remember(content, type)` and `recall(query, k)` + CLI command `recall <query>`
- **Voice Interface:**
  - TTS: Piper TTS `en_US-lessac-medium` - Natural, fully offline
  - STT: `faster-whisper` tiny (39MB) - Offline after initial download
- **Multi-User Support (v0.41):** First-run onboarding. Automatically creates a private, local profile for each user. No hardcoded user data in repository.
- **API Server:** FastAPI server via `server.py` at `http://localhost:8000`
- **Cross-Platform:** Portable path resolution using `PROJECT_ROOT`. No hardcoded system paths.
- **Identity Protection:** Strong system prompt + JARVIS-to-SALLY output filter in `core/brain.py`

## Project Structure

```
SALLY/
├── core/
│ ├── brain.py # LLM initialization, tool routing, system prompts, identity filter
│ ├── config.py # Configuration loader, multi-user profile management, env variables
│ ├── tools.py # External tools: get_weather, get_news
│ ├── memory.py # Memory management: legacy JSON + SQLite FTS5 (remember/recall)
│ └── voice.py # TTS/STT: Piper + Whisper with lazy loading
├── memory/
│ ├── core/
│ │ ├── human.json # Private user profile (gitignored, auto-created)
│ │ └── human.json.example # Template for repository
│ ├── episodes/ # Markdown episode logs (gitignored)
│ └── sally.db # SQLite FTS5 database (gitignored)
├── models/
│ ├── Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf # LLM (gitignored, ~800MB)
│ └── voices/
│ ├── en_US-lessac-medium.onnx
│ └── en_US-lessac-medium.onnx.json
├── main.py # Main CLI: Enter=voice, Text=text, recall <query>, exit=quit
├── server.py # FastAPI server
├── requirements.txt # Python dependencies
├──.env.example # Environment template
├──.env # Private environment (gitignored)
└── README.md
```

## Installation

### Prerequisites

- Python 3.10+
- Git
- 4GB+ RAM recommended (2GB minimum for 0.5B model)

### 1. Clone Repository

```bash
git clone https://github.com/techma2004/SALLY.git
cd SALLY
```

### 2. Platform-Specific Setup

#### Linux - Debian 13 Trixie / Ubuntu 22.04+ (Recommended)

```bash
# System dependencies
sudo apt update
sudo apt install -y python3 python3-venv python3-pip libsndfile1 portaudio19-dev ffmpeg alsa-utils pulseaudio

# Start audio service
pulseaudio --start

# Verify audio devices
arecord -l
aplay -l

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Windows 10 / 11

**Requirements:** Install Python 3.10+ from python.org (check "Add Python to PATH" during installation)

```powershell
# 1. Install ffmpeg: https://ffmpeg.org/download.html
# Add ffmpeg bin folder to system PATH
# 2. Install Git: https://git-scm.com/download/win

# Clone (if not done)
git clone https://github.com/techma2004/SALLY.git
cd SALLY

# Create virtual environment
python -m venv venv
.\venv\Scripts\activate

# Upgrade pip
python -m pip install --upgrade pip
pip install -r requirements.txt
```

> **Windows Audio Note:** If microphone detection fails, SALLY automatically falls back to text mode. Ensure microphone permissions are enabled: Settings > Privacy & Security > Microphone.

#### macOS - Intel & Apple Silicon

```bash
# Install Homebrew if not installed: https://brew.sh
# Install dependencies
brew install python@3.11 ffmpeg portaudio

# Clone (if not done)
git clone https://github.com/techma2004/SALLY.git
cd SALLY

# Create virtual environment
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Android - Termux (Experimental)

```bash
# Use F-Droid Termux version (Play Store version is deprecated)
pkg update && pkg upgrade -y
pkg install -y python git ffmpeg portaudio termux-api
termux-setup-storage

# Clone
git clone https://github.com/techma2004/SALLY.git
cd SALLY

# Create venv
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Note: For 3GB RAM devices, use Qwen2.5-0.5B model instead of 1.5B
```

**requirements.txt**
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

> **Common Error:** `externally-managed-environment` - You forgot to activate venv. Run `source venv/bin/activate` (Linux/macOS) or `.\venv\Scripts\activate` (Windows) before `pip install`.

### 3. Download LLM Model

**Linux / macOS:**
```bash
mkdir -p models
wget -O models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q3_k_l.gguf
```

**Windows PowerShell:**
```powershell
mkdir models
Invoke-WebRequest -Uri https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct-GGUF/resolve/main/qwen2.5-coder-1.5b-instruct-q3_k_l.gguf -OutFile models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf
```

Optional: Larger model for better quality (1.1GB):
```bash
wget -O models/Qwen2.5-1.5B-Instruct-Q4_K_M.gguf https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct-GGUF/resolve/main/qwen2.5-1.5b-instruct-q4_k_m.gguf
```

### 4. Download Voice Models

**Linux / macOS:**
```bash
mkdir -p models/voices
cd models/voices
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx
wget https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json
cd../..
```

**Windows PowerShell:**
```powershell
mkdir models/voices
Invoke-WebRequest -Uri https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx -OutFile models/voices/en_US-lessac-medium.onnx
Invoke-WebRequest -Uri https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/lessac/medium/en_US-lessac-medium.onnx.json -OutFile models/voices/en_US-lessac-medium.onnx.json
```

> Whisper `tiny` (39MB) will auto-download on first voice use. Allow it to complete without interruption.

## Configuration

### 1. Create Environment File

```bash
cp.env.example.env
```

Linux/macOS: `nano.env`
Windows: `notepad.env`

### 2. Configure `.env`

```ini
# LLM Configuration
LLM_MODEL_PATH=models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf
LLM_N_CTX=4096
LLM_N_THREADS=8
LLM_TEMPERATURE=0.7

# Voice Configuration
VOICE_MODEL_PATH=models/voices/en_US-lessac-medium.onnx
WHISPER_MODEL_SIZE=tiny

# Optional APIs (for tools)
# Get free keys from:
# https://openweathermap.org/api
# https://newsapi.org
OPENWEATHER_API_KEY=your_key_here
NEWS_API_KEY=your_key_here
DEFAULT_CITY=Calabar
DEFAULT_COUNTRY=NG

# Author machine detection (optional, for development)
# SALLY_OWNER=Author
```

### 3. Secure Private Files

Ensure `.gitignore` contains:

```
.env
venv/
__pycache__/
models/*.gguf
models/voices/*.onnx
models/voices/*.onnx.json
models/whisper-*/
memory/
memory.json
memory/core/human.json
.DS_Store
```

Create public template:

```bash
mkdir -p memory/core
echo '{ "user_name": "User", "user_handle": "User", "user_location": "Unknown" }' > memory/core/human.json.example
```

## Usage

### Start SALLY

**Linux / macOS:**
```bash
source venv/bin/activate
python3 main.py
```

**Windows:**
```powershell
.\venv\Scripts\activate
python main.py
```

### First Run - Onboarding

**On Author's development machine** (detected via `SALLY_OWNER` or system path):

```
[CONFIG] SALLY v0.41 | User: Author | Model: Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf
--- SALLY v0.41 Memory + Voice ---
```

**On new user machine** (Windows, macOS, Linux clone):

```
--- SALLY First Run Setup ---
SALLY is personal. Let's make her yours (30 seconds, stored locally, never pushed).

Your name: John Doe
Handle/nickname [John Doe]: John
Location / OS: New York, Windows 11
Hardware/OS: Dell XPS 15
Interests: AI, Software Development, Gaming

Saved! SALLY now knows you as John. Profile is local and private.

[CONFIG] SALLY v0.41 | User: John | Model: Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf
--- SALLY v0.41 Memory + Voice ---
```

Reset profile: `rm memory/core/human.json` then restart `main.py`

### Interaction Modes

- **Press Enter:** Voice mode (6-second recording)
- **Type message:** Text mode with voice response
- **Commands:**
  - `recall <query>` - Search long-term memory (e.g., `recall python`)
  - `who am i` - Recall your profile
  - `what is your name` - Verify SALLY identity
  - `exit` / `quit` / `bye` - Save and quit

**Example Session:**
```
Press Enter for VOICE or type your message: weather in Calabar

SALLY: [TOOL] 28°C, 85% humidity in Calabar. Currently partly cloudy.

Press Enter for VOICE or type your message: recall hardware

[episodic] 2026-08-16: Tested voice on Debian 13 Trixie...

Press Enter for VOICE or type your message: exit
Goodbye. SALLY going offline.
```

### API Server

```bash
python3 server.py
# Server running at http://localhost:8000
# Docs at http://localhost:8000/docs
```

## Architecture

### Core Flow

1. **Configuration (`core/config.py`):**
   - `load_or_create_facts()` checks for `memory/core/human.json`
   - If not found and author machine detected → loads author profile template
   - If not found and new user → triggers 30-second onboarding → saves to `human.json` (gitignored)

2. **Main Loop (`main.py`):**
   - Loads legacy `memory.json` (last 10 turns) + new SQLite memory
   - `init_core_from_facts()` migrates profile to markdown + SQLite
   - For each input: `recall(query)` performs FTS5 search → injects as `[MEMORY CONTEXT]`

3. **Brain (`core/brain.py`):**
   - `detect_intent()` - Regex-based tool detection (weather/news)
   - Calls tools from `core/tools.py` if needed
   - Constructs prompt: `PERSONALITY` + memory context + conversation history + user input
   - `llm.create_chat_completion()` via llama.cpp
   - Post-process filter: Replaces "JARVIS" with "SALLY"

4. **Memory (`core/memory.py`):**
   - Legacy: `load_memory()` / `save_memory()` → `memory.json`
   - v0.40: `remember(content, type)` → SQLite + Markdown, `recall(query)` → FTS5

5. **Voice (`core/voice.py`):**
   - `listen(duration)` - sounddevice + faster-whisper
   - `speak(text)` - Piper TTS
   - Lazy loading to reduce startup time

**Path Handling:**
```python
PROJECT_ROOT = Path(__file__).parent.parent
VOICE_MODEL_PATH = PROJECT_ROOT / os.getenv("VOICE_MODEL_PATH")
```
Ensures portability across all platforms.

## Memory System

| Type | Purpose | Storage | Example |
|------|---------|---------|---------|
| `semantic` | Facts about user | `human.json` + `human.md` + SQLite | "User name is John, uses Windows 11" |
| `episodic` | Conversational history | `episodes/*.md` + SQLite | "User asked about weather on 2026-08-16" |
| `procedural` | Rules and preferences (future) | `rules.md` + SQLite | "User prefers concise answers" |

**Commands:**
```python
from core.memory import remember, recall

remember("User prefers Python over JavaScript", type="semantic")
results = recall("Python preferences", k=3)
```

## Personalization

**Problem Solved in v0.41:** Earlier versions hardcoded author data, causing confusion for other users ("Why is SALLY saying it's built for someone else?").

**Solution:**
- Repository contains only `memory/core/human.json.example` (generic template)
- Real profile `memory/core/human.json` is gitignored and private
- On first run:
  - **Author machine:** Detected via `SALLY_OWNER` env variable or known development path → loads author template
  - **New user:** Onboarding wizard (4 questions, 30 seconds) → creates personal profile
- Result: Each clone becomes YOUR SALLY

To switch users: `rm memory/core/human.json && python3 main.py`

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `externally-managed-environment` | Venv not activated | `source venv/bin/activate` (Linux/macOS) or `.\venv\Scripts\activate` (Windows) |
| `I am JARVIS` response | Corrupted memory | `rm -rf memory/ memory.json && echo "[]" > memory.json` then restart |
| `IndentationError in brain.py` | Copy-paste error | Re-clone file, ensure `str(LLM_MODEL_PATH)` used for path conversion |
| Whisper hangs on download | Interrupted first download | Delete `models/whisper-*` or `~/.cache/huggingface/` and retry, use `tiny` model |
| No sound / `PaAlsa` error (Linux) | PulseAudio not running | `pulseaudio --start` then `arecord -l` to list devices |
| No microphone (Windows) | Permission denied | Settings > Privacy & Security > Microphone > Enable for apps. SALLY falls back to text mode. |
| No microphone (macOS) | Permission denied | System Settings > Privacy & Security > Microphone > Enable Terminal |
| `git push rejected` | Remote ahead | `git pull --rebase origin main` then `git push` |
| Hardcoded path error | Old version | Update to v0.41, uses `PROJECT_ROOT` |
| `human.json not found` | First run | Will auto-create on first run. For author machine, set `SALLY_OWNER=Author` in `.env` |
| Model not found | Incorrect path in.env | Verify `LLM_MODEL_PATH` points to existing `.gguf` file relative to project root |

## Version History

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| v0.1 | 2025-06 | Initial offline LLM integration with llama.cpp | ✅ Completed |
| v0.2 | 2025-07 | Tool support (weather/news), basic memory with memory.json | ✅ Completed |
| v0.35 | 2025-08-10 | Voice integration (Piper TTS + Whisper STT), portable paths (PROJECT_ROOT), identity lock, anti-JARVIS filter | ✅ Completed |
| v0.36 | 2025-08-12 | Personality hardening, config improvements, VOICE_MODEL_PATH from env | ✅ Completed |
| v0.40 | 2025-08-16 | **Memory upgrade:** SQLite FTS5 + Markdown hybrid, remember()/recall() API, episodes storage, recall CLI command | ✅ Completed |
| v0.41 | 2025-08-18 | **Multi-user support:** Onboarding wizard, human.json gitignored, author detection, professional README, Windows/macOS/Linux/Termux install guides, fixes hardcoded user data issue | ✅ Current |
| v0.45 | Planned Q3 2026 | Hybrid search: all-MiniLM-L6-v2 ONNX embeddings (80MB) + sqlite-vec for semantic search | 🔄 Planned |
| v0.50 | Planned Q4 2026 | Sleep cycle: episode clustering, promote to rules.md, Memory-Git with reversible diffs, wake word "Hey SALLY" with openWakeWord | 📋 Planned |
| v0.60 | Planned Q1 2027 | Station OS: Linux desktop environment integration, system tray, autostart | 📋 Planned |
| v0.70 | Planned Q2 2027 | Android companion: Termux APK, background service, full JARVIS Mode | 📋 Planned |
| v0.80 | Planned Q3 2027 | RAG over local documents, file system indexing, project-aware context | 📋 Planned |

**Git Log:**
```bash
git log --oneline
# v0.41 multi-user + professional docs + cross-platform guides
# v0.40 SQLite FTS5 memory + markdown
# v0.36 identity hardening
# v0.35 voice + portable paths
```

## Roadmap

**Immediate (v0.45):**
- [ ] ONNX embeddings for semantic search (offline, <100MB)
- [ ] Hybrid keyword + semantic recall
- [ ] Memory consolidation background task

**Short-term (v0.50):**
- [ ] Wake word detection "Hey SALLY" via openWakeWord
- [ ] Sleep cycle: automatic memory summarization
- [ ] Desktop GUI (Tkinter/Electron)
- [ ] Memory-Git versioning

**Mid-term (v0.60 - v0.70):**
- [ ] Station OS integration
- [ ] Android companion app
- [ ] Voice customization
- [ ] Plugin system for custom tools

**Long-term (v0.80+):**
- [ ] Full RAG over local files
- [ ] Multi-modal (vision via LLaVA)
- [ ] Collaborative memory (opt-in, encrypted)

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/your-feature`
3. Ensure you never commit: `memory/`, `models/*.gguf`, `models/voices/*.onnx`, `.env`
4. Test on your platform: `python3 main.py` and verify onboarding + voice
5. Commit: `git commit -m "feat: your feature description"`
6. Push and create PR

**Guidelines:**
- Keep it offline-first
- Maintain portability (use `PROJECT_ROOT`)
- Add tests for memory functions if modifying `core/memory.py`
- Update version history in README

## Author

**Edima Bassey** - Independent Developer, AI Enthusiast
- Location: Calabar, Cross River State, Nigeria
- Focus: Offline AI, Linux Systems, Android, Robotics
- Project: SALLY - Personal AI Assistant & Station OS

Built with focus on privacy, modularity, and offline capability.

## License

MIT License - Free for personal and commercial use. See LICENSE file for details.

If you build your own SALLY, consider starring the repository and sharing your setup!

---

> "We are blessed — SALLY speaks." - Offline AI for everyone, everywhere.

⭐ Star this repository if you like private, offline AI!
