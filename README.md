# SALLY — Science Artificial Learning Logic and You

<p align="center">
  <img src="logo.png" width="400" alt="SALLY Logo" />
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/Platform-Linux%20%7C%20Windows%20%7C%20macOS-lightgrey" alt="Platform">
  <img src="https://img.shields.io/badge/Offline--first-Local%20LLM-2ECC71?style=flat" alt="Offline-first">
  <img src="https://img.shields.io/badge/Version-v0.2.0-blue?style=flat" alt="Version">
</p>

**SALLY is an offline-first, private-by-default personal AI assistant built around a modular agent runtime.** It runs a local GGUF model through `llama-cpp-python`, uses deterministic tools where possible, stores memory locally in SQLite/FTS5, and exposes a unified Gateway for its interfaces.

> The current runtime is designed around a lightweight local setup first, with optional external integrations available when explicitly configured.

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
- [Author](#author)

## Features

**v0.2.0 Current Runtime:**

- **Offline-first Chat:** Powered by a local GGUF model through `llama-cpp-python`. The default model is `llama-3.2-1b-instruct-q4_k_m.gguf`.
- **Deterministic Tooling:** Requests that can be handled without generation are routed directly to registered tools, including:
  - `calculator` - safe arithmetic and chained calculations
  - `datetime` - current date/time handling
  - `unit_convert` - common unit conversions
  - `scientific_constant` - known scientific constants
  - `science_calculate` - supported scientific calculations
- **Agent Routing:** Specialized routes for coding, testing/review, research/investigation, and general planning.
- **Persistent Memory:** SQLite with FTS5 search, structured user-model entries, sessions, tool usage, and conversation storage.
- **Conversation Storage:** Web conversations are persisted locally with user IDs, message history, timestamps, and conversation metadata.
- **Unified Gateway:** One runtime entry point coordinates requests across the available interfaces.
- **Web Workspace:** A React + Vite interface with real conversation, memory, tool, and system/runtime views.
- **Telegram Adapter:** Optional Telegram interface through the same Gateway.
- **WhatsApp Adapter:** Optional WhatsApp Cloud API webhook integration when the required credentials are configured.
- **Local Health and Runtime APIs:** FastAPI endpoints expose chat, conversations, memory, tools, health, system statistics, and integration webhooks.
- **Configurable Runtime:** Model path, context size, CPU threading, agent limits, memory database path, host, and port are read from `.env`.
- **No hardcoded personal profile:** Without a local profile file, SALLY falls back to generic `User` / `Unknown` values.

> Voice settings remain available as configuration placeholders, but voice is not the primary active interface in the current runtime. The current launcher starts the web interface by default.

## Project Structure

```text
SALLY/
├── core/
│ ├── agent/
│ │ ├── coordinator.py # Request coordination and tool/agent execution
│ │ ├── inference.py # Local LLM language generation
│ │ ├── protocol.py # Runtime action protocol
│ │ ├── registry.py # Agent registration
│ │ ├── router.py # Deterministic route selection
│ │ ├── runtime.py # Agent runtime and execution loop
│ │ ├── tools.py # Runtime tool bridge
│ │ └── types.py # Shared agent types
│ ├── gateway/
│ │ ├── gateway.py # Unified request gateway
│ │ ├── telegram.py # Optional Telegram adapter
│ │ ├── whatsapp.py # Optional WhatsApp Cloud API adapter
│ │ └── web.py # FastAPI + local web gateway
│ ├── memory/
│ │ ├── manager.py # High-level memory API
│ │ ├── models.py # Memory and conversation models
│ │ └── store.py # SQLite / FTS5 persistence
│ ├── science/
│ │ └── engine.py # Scientific calculation support
│ ├── tools/
│ │ ├── calculator.py # Safe arithmetic tool
│ │ ├── datetime_tool.py # Date/time tool
│ │ └── registry.py # Registered tool definitions
│ ├── config.py # Runtime settings and local user profile loader
│ └── llm.py # Local LLM loading
├── memory/
│ ├── MEMORY.md # Local memory notes
│ ├── USER.md # Local user notes
│ ├── episodes/ # Local episode storage
│ ├── history.json # Local conversation/history data
│ └── skills/ # Local skill data
├── models/
│ └── llama-3.2-1b-instruct-q4_k_m.gguf # Default local GGUF model (not committed)
├── web/
│ ├── src/ # React web application
│ ├── index.html
│ ├── package.json
│ └── vite.config.js
├── tests/ # Python test suite
├── main.py # Web / Telegram interface launcher
├── pyproject.toml # Python project and package configuration
├── uv.lock # Locked Python dependencies
├── .env.example # Environment template
├── .env # Private local configuration (gitignored)
├── logo.png
└── README.md
```

## Installation

### Prerequisites

- Python 3.11
- Git
- `uv`
- Node.js + npm (required to build the web interface)
- 4GB+ RAM recommended for comfortable local LLM use

### 1. Clone Repository

```bash
git clone https://github.com/techma2004/SALLY.git
cd SALLY
```

### 2. Create the Python Environment

SALLY uses `uv` for Python environment and dependency management.

```bash
uv venv --python 3.11
uv sync
```

You can also use the checked-in `.python-version` with normal `uv` commands.

### 3. Build the Web Interface

```bash
cd web
npm install
npm run build
cd ..
```

The FastAPI Gateway serves the generated web application from `web/dist/`.

### 4. Prepare the Local Model

The default model path is:

```text
models/llama-3.2-1b-instruct-q4_k_m.gguf
```

Place that GGUF file at the path above, or set `LLM_MODEL_PATH` in `.env` to another compatible local GGUF model.

Model files are intentionally excluded from Git because of their size.

### 5. Create Environment File

```bash
cp .env.example .env
```

Edit `.env` with your local settings.

> **Important:** If your existing `.env` contains an older `PORT=8080`, it will override the default. Set `PORT=5678` when you want the current default port.

## Configuration

### 1. Core Runtime Configuration

The current configuration is driven by `.env`:

```ini
# SALLY
SALLY_NAME=SALLY
SALLY_VERSION=0.2.0

# LLM
LLM_MODEL_PATH=models/llama-3.2-1b-instruct-q4_k_m.gguf
LLM_N_CTX=2048
LLM_N_THREADS=4
LLM_N_BATCH=128
LLM_N_GPU_LAYERS=0
LLM_TEMPERATURE=0.4
LLM_MAX_TOKENS=256
LLM_VERBOSE=false

# AGENT RUNTIME
AGENT_MAX_STEPS=3
AGENT_MAX_DEPTH=2
AGENT_MAX_SUBAGENTS=4
AGENT_TIMEOUT=120
AGENT_MAX_TOKENS=256
AGENT_TEMPERATURE=0.4

# MEMORY
MEMORY_ENABLED=true
MEMORY_DB_PATH=memory/memory.db
MEMORY_SEARCH_LIMIT=5

# SERVER
HOST=127.0.0.1
PORT=5678
```

### 2. Optional Integrations

WhatsApp can be enabled through the Cloud API adapter:

```ini
WHATSAPP_ENABLED=false
WHATSAPP_ACCESS_TOKEN=
WHATSAPP_PHONE_NUMBER_ID=
WHATSAPP_VERIFY_TOKEN=
WHATSAPP_APP_SECRET=
WHATSAPP_GRAPH_VERSION=
```

Optional voice-related settings are still present for future/adapter work:

```ini
VOICE_MODEL_PATH=models/voices/en_US-lessac-medium.onnx
WHISPER_MODEL_SIZE=tiny
```

### 3. Secure Private Files

Never commit local secrets, databases, model files, or private runtime state.

The repository's ignore rules cover the private environment and large model/database artifacts.

## Usage

### Start SALLY

The default launcher starts the local web interface:

```bash
uv run python main.py
```

By default, the Gateway listens on:

```text
http://127.0.0.1:5678
```

The exact host and port come from `.env`.

### Telegram Interface

The Telegram adapter uses the same runtime:

```bash
uv run python main.py telegram
```

Telegram credentials must be configured in the environment used by the adapter.

### Web Interface

After running the server, open:

```text
http://127.0.0.1:5678
```

The current web workspace includes:

- persistent conversations
- message history
- memory search and recent memory views
- registered tool information
- model and gateway health
- live system/runtime information
- response route and elapsed-time metadata

### HTTP API

FastAPI also exposes the underlying API. Useful endpoints include:

```text
POST /chat
GET  /conversations
GET  /conversations/{conversation_id}
GET  /tools
GET  /memory
GET  /memory/recent
GET  /health
GET  /system/stats
GET  /docs
```

WhatsApp webhook endpoints are available under:

```text
GET  /webhooks/whatsapp
POST /webhooks/whatsapp
```

### Development Checks

Python tests:

```bash
uv run pytest -q
```

Web production build:

```bash
cd web
npm install
npm run build
```

## Architecture

### Core Flow

The current runtime follows a Gateway → Coordinator → Router / Runtime model:

1. **Configuration (`core/config.py`):**
   - Loads local environment values from `.env`
   - Provides the runtime model, memory, agent, and server settings
   - Loads a generic local user profile when `memory/core/human.json` exists
   - Falls back to generic values when no profile is present

2. **Gateway (`core/gateway/`):**
   - Receives requests from web, Telegram, and WhatsApp
   - Normalizes the incoming request
   - Passes the request into the shared SALLY runtime

3. **Coordinator (`core/agent/coordinator.py`):**
   - Coordinates task handling
   - Uses the deterministic router for obvious requests
   - Executes registered tools directly when no LLM generation is needed
   - Uses the local LLM only where inference is required
   - Records memory and tool-related runtime state

4. **Router (`core/agent/router.py`):**
   - Detects calculations, date/time requests, conversions, constants, scientific calculations, coding, testing, research, and general planning
   - Routes specialized requests before falling back to a planning route

5. **Runtime (`core/agent/runtime.py`):**
   - Executes registered actions under runtime limits
   - Tracks active tasks and execution state
   - Provides the capability boundary for tools and agents

6. **Memory (`core/memory/`):**
   - Persists memories and structured user-model data
   - Stores conversations and messages in SQLite
   - Uses FTS5 for local memory search

7. **Local LLM (`core/llm.py`, `core/agent/inference.py`):**
   - Loads the configured GGUF model with `llama-cpp-python`
   - Converts verified runtime evidence into natural-language responses when inference is required

**Path Handling:**
```python
PROJECT_ROOT = Path(__file__).resolve().parent.parent
```

Runtime paths are resolved relative to the project root unless an absolute path is supplied.

## Memory System

The current memory layer is SQLite-based and managed through `MemoryManager` and `MemoryStore`.

| Type | Purpose | Storage |
|------|---------|---------|
| `fact` | Stable factual information | SQLite + FTS5 |
| `preference` | User preferences and choices | SQLite + FTS5 |
| `episode` | Important events and interactions | SQLite + FTS5 |
| `profile` | Structured user profile data | SQLite + user model |
| `daily` | Short-lived daily notes | SQLite + FTS5 |
| `session` | Session-level information | SQLite + FTS5 |

The same SQLite store also keeps:

- structured user-model entries
- sessions
- tool usage statistics
- conversations
- conversation messages

Example memory API:

```python
from core.memory import MemoryManager, MemoryType

memory = MemoryManager()

memory.remember(
    "User prefers Python",
    memory_type=MemoryType.PREFERENCE,
)

results = memory.search("Python preferences", limit=3)
```

## Personalization

The current runtime avoids hardcoded personal identity data.

If a local profile exists at:

```text
memory/core/human.json
```

SALLY can read:

- `user_name`
- `user_handle`
- `user_location`

Otherwise it safely falls back to generic values:

```text
User / Unknown
```

There is no active onboarding wizard in the current runtime.

To create a local profile manually, add a gitignored `memory/core/human.json` with only the information you want SALLY to use.

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Server starts on 8080 instead of 5678 | Existing `.env` overrides the default | Set `PORT=5678` in `.env`, then restart |
| Model not found | `LLM_MODEL_PATH` points to a missing file | Verify the GGUF file exists at the configured path |
| Web page says build not found | `web/dist` has not been generated | Run `cd web && npm install && npm run build` |
| `ModuleNotFoundError` after installation | Environment or package installation is incomplete | Run `uv sync` and verify the active interpreter with `uv run python -c "import core"` |
| Tests fail during collection | A development dependency is missing | Run `uv sync` and then `uv run pytest -q` |
| Requests that should be instant take a long time | The request may have been routed into LLM inference | Check the response `route` and `elapsed_ms` fields; deterministic tools should not require model generation |
| Gateway returns a 500 error | Runtime or integration configuration problem | Check the server terminal for the exception and verify the relevant `.env` values |
| WhatsApp webhook returns 503 | WhatsApp integration is not configured | Set the required WhatsApp Cloud API credentials and enable the adapter |
| Private data appears in Git status | Local runtime state is not ignored | Verify `.env`, database files, and model files are covered by `.gitignore` |

## Version History

| Version | Date | Changes | Status |
|---------|------|---------|--------|
| v0.1 | 2025-06 | Initial offline LLM integration with llama.cpp | ✅ Historical |
| v0.2 | 2025-07 | Tool support and basic memory | ✅ Historical |
| v0.35 | 2025-08-10 | Voice integration, portable paths, identity protection | ✅ Historical |
| v0.36 | 2025-08-12 | Personality hardening and configuration improvements | ✅ Historical |
| v0.40 | 2025-08-16 | SQLite FTS5 memory + Markdown-oriented memory workflow | ✅ Historical |
| v0.41 | 2025-08-18 | Multi-user onboarding and cross-platform documentation | ✅ Historical |
| v0.2.0 | 2026-09 | Modular agent runtime, Gateway architecture, deterministic tools, SQLite memory manager, React web workspace, unified interface configuration, `uv` environment | ✅ Current |

**Current runtime line:** The project version was reset during the architecture migration so the active package version (`0.2.0`) reflects the new runtime rather than the historical pre-migration feature set.

## Roadmap

**Immediate:**
- [ ] Conversation context continuity across turns
- [ ] Stronger evidence-grounded natural-language responses
- [ ] Strict capability enforcement across tools and agents

**Short-term:**
- [ ] Dynamic context and memory management
- [ ] Hardware-aware model/context management
- [ ] More deterministic tools for common tasks
- [ ] Better runtime diagnostics and failure reporting

**Mid-term:**
- [ ] Deeper Telegram and WhatsApp integration
- [ ] Optional voice interface restoration
- [ ] More advanced memory consolidation
- [ ] Expanded local project awareness

**Long-term:**
- [ ] Retrieval over local files
- [ ] Multimodal capabilities
- [ ] Broader plugin/tool ecosystem
- [ ] More complete desktop and Android interfaces

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Do not commit secrets, local databases, or model files
4. Run the Python tests: `uv run pytest -q`
5. Build the web interface: `cd web && npm install && npm run build`
6. Commit: `git commit -m "feat: your feature description"`
7. Push and create a pull request

**Guidelines:**
- Keep the project offline-first
- Prefer deterministic tools for deterministic tasks
- Keep runtime configuration in `.env`
- Keep active subsystems free of stale duplicate implementations
- Add tests when changing runtime behavior
- Update the README when the public runtime contract changes

## Author

**Edima Bassey** - Independent Developer, AI Enthusiast

- Project: SALLY - Personal AI Assistant

Built with a focus on privacy, modularity, portability, and offline-first capability.

## License

The repository currently does not declare a license file or GitHub license metadata.

---

> "We are blessed — SALLY speaks." - Offline AI for everyone, everywhere.

⭐ Star this repository if you like private, offline-first AI!
