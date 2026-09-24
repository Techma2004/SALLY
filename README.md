# SALLY — Science Artificial Learning Logic And You

SALLY is a lightweight, offline-first personal AI assistant built around a modular agent runtime.

The current runtime is centered on:

- a local LLM through `llama-cpp-python`
- a deterministic coordinator/router
- verified calculator and scientific tools
- SQLite FTS5 persistent memory
- a unified Gateway
- a local React web workspace
- optional Telegram and WhatsApp adapters

## Current stack

- Python 3.11
- FastAPI + Uvicorn
- React + Vite
- SQLite + FTS5
- `llama-3.2-1b-instruct-q4_k_m.gguf` by default
- `uv` for Python environment and dependency management

## Architecture

```
Browser / Telegram / WhatsApp
            |
          Gateway
            |
        Coordinator
       /     |      \
   Router  Memory   Tools
             |        |
             +---- Local LLM
```

The active Python packages live below `core/` and are discovered automatically by setuptools.

## Web interface

The local web workspace is served from:

```
http://127.0.0.1:5678
```

Start it with:

```bash
uv run python main.py
```

The web interface provides real views for:

- persistent conversations
- SQLite memory
- registered tools
- live system/runtime information
- model and gateway health

There is no terminal UI in the current architecture.

## Installation

Clone the repository and install with `uv`:

```bash
git clone https://github.com/techma2004/SALLY.git
cd SALLY
uv venv --python 3.11
uv sync
```

Copy the environment template when needed:

```bash
cp .env.example .env
```

The default model path is:

```
models/llama-3.2-1b-instruct-q4_k_m.gguf
```

Set `LLM_MODEL_PATH` in `.env` when using another local GGUF model.

## Development checks

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

## Configuration

Important local settings are read from `.env`.

```ini
SALLY_NAME=SALLY
SALLY_VERSION=0.2.0

LLM_MODEL_PATH=models/llama-3.2-1b-instruct-q4_k_m.gguf
LLM_N_CTX=2048
LLM_N_THREADS=4
LLM_N_BATCH=128
LLM_N_GPU_LAYERS=0
LLM_TEMPERATURE=0.4
LLM_MAX_TOKENS=256

MEMORY_ENABLED=true
MEMORY_DB_PATH=memory/memory.db
MEMORY_SEARCH_LIMIT=5

HOST=127.0.0.1
PORT=5678
```

The runtime configuration is the single source of truth for the web port and memory database location.

## Design rules

SALLY should not contain:

- fake telemetry
- demo conversations
- mock capabilities
- UI controls for unimplemented actions
- stale duplicate implementations of active subsystems

New capabilities should be implemented end-to-end through the Gateway and verified with tests before being exposed in the interface.

## Roadmap

Current engineering priorities:

1. conversation context continuity
2. evidence-grounded inference
3. strict capability enforcement
4. dynamic context and memory management
5. hardware-aware model management
6. deliberate tool expansion

SALLY is intentionally kept lightweight so the base system remains practical on low-resource hardware.
