# SALLY Development Track

**Project:** SALLY
**Meaning:** Science Artificial Learning Logic And You
**Developer:** Edima Bassey
**Repository:** `Techma2004/SALLY`
**Current Version:** `0.2.3`
**Runtime:** Python 3.11 + Rust/PyO3 native extension
**Primary Goal:** Lightweight, private, modular personal AI assistant for Linux, Termux, Android, and web environments.

---

## 1. Project Vision

SALLY is being developed as a personal AI assistant rather than a simple chatbot.

The long-term system is intended to provide:

* Natural conversation
* Local AI inference
* Persistent memory
* Persistent sessions
* Tool use
* Deterministic calculations
* Scientific reasoning
* System monitoring
* Reminders and scheduling
* Skills
* Terminal interaction
* Web interface
* Full terminal user interface
* Telegram and WhatsApp gateways
* Android/Termux support
* Voice capabilities
* Screen and camera capabilities
* Media and system control
* External integrations
* Optional MCP support
* Project and workspace management
* Security and permission enforcement
* Diagnostics and observability
* Hardware-aware model management
* Local-first operation
* Optional scalable database infrastructure

SALLY must remain lightweight, understandable, private, and maintainable throughout this expansion.

---

# 2. Engineering Philosophy

SALLY follows a senior-engineering principle:

> **Do less, but make every part meaningful, reliable, and efficient.**

The project should prefer:

* Less code
* Fewer dependencies
* Smaller abstractions
* Lazy loading
* Deterministic operations where possible
* Local-first operation
* Low RAM usage
* Low CPU overhead
* Fast startup
* Stable interfaces
* Clear module boundaries
* Reusable components
* Real capabilities
* Evidence-based testing

SALLY must not become unnecessarily large simply because another project has a feature.

Hermes is an architectural reference, not a template to copy.

---

# 3. Development Workflow

Every meaningful milestone follows:

**Build → Test → Document → Mark milestone → Commit → Continue**

Testing must be proportional to the change.

### Testing policy

* Small code change → targeted test
* Bug fix → targeted regression test
* New module → relevant module tests
* Database migration → migration/integration tests
* Architecture change → affected subsystem suite
* Cross-cutting change → broader suite
* Documentation-only change → no unnecessary test
* Release candidate → complete verification

SALLY should not waste resources running enormous test suites for unrelated changes.

---

# 4. Completed Foundation

## Core Runtime

* ✅ Modular coordinator and agent runtime
* ✅ Dynamic agent actions
* ✅ Agent routing architecture
* ✅ Deterministic calculator tool
* ✅ Deterministic datetime tool
* ✅ Scientific calculation engine
* ✅ Unit conversion
* ✅ Verified scientific constants
* ✅ Hidden inference layer
* ✅ Tool-grounded explanation flow
* ✅ SQLite FTS5 memory
* ✅ Unified Gateway
* ✅ Rust/Python hybrid native runtime
* ✅ PyO3 native extension
* ✅ Native system information
* ✅ CPU information
* ✅ Memory information
* ✅ Battery information
* ✅ Uptime information
* ✅ Platform detection
* ✅ Architecture detection
* ✅ Native current-time support
* ✅ Reproducible native builds with maturin

---

# 5. Model and Runtime Work

* ✅ Migrated project direction from pip toward `uv`
* ✅ Added `pyproject.toml`
* ✅ Added `uv.lock`
* ✅ Established Python `3.11`
* ✅ Added local workspace dependency management
* ✅ Added Rust/PyO3 workspace
* ✅ Added `llama-cpp-python`
* ✅ Switched SALLY's intended local model from Qwen to Llama
* ✅ Selected local Llama 3.2 1B GGUF as the low-resource model
* ✅ Established lazy LLM loading
* ✅ Added model loading protection with a lock
* ✅ Added configurable context length
* ✅ Added configurable thread count
* ✅ Added configurable batch size
* ✅ Disabled GPU layers by default for the low-resource target
* ✅ Added bounded generation settings
* ✅ Kept model loading out of startup until inference is required

Current default model:

`models/llama-3.2-1b-instruct-q4_k_m.gguf`

The Qwen model remains historical project material but is no longer the intended default model.

---

# 6. Gateway and Interface Work

## Web

* ✅ FastAPI gateway
* ✅ React web workspace
* ✅ Persistent browser conversations
* ✅ Real memory view
* ✅ Real tools view
* ✅ Real system view
* ✅ Real health/runtime view
* ✅ Local-first runtime status
* ✅ Responsive three-panel workspace
* ✅ Fixed local web port `5678`

## Messaging

* ✅ Telegram adapter
* ✅ WhatsApp adapter

The adapters are intended to remain thin platform layers around the shared SALLY core.

---

# 7. Conversation Architecture

SALLY originally routed ordinary conversation through the agent/action path.

This caused normal messages to be treated like planning/action requests.

The architecture was corrected to distinguish:

```text
TOOL
AGENT
CHAT
```

The intended flow is now:

```text
User
 │
 ▼
Router
 ├── TOOL  ──► deterministic tool
 │
 ├── AGENT ──► agent runtime/action loop
 │
 └── CHAT  ──► direct conversational LLM response
```

Normal conversation must not unnecessarily produce JSON action objects.

The conversational system prompt establishes SALLY as:

* Friendly
* Intelligent
* Helpful
* Natural
* Direct

The chat path must not claim capabilities that SALLY does not actually possess.

---

# 8. Native Runtime

SALLY now contains a Rust/Python hybrid system layer.

The native extension provides:

* Platform information
* CPU architecture
* Current time
* System information
* System status
* Battery information

The native package is:

`sally-native`

It is built using:

* Rust
* PyO3
* maturin

The native component is part of the `uv` workspace.

---

# 9. Documentation and Project Protection

* ✅ README updated for current architecture
* ✅ Native runtime documented
* ✅ `uv sync` workflow documented
* ✅ Workspace structure documented
* ✅ Lazy LLM behavior documented
* ✅ Private-project contribution workflow documented
* ✅ Public-fork language removed
* ✅ Private development policy documented
* ✅ Custom SALLY private license added

SALLY currently uses a private/proprietary development license.

The project is **not currently intended to be open sourced**.

Third-party components retain their own licenses.

Hermes-derived architectural ideas must not be copied wholesale.

---

# 10. Git and Release State

Important completed commits include:

* `4d95d47` — added SALLY private license
* `71524e8` — synchronized native workspace and updated project documentation

The native workspace and documentation were successfully pushed to:

`Techma2004/SALLY`

The project has also undergone successful verification with:

* `uv sync`
* Native import verification
* Pytest
* `git diff --check`

Latest verified broad test result after the native workspace work:

**61 tests passed**

There was one dependency-side deprecation warning related to Starlette/AnyIO. It was not a SALLY test failure.

---

# 11. Hermes Research

Hermes Agent was studied as an architectural reference.

The goal is not to reproduce Hermes.

The goal is to identify architectural ideas that can improve SALLY while preserving:

* SALLY's identity
* SALLY's lightweight design
* SALLY's privacy
* SALLY's security model
* SALLY's local-first philosophy
* SALLY's low-resource requirements

Important Hermes concepts examined include:

* Conversation loops
* Context compression
* Persistent scheduling
* Platform adapters
* API/session architecture
* MCP integration
* Modular tool systems
* Gateway abstraction
* Long-running agent infrastructure
* State persistence
* Project/workspace concepts
* Skills
* Observability
* Terminal workflows

The conclusion is:

> **SALLY selectively adopts appropriate architectural ideas from Hermes while preserving SALLY's own identity, lightweight design, security model, and implementation.**

---

# 12. Current Architecture Direction

The target architecture is:

```text
SALLY
│
├── core/
│   ├── agent/
│   ├── context/
│   ├── memory/
│   ├── session/
│   ├── database/
│   ├── tools/
│   ├── skills/
│   ├── scheduler/
│   ├── gateway/
│   ├── tui/
│   └── llm.py
│
├── native/
├── skills/
├── data/
├── migrations/
└── tests/
```

The architecture is intentionally modular so that:

* The LLM is replaceable
* Gateways are replaceable
* Storage backends are replaceable
* Tools are independently testable
* Skills are independently loadable
* Sessions are independent from UI
* Memory is independent from the model
* Native capabilities remain isolated
* Android can reuse the same core
* Web and TUI can share the same backend

---

# 13. Milestone Roadmap

## M0 — Foundation

**Status: ✅ Completed**

Establish:

* Python project
* `uv`
* configuration
* package structure
* Git workflow
* testing foundation
* private licensing
* documentation foundation
* low-resource engineering rules

---

## M1 — AI Runtime

**Status: 🔄 Implemented / continuing refinement**

Establish:

* Local Llama runtime
* Lazy model loading
* Configurable inference
* LLM abstraction
* Conversation generation
* Model configuration
* Resource-aware inference

Future refinement:

* Model discovery
* Model metadata
* Hardware-aware configuration
* Benchmarking
* Model switching
* Optional Ollama integration

---

## M2 — Native Runtime

**Status: ✅ Completed**

Establish:

* Rust/PyO3 integration
* Native system information
* CPU information
* Memory information
* Battery information
* Platform information
* Architecture information
* Native build reproducibility

---

## M3 — Agent Architecture

**Status: 🔄 Next major architecture milestone**

Build:

* Dedicated conversation loop
* Clean turn lifecycle
* Tool dispatch
* Agent action loop
* Agent state
* Agent registry
* Capability-aware execution
* Better inference boundaries
* Error handling
* Retry boundaries
* Final response handling

Target:

```text
Input
 ↓
Router
 ↓
Conversation / Agent Loop
 ↓
Tool or LLM
 ↓
Validation
 ↓
Memory / Session
 ↓
Response
```

---

## M4 — Database Architecture

**Status: 📋 Planned**

Build a proper persistence layer supporting:

### SQLite

Primary local/offline database.

Used for:

* Local SALLY installations
* Termux
* Desktop
* Android
* Single-user deployments
* Lightweight operation

### PostgreSQL

Optional scalable backend.

Used where appropriate for:

* Larger deployments
* Multiple clients
* Remote infrastructure
* Higher concurrency
* Server environments

The database layer must abstract storage from the rest of SALLY.

---

## M5 — Persistent Sessions

**Status: 📋 Planned**

Implement:

* Session IDs
* Conversation records
* Message records
* Session metadata
* Session timestamps
* Session restoration
* Session deletion
* Session listing
* Session summaries
* Gateway-independent sessions

Web, TUI, terminal, Telegram, WhatsApp, and Android should ultimately be able to use the same session system.

---

## M6 — Memory

**Status: 🔄 Basic SQLite FTS5 memory exists**

Expand memory into:

* Short-term conversation memory
* Long-term memory
* User preferences
* Important facts
* Session summaries
* Search/retrieval
* Memory ranking
* Memory expiration where appropriate
* Memory privacy controls

Memory must not blindly save every message.

---

## M7 — Context Management

**Status: 📋 Planned**

Build:

* Context budgets
* Recent-message protection
* Important-memory protection
* Tool-output management
* Conversation summarization
* Context compression
* Token-aware assembly
* Long-session handling

The system should preserve important context while preventing uncontrolled context growth.

---

## M8 — Full TUI

**Status: 📋 Planned**

The TUI is intentionally planned as a first-class interface.

The previous lightweight web-only direction should not be interpreted as permanent removal of the TUI.

The future TUI should support:

* Conversations
* Sessions
* Tools
* Skills
* Memory
* System status
* Tasks
* Reminders
* Model information
* Settings
* Project/workspace selection
* Logs where appropriate
* Keyboard-first navigation

It must remain lightweight enough for the target hardware.

---

## M9 — Web Interface

**Status: 🔄 Existing implementation; future refinement**

Continue improving:

* Chat
* Sessions
* Memory
* Tools
* System status
* Health
* Settings
* Streaming
* Authentication where required
* Responsive layout
* Mobile usability

The web UI must expose real functionality only.

---

## M10 — Tools

**Status: 🔄 Basic tools implemented**

Existing:

* Calculator
* Datetime
* Scientific calculations
* Unit conversion
* Verified constants
* System information

Future:

* File operations
* Safe terminal operations
* Search
* Network tools
* Media control
* Device operations
* Other capabilities as required

Every tool must have:

* Clear input schema
* Validation
* Permission boundaries
* Error handling
* Deterministic behavior where possible
* Tests

---

## M11 — Skills

**Status: 📋 Planned**

Create a lightweight skill system.

Skills should:

* Be modular
* Be discoverable
* Have metadata
* Declare capabilities
* Load lazily
* Avoid unnecessary dependencies
* Be independently testable

Skills must not silently obtain capabilities they were not granted.

---

## M12 — Scheduler and Reminders

**Status: 📋 Planned**

Build:

* Reminders
* Scheduled tasks
* Recurring tasks
* Persistent jobs
* Missed-job handling
* Safe background execution
* SQLite-backed scheduling
* Optional scalable scheduling

The scheduler should remain lightweight.

---

## M13 — Security and Permissions

**Status: 🔄 Ongoing / foundational work**

Build a capability-based security model.

Important rules:

* User input is untrusted
* Tool arguments are untrusted
* Client metadata is untrusted
* Model output is untrusted
* Tools are the source of truth for capabilities
* SALLY must never claim an action happened unless the tool confirms it
* Dangerous operations require explicit boundaries
* External integrations require explicit configuration
* Secrets must never be hardcoded
* Logs must avoid leaking secrets
* Permissions must be enforced below the model layer

---

## M14 — Gateway Architecture

**Status: 🔄 Existing gateways; architecture refinement planned**

Maintain a common core with thin adapters for:

* Web
* Terminal
* TUI
* Telegram
* WhatsApp
* Android
* Future integrations

Gateways should translate platform-specific input/output rather than implement SALLY's intelligence themselves.

---

## M15 — Android and Termux

**Status: 📋 Planned**

SALLY should eventually run on Android using a practical lightweight architecture.

Targets include:

* Termux
* Android application
* Shared Python core where practical
* Local model support
* Android system APIs
* Termux:API
* Notifications
* Battery information
* Device information
* Media control
* Camera/screen capabilities where permitted

A future Android application may use a native Android shell around the SALLY core rather than duplicating all logic.

Chaquopy has previously been considered for Android testing.

---

## M16 — Voice

**Status: 📋 Planned**

Future capabilities:

* Speech-to-text
* Text-to-speech
* Voice conversations
* Wake/activation mechanisms where practical
* Local/offline options where hardware permits

Voice should remain optional rather than becoming a mandatory base dependency.

---

## M17 — Screen and Camera

**Status: 📋 Planned**

Future capabilities may include:

* Screen understanding
* Screenshot analysis
* Camera input
* Image understanding
* Visual assistance

These capabilities must be permission-controlled and platform-aware.

---

## M18 — Media and System Control

**Status: 📋 Planned**

Future SALLY capabilities:

* Music control
* Sound playback
* Media controls
* Volume control where supported
* Device status
* Battery information
* Android system integrations
* Desktop system integrations

---

## M19 — MCP and External Integrations

**Status: 💡 Evaluated / planned**

MCP may be added as an optional integration layer.

Important rule:

> MCP must not become a mandatory SALLY dependency.

External integrations should be:

* Optional
* Lazy-loaded
* Permission-controlled
* Isolated
* Failure-tolerant

---

## M20 — Advanced Agent Features

**Status: 📋 Planned**

Future capabilities may include:

* Multi-step tasks
* Planning
* Task state
* Controlled tool loops
* Better retry behavior
* Task cancellation
* Background tasks
* Approval boundaries
* Task progress
* Agent handoff where genuinely useful

Complexity must be justified by real functionality.

---

## M21 — Projects and Workspaces

**Status: 📋 Planned**

SALLY should eventually understand project/workspace boundaries.

A workspace may contain:

* Sessions
* Memory
* Files
* Skills
* Tools
* Configuration
* Tasks
* Project-specific instructions

This will allow SALLY to support separate development projects without mixing their context.

---

## M22 — Observability and Diagnostics

**Status: 📋 Planned**

Add lightweight diagnostics:

* Runtime health
* Model status
* Tool execution status
* Database health
* Gateway health
* Memory usage
* CPU usage
* Errors
* Startup diagnostics
* Version information

Observability must not become excessive background overhead.

---

## M23 — Deployment

**Status: 📋 Planned**

Support practical deployment targets:

* Linux desktop
* Termux
* Android
* Local web server
* Optional remote server
* Optional PostgreSQL infrastructure

Deployment should be reproducible.

---

## M24 — Quality and Release Engineering

**Status: 📋 Final milestone**

Establish:

* Versioning
* Migration strategy
* Release checklist
* Regression testing
* Documentation verification
* Dependency auditing
* Security review
* Performance review
* Database migration verification
* Native build verification
* Clean installation verification
* Upgrade testing

Final SALLY releases must be reproducible and documented.

---

# 14. Final SALLY Architecture

The intended mature architecture is:

```text
                         ┌──────────────────┐
                         │      USERS       │
                         └────────┬─────────┘
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
             ▼                    ▼                    ▼
          Web UI                 TUI               Android
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                           Gateway Layer
                                  │
                           Session Manager
                                  │
                         Conversation Loop
                                  │
                              Router
                     ┌────────────┼────────────┐
                     │            │            │
                   CHAT         AGENT         TOOL
                     │            │            │
                     └────────────┼────────────┘
                                  │
                         Capability Layer
                                  │
             ┌────────────────────┼────────────────────┐
             │                    │                    │
           Tools                Skills             Integrations
             │                    │                    │
             └────────────────────┼────────────────────┘
                                  │
                         Context Manager
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                 Memory                     Sessions
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                              Database
                         ┌────────┴────────┐
                         │                 │
                       SQLite          PostgreSQL
                         │
                         ▼
                       Storage

                         LLM Runtime
                             │
                    Local / Optional Remote
                             │
                       Native Runtime
                             │
                         Rust / PyO3
```

---

# 15. Definition of Done

SALLY 1.0 is not defined simply by having many features.

SALLY 1.0 means:

* Core architecture is stable
* Conversation works naturally
* Agent execution is controlled
* Tools are real
* Memory is persistent
* Sessions are persistent
* Context is managed
* SQLite works reliably
* PostgreSQL integration is clean where needed
* TUI works
* Web works
* Termux works
* Android architecture is practical
* Voice is optional
* Screen/camera capabilities are permission-controlled
* Scheduler works
* Skills work
* Gateways are modular
* Security boundaries are enforced
* Native system integration is stable
* Diagnostics are useful
* Documentation matches reality
* Installation is reproducible
* Tests cover important behavior
* Performance remains appropriate for low-resource hardware
* No fake capabilities exist

---

# 16. Permanent Rules

### Rule 1

Do not add complexity without a reason.

### Rule 2

Do not claim a capability that has not actually been implemented.

### Rule 3

Do not replace deterministic logic with an LLM unnecessarily.

### Rule 4

Do not let model output override verified tool results.

### Rule 5

Do not expose secrets.

### Rule 6

Do not blindly store everything in memory.

### Rule 7

Do not make optional capabilities mandatory dependencies.

### Rule 8

Do not copy Hermes wholesale.

### Rule 9

Do not sacrifice SALLY's identity for architectural similarity.

### Rule 10

Do not run expensive tests unrelated to the change.

### Rule 11

Every major architectural change must leave SALLY runnable.

### Rule 12

Documentation must describe the actual implementation, not the desired implementation.

### Rule 13

Keep SALLY private unless the licensing strategy is deliberately changed.

### Rule 14

Prefer small, maintainable modules over giant abstractions.

### Rule 15

Performance, stability, privacy, and correctness take priority over feature count.

---

# 17. Current State

SALLY has moved beyond the initial chatbot prototype.

The project now has:

* A modular Python core
* Local Llama inference
* Agent infrastructure
* Tool infrastructure
* Scientific capabilities
* SQLite memory
* FastAPI
* React
* Telegram
* WhatsApp
* Rust/PyO3 native integration
* System monitoring
* `uv` workspace management
* Private licensing
* Documentation
* Automated tests
* A defined long-term architecture

The next major engineering focus is:

**M3 → Agent Architecture**

followed by:

**M4 → Database Architecture**
**M5 → Persistent Sessions**
**M6 → Memory**
**M7 → Context Management**
**M8 → Full TUI**

The project should then continue through the remaining milestones without abandoning the lightweight, private, efficient design that defines SALLY.
