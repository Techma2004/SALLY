# SALLY — Hermes-Inspired Upgrade Plan

**Project:** SALLY
**Meaning:** Science Artificial Learning Logic And You
**Developer:** Edima Bassey
**Repository:** `Techma2004/SALLY`
**Current Version:** `0.2.3`
**Status:** Private Development
**Primary Runtime:** Python 3.11
**Native Runtime:** Rust + PyO3
**Primary Local Model:** Llama 3.2 1B GGUF

---

# 1. Purpose

This document defines the long-term engineering direction of SALLY.

SALLY is being developed as a lightweight personal AI assistant capable of running across:

* Linux
* Termux
* Android
* Web interfaces
* Terminal interfaces
* Messaging gateways

The project takes selected architectural inspiration from Hermes Agent and other mature agent systems.

However:

> **SALLY is not a Hermes clone.**

Hermes is a source of architectural ideas.

SALLY retains its own:

* Identity
* Architecture
* Security model
* Private development model
* Lightweight philosophy
* Local-first philosophy
* Hardware constraints
* User experience
* Implementation

---

# 2. Core Engineering Objective

Build an AI assistant that is:

* Fast
* Lightweight
* Modular
* Private
* Reliable
* Secure
* Understandable
* Extensible
* Local-first
* Resource-conscious

The objective is not to build the largest assistant.

The objective is to build the smallest architecture capable of reliably providing the required capabilities.

---

# 3. Engineering Principles

## 3.1 Less code, more capability

Avoid unnecessary abstraction.

Prefer:

```text
simple module
→ clear interface
→ real capability
→ focused tests
```

over:

```text
large framework
→ multiple abstraction layers
→ unnecessary dependencies
→ difficult debugging
```

---

## 3.2 Local first

SALLY should work locally whenever practical.

Local components include:

* LLM inference
* Memory
* Sessions
* SQLite
* Native system information
* Tools
* TUI
* Web server
* Scheduler

Remote services should be optional.

---

## 3.3 Lazy loading

Heavy components must not load until required.

Examples:

* LLM
* Optional integrations
* MCP
* Voice
* Vision
* External APIs

This reduces startup time and RAM usage.

---

## 3.4 Deterministic where possible

If a task can be solved reliably without an LLM, use deterministic code.

Examples:

* Calculator
* Datetime
* Unit conversion
* Scientific constants
* System information
* Database operations
* Scheduling

The LLM should not unnecessarily replace deterministic functionality.

---

# 4. Existing SALLY Architecture

Current core areas include:

```text
core/
├── agent/
├── config.py
├── gateway/
├── llm.py
├── memory/
├── native/
├── science/
├── tooling.py
└── tools/
```

The project also contains:

```text
native/
web/
tests/
models/
```

The native package is integrated through the `uv` workspace.

---

# 5. Completed Development History

## Project modernization

Completed:

* `uv` migration direction
* `pyproject.toml`
* `uv.lock`
* Python 3.11 environment
* Local workspace architecture
* Native Rust package integration
* Reproducible native build process

---

## AI runtime

Completed:

* Local GGUF inference
* Llama 3.2 1B target
* Lazy LLM initialization
* Model locking
* Configurable inference parameters
* Low-resource defaults

---

## Agent foundation

Completed:

* Coordinator
* Router
* Agent runtime
* Agent registry
* Agent protocol
* Dynamic actions
* Inference layer
* Tool infrastructure

---

## Deterministic intelligence

Completed:

* Calculator
* Datetime
* Scientific calculations
* Unit conversion
* Verified scientific constants

---

## Memory

Completed:

* SQLite
* FTS5
* Persistent memory foundation

Future memory architecture will expand this foundation rather than discard it.

---

## Native system layer

Completed:

* Rust
* PyO3
* Native system status
* CPU information
* Memory information
* Battery information
* Uptime
* Platform
* Architecture
* Current time

---

## Gateways

Completed foundations:

* FastAPI/Web
* Telegram
* WhatsApp

Future interfaces must reuse the same core.

---

## Web application

Completed:

* React workspace
* Persistent browser conversations
* Memory interface
* Tools interface
* System interface
* Health interface
* Responsive layout
* Local runtime status
* Port `5678`

---

## Security and privacy

Completed:

* Private project development model
* Custom SALLY Private Software License
* Secret/configuration separation
* Security-focused development work
* Capability-oriented architecture direction

Security remains an ongoing milestone rather than a one-time feature.

---

# 6. Hermes Architectural Research

Hermes was inspected to understand how a mature agent system separates responsibilities.

The most useful architectural ideas identified were:

### Conversation loop

A dedicated lifecycle for each user turn.

### Context compression

Context should be actively managed instead of allowing unlimited conversation history.

### Scheduler

Persistent scheduling should be independent of the conversational interface.

### Platform adapters

External platforms should be adapters around the core rather than separate implementations of the assistant.

### API/session architecture

Sessions and execution state should be separate from individual UI requests.

### MCP

External tool ecosystems can be integrated behind an optional adapter.

### Modular tools

Capabilities should be registered and executed through controlled interfaces.

### Skills

Reusable functionality should be loadable independently.

### State

Long-running systems require explicit persistent state.

### Observability

The runtime should expose useful health and diagnostic information without excessive overhead.

---

# 7. What SALLY Will Not Copy

SALLY will not copy:

* Hermes' entire source tree
* Hermes' entire gateway architecture
* Hermes' complete API server
* Hermes' large conversation loop implementation
* Hermes' complete context compressor
* Hermes' complete scheduler
* Hermes' complete MCP implementation
* Hermes' full feature set

Instead, SALLY will implement smaller equivalents appropriate to its own requirements.

---

# 8. M3 — Agent Architecture

The current agent runtime should evolve into a clear turn-based architecture.

Target:

```text
User Input
    ↓
Router
    ↓
Conversation Loop
    ↓
Intent / Capability Decision
    ↓
Tool or LLM
    ↓
Tool Validation
    ↓
Context Update
    ↓
Memory / Session Update
    ↓
Final Response
```

The conversation loop should own the lifecycle of a turn.

It should not become a giant class.

---

# 9. M4 — Database Architecture

SALLY needs a clean persistence boundary.

The application should not directly depend on SQL statements throughout the codebase.

Target:

```text
Application
     ↓
Repositories
     ↓
Database Layer
     ↓
SQLite / PostgreSQL
```

### SQLite

Default.

Best for:

* Personal use
* Desktop
* Termux
* Android
* Offline operation

### PostgreSQL

Optional.

Best for:

* Server deployments
* Multiple users
* Larger workloads
* Remote deployments
* Higher concurrency

The database abstraction should prevent the rest of SALLY from caring which backend is active.

---

# 10. M5 — Sessions

Sessions should become first-class objects.

A session should contain:

* ID
* Title
* Created time
* Updated time
* Status
* Metadata
* Message history
* Summary
* Workspace/project association

Sessions should be available to:

* Web
* TUI
* Terminal
* Telegram
* WhatsApp
* Android

---

# 11. M6 — Memory

Memory should be divided conceptually.

### Working memory

Current conversation context.

### Session memory

Important information from the current conversation.

### Long-term memory

Facts intentionally retained for future conversations.

### Retrieved memory

Relevant memories selected for a new request.

### System memory

SALLY configuration and runtime state.

SALLY should not treat all conversation history as permanent memory.

---

# 12. M7 — Context Management

Long conversations require controlled context assembly.

Target:

```text
System instructions
+
Relevant long-term memory
+
Session summary
+
Recent messages
+
Current user request
+
Required tool information
```

When context becomes too large:

1. Protect system instructions.
2. Protect important memories.
3. Protect recent messages.
4. Remove unnecessary tool output.
5. Summarize older conversation.
6. Rebuild the context.

This should be significantly smaller than Hermes' full context-management system.

---

# 13. M8 — Full TUI

The TUI is a planned first-class SALLY interface.

It should eventually provide:

* Chat
* Session management
* Memory
* Tools
* Skills
* Tasks
* Reminders
* System status
* Model status
* Settings
* Projects
* Workspaces

The TUI must remain usable on low-resource systems.

---

# 14. M9 — Web

The existing React/FastAPI system becomes the web presentation layer.

Future improvements:

* Streaming responses
* Session management
* Better conversation UX
* Memory controls
* Tool status
* Skills
* Scheduler
* Projects
* Authentication where required
* Mobile-friendly interface

The web client must never become the source of truth for permissions.

The backend must enforce capabilities.

---

# 15. M10 — Tools

Every tool should have a lifecycle:

```text
Discover
 ↓
Validate
 ↓
Authorize
 ↓
Execute
 ↓
Validate Result
 ↓
Return Result
```

The model does not directly control the operating system.

The model requests a capability.

The capability layer decides whether it can execute.

---

# 16. M11 — Skills

Skills should provide higher-level reusable behavior.

Example conceptual structure:

```text
skills/
├── system/
├── productivity/
├── development/
├── science/
└── media/
```

A skill should declare:

* Name
* Description
* Required capabilities
* Optional dependencies
* Entry point
* Configuration

Skills should be loaded only when necessary.

---

# 17. M12 — Scheduler

The scheduler should support:

* One-time reminders
* Recurring reminders
* Scheduled tasks
* Persistent jobs
* Job status
* Failure handling
* Cancellation

The scheduler must survive application restarts.

SQLite is the default local persistence layer.

---

# 18. M13 — Security

Security is an architectural layer.

The system must distinguish:

```text
User request
≠
Model output
≠
Tool authorization
≠
Tool result
```

The model cannot grant itself permissions.

Examples:

```text
"Delete this file"
```

must not automatically mean:

```text
permission granted
```

The tool layer must decide.

Secrets must be stored outside source code.

Logs must not expose secrets.

External integrations must be explicitly configured.

---

# 19. M14 — Gateway System

All interfaces should converge on the same core.

```text
Web ───────┐
TUI ───────┤
Terminal ──┤
Telegram ──┤
WhatsApp ──┤
Android ───┤
            ▼
       SALLY Core
```

This prevents each platform from developing its own separate intelligence.

---

# 20. M15 — Android and Termux

SALLY's Android direction is:

```text
Android
   │
   ├── Native Android shell
   │
   ├── Termux
   │
   └── Shared SALLY core
```

Possible capabilities:

* Local model
* Notifications
* Battery
* Device information
* Termux:API
* Media control
* Voice
* Camera
* Screen access

Android-specific functionality must remain optional to the core.

Chaquopy has previously been considered as an implementation route for Android testing.

---

# 21. M16 — Voice

Voice should remain modular.

Possible architecture:

```text
Microphone
   ↓
Speech-to-Text
   ↓
SALLY Core
   ↓
Response
   ↓
Text-to-Speech
   ↓
Speaker
```

Neither STT nor TTS should become mandatory dependencies for basic SALLY operation.

---

# 22. M17 — Vision

Future visual capabilities may include:

* Screenshots
* Camera frames
* Image analysis
* Screen understanding

Vision must be:

* Permission controlled
* Explicit
* Resource aware
* Optional

---

# 23. M18 — Media and System Control

SALLY may eventually control:

* Music
* Audio playback
* Media players
* Volume
* Device status
* Android controls
* Desktop controls

These should be implemented as tools rather than hardcoded into the conversational layer.

---

# 24. M19 — MCP

MCP may provide a bridge to external capabilities.

However:

```text
MCP = optional
```

not:

```text
MCP = SALLY core
```

The MCP layer should be lazy-loaded and isolated.

---

# 25. M20 — Advanced Agents

Future agent capabilities may include:

* Planning
* Multi-step execution
* Task state
* Background work
* Cancellation
* Approval gates
* Retry policies
* Progress reporting

The architecture must prevent an agent loop from becoming an uncontrolled autonomous process.

---

# 26. M21 — Projects and Workspaces

Projects should isolate:

* Sessions
* Memory
* Files
* Instructions
* Skills
* Tools
* Tasks

Example:

```text
SALLY
├── Personal
├── SALLY Development
├── UNICTO
├── NNSS Calabar
└── Bamesie Book
```

Each workspace can eventually have its own context and configuration.

---

# 27. M22 — Observability

SALLY should expose useful diagnostics.

Examples:

```text
SALLY version
Python version
Model
Model status
Database status
Memory usage
CPU usage
Battery
Gateway status
Tool status
Active session
```

Diagnostics should be lightweight.

---

# 28. M23 — Deployment

Supported targets should eventually include:

### Linux

Primary development platform.

### Termux

Lightweight mobile/Android environment.

### Android

Dedicated application.

### Local server

LAN/local web deployment.

### Optional remote deployment

For environments where PostgreSQL and remote access are useful.

---

# 29. M24 — Release Engineering

Before a major release:

### Code

* Remove dead code
* Verify imports
* Verify dependency graph
* Check migrations
* Check security boundaries

### Tests

Run:

* Targeted tests
* Regression tests
* Integration tests
* Full suite when justified

### Runtime

Verify:

* Startup
* Model loading
* Tool execution
* Memory
* Sessions
* Gateway
* Database
* Native extension

### Documentation

Verify:

* README
* Track
* Upgrade Plan
* Configuration
* Installation
* Architecture
* Migration notes

### Git

Verify:

```text
git status
git diff --check
git log
```

Then create the release commit/tag as appropriate.

---

# 30. Testing Strategy

SALLY deliberately avoids indiscriminate testing.

The test strategy is:

```text
Change
 ↓
Identify affected subsystem
 ↓
Run smallest meaningful test
 ↓
Add regression test if required
 ↓
Run broader tests only when justified
```

Examples:

### Documentation change

No full test suite required.

### Calculator change

Calculator tests.

### Memory change

Memory tests + affected agent tests.

### Database migration

Migration/integration tests.

### Router change

Router + coordinator tests.

### Core architecture change

Relevant subsystem suite.

### Release

Full verification.

This keeps development efficient on limited hardware.

---

# 31. Performance Requirements

SALLY is being developed on constrained hardware.

Therefore:

* Avoid unnecessary background threads
* Avoid unnecessary processes
* Avoid large always-loaded frameworks
* Avoid unnecessary model copies
* Avoid eager imports
* Avoid uncontrolled context growth
* Avoid excessive logging
* Avoid unnecessary network requests
* Avoid large caches
* Avoid duplicated state

The LLM should remain the dominant expensive component rather than the surrounding framework.

---

# 32. Dependency Policy

Every dependency must justify its existence.

Before adding a dependency, consider:

1. Is the functionality actually required?
2. Can the standard library provide it?
3. Is the dependency compatible with Linux?
4. Is it compatible with Termux?
5. Is it practical for Android?
6. Does it increase RAM usage?
7. Does it complicate installation?
8. Does it introduce security or maintenance concerns?

Optional features should preferably use optional dependencies.

---

# 33. Documentation Policy

Documentation must always distinguish:

```text
Implemented
```

from:

```text
Planned
```

from:

```text
Experimental
```

from:

```text
Evaluated
```

No roadmap should describe a future feature as if it already exists.

---

# 34. Git Policy

Development should be incremental.

Preferred sequence:

```text
Implement
 ↓
Focused verification
 ↓
Document
 ↓
Review diff
 ↓
Commit
 ↓
Push
```

Commits should describe the actual change.

Avoid large unrelated commits.

---

# 35. Privacy and Licensing

SALLY remains a private project.

The current private license protects the project from unauthorized:

* Copying
* Redistribution
* Publishing
* Mirroring
* Selling
* Sublicensing
* Public forks
* Competing-product reuse

Third-party components retain their own licenses.

Hermes uses MIT licensing, so any directly incorporated MIT-licensed code must preserve the required copyright and license notices.

Architectural inspiration does not automatically make SALLY a derivative copy.

---

# 36. No Fake Capabilities

This is one of SALLY's most important rules.

Never build UI such as:

```text
Browse Web
Camera
Voice
Terminal
Memory
Scheduler
```

unless the underlying operation actually works.

Never return:

> "Done."

unless the requested operation was actually performed.

Never fabricate:

* Tool results
* System status
* Search results
* File operations
* Messages
* Reminders
* Device actions
* External API results

---

# 37. Model Reliability

The model is not the authority.

The architecture is:

```text
Model
  ↓
Request
  ↓
Capability Layer
  ↓
Actual Operation
  ↓
Verified Result
  ↓
Model Explanation
```

The model may explain a result.

It does not invent the result.

---

# 38. Long-Term Vision

The mature SALLY system should feel like one assistant regardless of interface.

The user should be able to move between:

* Web
* TUI
* Terminal
* Android
* Termux
* Telegram
* WhatsApp

while the underlying SALLY identity, memory, sessions, tools, and capabilities remain coherent.

The interfaces are different.

The assistant is the same.

---

# 39. Final Architecture

The final conceptual architecture is:

```text
                         USER
                           │
             ┌─────────────┼─────────────┐
             │             │             │
            WEB            TUI         ANDROID
             │             │             │
             └─────────────┼─────────────┘
                           │
                      GATEWAYS
                           │
                    SESSION MANAGER
                           │
                   CONVERSATION LOOP
                           │
                         ROUTER
             ┌─────────────┼─────────────┐
             │             │             │
            CHAT          AGENT         TOOL
             │             │             │
             └─────────────┼─────────────┘
                           │
                    CAPABILITY LAYER
                           │
           ┌───────────────┼────────────────┐
           │               │                │
         TOOLS           SKILLS        INTEGRATIONS
           │               │                │
           └───────────────┼────────────────┘
                           │
                     CONTEXT MANAGER
                           │
              ┌────────────┴────────────┐
              │                         │
           MEMORY                    SESSIONS
              │                         │
              └────────────┬────────────┘
                           │
                      DATABASE LAYER
                       │           │
                    SQLite     PostgreSQL
                       │
                       ▼
                    STORAGE

                           │
                       LLM RUNTIME
                           │
                  Local / Optional Remote
                           │
                     NATIVE LAYER
                           │
                       Rust/PyO3
```

---

# 40. Final Definition of SALLY

SALLY is not simply:

> an AI chatbot.

SALLY is intended to become:

> **a lightweight, private, modular personal AI operating layer that can communicate naturally, remember useful information, execute controlled capabilities, interact with devices and services, manage tasks, and operate across Linux, Termux, Android, web, terminal, and messaging environments.**

The system should achieve that without losing the original principles:

**small, fast, private, stable, understandable, and useful.**

---

# 41. Immediate Development Order

The next engineering sequence is intentionally:

```text
M3  Agent Architecture
 ↓
M4  Database Architecture
 ↓
M5  Persistent Sessions
 ↓
M6  Memory
 ↓
M7  Context Management
 ↓
M8  Full TUI
 ↓
M9  Web Refinement
 ↓
M10 Tools
 ↓
M11 Skills
 ↓
M12 Scheduler
 ↓
M13 Security
 ↓
M14 Gateways
 ↓
M15 Android / Termux
 ↓
M16 Voice
 ↓
M17 Vision
 ↓
M18 Media / System Control
 ↓
M19 MCP / Integrations
 ↓
M20 Advanced Agents
 ↓
M21 Projects / Workspaces
 ↓
M22 Observability
 ↓
M23 Deployment
 ↓
M24 Quality / Release
```

Every milestone should leave the project runnable.

Every milestone should be documented.

Every major architectural change should be verified.

And no feature should be marked complete until it actually works.
