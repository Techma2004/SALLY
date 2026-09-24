# SALLY Upgrade Plan

SALLY is a lightweight, offline-first personal AI assistant built around a modular agent runtime.

## Current architecture

User
→ Gateway
→ Coordinator
→ Router / Agent Runtime
→ Tools
→ Memory
→ Local LLM
→ Response

## Completed

- SQLite FTS5 persistent memory
- Unified Gateway
- React web workspace
- Persistent web conversations
- Telegram adapter
- WhatsApp adapter
- Deterministic calculator and datetime tools
- Scientific calculation and unit conversion
- Verified scientific constants
- Hidden inference layer
- Dynamic agent actions
- Live system/runtime dashboard
- Fixed local web port: 8080
- TUI removed

## Next engineering priorities

### 1. Conversation continuity
Persist useful conversation context and summaries so reopening a conversation also restores the relevant model context.

### 2. Evidence-grounded inference
Constrain explanation generation to verified tool evidence and prevent unsupported numerical or factual additions.

### 3. Capability enforcement
Make registered tools the source of truth for external actions and prevent unsupported claims such as browsing or filesystem access.

### 4. Context and memory management
Introduce session summaries, retrieval, and dynamic context budgets so long conversations remain usable without pretending that model context is unlimited.

### 5. Tool expansion
Add capabilities only when implemented end-to-end and exposed through the Gateway.

### 6. Model management
Add hardware-aware model information, benchmarking, and clean model selection without increasing the base runtime unnecessarily.

## Design rule

No mock capabilities, fake telemetry, demo data, or UI controls that do not perform a real SALLY operation.
