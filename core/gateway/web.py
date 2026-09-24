from __future__ import annotations

import os
import time
from pathlib import Path

import psutil
from fastapi import BackgroundTasks, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.config import PROJECT_ROOT, settings
from core.gateway.gateway import Gateway
from core.gateway.whatsapp import WhatsAppGateway


ROOT = PROJECT_ROOT
WEB_DIST = ROOT / "web" / "dist"

gateway = Gateway()
whatsapp_gateway = WhatsAppGateway(gateway)
WEB_PORT = 5678

app = FastAPI(
    title="SALLY Gateway",
    version=settings.sally.version,
    description="Unified HTTP gateway for SALLY.",
)

cors_raw = os.getenv(
    "SALLY_CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173",
)
cors_origins = [origin.strip() for origin in cors_raw.split(",") if origin.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1)
    user_id: str = "web"
    source: str = "web"
    conversation_id: str | None = None


class ChatResponse(BaseModel):
    answer: str
    status: str
    task_id: str
    agent_name: str
    source: str
    conversation_id: str
    elapsed_ms: float
    route: str
    error: str | None = None


def _user_id(value: str) -> str:
    value = value.strip()
    return value[:160] or "anonymous"


def _title(message: str) -> str:
    clean = " ".join(message.split())
    return clean[:80] + ("…" if len(clean) > 80 else "")


def _conversation_dict(conversation) -> dict[str, object]:
    return {
        "id": conversation.id,
        "title": conversation.title,
        "created_at": conversation.created_at.isoformat(),
        "updated_at": conversation.updated_at.isoformat(),
    }


def _message_dict(message) -> dict[str, object]:
    return {
        "id": message.id,
        "role": message.role,
        "content": message.content,
        "created_at": message.created_at.isoformat(),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    user_id = _user_id(request.user_id)
    memory = gateway.coordinator.memory

    conversation = None
    if request.conversation_id:
        conversation = memory.get_conversation(
            request.conversation_id,
            user_id,
        )
        if conversation is None:
            raise HTTPException(status_code=404, detail="Conversation not found.")

    if conversation is None:
        conversation = memory.create_conversation(
            user_id,
            title=_title(request.message),
        )

    memory.add_conversation_message(
        conversation.id,
        role="user",
        content=request.message,
    )

    started = time.perf_counter()
    route = gateway.coordinator.describe_route(request.message)

    try:
        result = gateway.handle(
            request.message,
            user_id=user_id,
            source=request.source,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"SALLY gateway error: {exc}",
        ) from exc

    if result.answer:
        memory.add_conversation_message(
            conversation.id,
            role="assistant",
            content=result.answer,
        )

    return ChatResponse(
        answer=result.answer,
        status=result.status.value,
        task_id=result.task_id,
        agent_name=result.agent_name,
        source=result.source,
        conversation_id=conversation.id,
        elapsed_ms=round((time.perf_counter() - started) * 1000, 2),
        route=route,
        error=result.error,
    )


@app.post("/conversations")
def create_conversation(
    user_id: str = "web",
    title: str = "New Conversation",
) -> dict[str, object]:
    conversation = gateway.coordinator.memory.create_conversation(
        _user_id(user_id),
        title=title,
    )
    return _conversation_dict(conversation)


@app.get("/conversations")
def list_conversations(
    user_id: str = "web",
    limit: int = 20,
) -> dict[str, object]:
    conversations = gateway.coordinator.memory.recent_conversations(
        _user_id(user_id),
        limit=max(1, min(limit, 100)),
    )
    return {
        "conversations": [
            _conversation_dict(conversation)
            for conversation in conversations
        ]
    }


@app.get("/conversations/{conversation_id}")
def get_conversation(
    conversation_id: str,
    user_id: str = "web",
) -> dict[str, object]:
    memory = gateway.coordinator.memory
    conversation = memory.get_conversation(
        conversation_id,
        _user_id(user_id),
    )
    if conversation is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return {
        "conversation": _conversation_dict(conversation),
        "messages": [
            _message_dict(message)
            for message in memory.conversation_messages(conversation.id)
        ],
    }


@app.get("/tools")
def tools() -> dict[str, object]:
    registry = gateway.coordinator.runtime.tools
    return {
        "tools": [
            {
                "name": name,
                "description": registry.metadata(name).description,
                "safety": registry.metadata(name).safety,
                "requires_inference": registry.requires_inference(name),
            }
            for name in registry.names()
        ]
    }


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "name": settings.sally.name,
        "version": settings.sally.version,
        "web_dist": WEB_DIST.exists(),
    }


@app.get("/system/stats")
def system_stats() -> dict[str, object]:
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ROOT))
    db_path = ROOT / "memory" / "memory.db"
    db_size_mb = db_path.stat().st_size / (1024 * 1024) if db_path.exists() else 0.0

    return {
        "ram_percent": vm.percent,
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "memory_db_mb": round(db_size_mb, 2),
        "model": settings.llm.model_path,
        "context_tokens": settings.llm.n_ctx,
        "web_port": WEB_PORT,
    }


@app.get("/memory")
def memory_search(
    q: str,
    limit: int = 5,
) -> dict[str, object]:
    query = q.strip()
    if not query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    memories = gateway.coordinator.memory.search(
        query,
        limit=max(1, min(limit, 50)),
    )

    return {
        "query": query,
        "results": [
            {
                "id": memory.id,
                "content": memory.content,
                "type": memory.memory_type.value,
                "importance": memory.importance,
                "created_at": memory.created_at.isoformat(),
            }
            for memory in memories
        ],
    }


@app.get("/memory/recent")
def recent_memory(limit: int = 20) -> dict[str, object]:
    memories = gateway.coordinator.memory.recent(
        limit=max(1, min(limit, 100)),
    )
    return {
        "results": [
            {
                "id": memory.id,
                "content": memory.content,
                "type": memory.memory_type.value,
                "importance": memory.importance,
                "created_at": memory.created_at.isoformat(),
            }
            for memory in memories
        ]
    }


@app.get("/webhooks/whatsapp")
def whatsapp_webhook_verify(
    mode: str | None = Query(default=None, alias="hub.mode"),
    verify_token: str | None = Query(default=None, alias="hub.verify_token"),
    challenge: str | None = Query(default=None, alias="hub.challenge"),
) -> str:
    try:
        return whatsapp_gateway.verify_webhook(mode, verify_token, challenge)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=403, detail=str(exc)) from exc


@app.post("/webhooks/whatsapp")
async def whatsapp_webhook(
    request: Request,
    background_tasks: BackgroundTasks,
) -> dict[str, object]:
    body = await request.body()
    if not whatsapp_gateway.verify_signature(
        body,
        request.headers.get("X-Hub-Signature-256"),
    ):
        raise HTTPException(status_code=403, detail="Invalid WhatsApp webhook signature.")

    payload = await request.json()
    messages = whatsapp_gateway.extract_messages(payload)

    for message in messages:
        background_tasks.add_task(whatsapp_gateway.process_message, message)

    return {"status": "ok", "messages_queued": len(messages)}


if WEB_DIST.exists():
    assets_dir = WEB_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/", response_class=HTMLResponse)
    def home() -> FileResponse:
        return FileResponse(WEB_DIST / "index.html")

    @app.get("/{path:path}", response_class=HTMLResponse)
    def spa(path: str) -> FileResponse:
        file_path = WEB_DIST / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(WEB_DIST / "index.html")
else:
    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (
            "<h1>SALLY Gateway</h1>"
            "<p>Web build not found. Run "
            "<code>cd web && npm run build</code>.</p>"
        )
