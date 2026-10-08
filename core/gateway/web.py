from __future__ import annotations

import json
import os
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import psutil
from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from core.config import PROJECT_ROOT, settings
from core.gateway.gateway import Gateway
from core.gateway.security import enforce_rate_limit, require_local_or_api_key
from core.gateway.whatsapp import WhatsAppGateway

_PROTECTED = [Depends(require_local_or_api_key), Depends(enforce_rate_limit)]


ROOT = PROJECT_ROOT
WEB_DIST = ROOT / "web" / "dist"

gateway = Gateway()
whatsapp_gateway = WhatsAppGateway(gateway)
WEB_PORT = settings.server.port

def _preload_model() -> None:
    try:
        from core.llm import get_llm

        get_llm()
    except Exception as exc:  # model missing or failed to load
        print(f"SALLY: model preload failed: {exc}")


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # Opt-in (LLM_PRELOAD=true): load the model in the background at startup
    # so the first message does not pay the model-load delay.
    if os.getenv("LLM_PRELOAD", "false").strip().lower() in {"1", "true", "yes"}:
        threading.Thread(target=_preload_model, daemon=True).start()

    yield


app = FastAPI(
    lifespan=lifespan,
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
    trace: dict | None = None


class SettingsUpdate(BaseModel):
    values: dict[str, object]


def _route_label(trace: dict | None, fallback: str) -> str:
    """The route that actually ran (the rules may have been overridden)."""
    if trace and trace.get("route"):
        route = trace["route"]
        return f"{route['type']} \u2192 {route['target']}: {route['reason']}"

    return fallback


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


def _sse(payload: dict[str, object]) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def _open_conversation(request: ChatRequest, user_id: str):
    """
    Find or create the conversation, load its recent history, and store the
    new user message. Returns (conversation, history).
    """
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

    history = [
        {"role": item.role, "content": item.content}
        for item in memory.conversation_messages(conversation.id, limit=8)
    ]

    memory.add_conversation_message(
        conversation.id,
        role="user",
        content=request.message,
    )

    return conversation, history


@app.post("/chat", response_model=ChatResponse, dependencies=_PROTECTED)
def chat(request: ChatRequest) -> ChatResponse:
    user_id = _user_id(request.user_id)
    memory = gateway.coordinator.memory

    conversation, history = _open_conversation(request, user_id)

    started = time.perf_counter()
    route = gateway.coordinator.describe_route(request.message)

    try:
        result = gateway.handle(
            request.message,
            user_id=user_id,
            source=request.source,
            history=history,
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
        route=_route_label(result.trace, route),
        error=result.error,
        trace=result.trace,
    )


@app.post("/chat/stream", dependencies=_PROTECTED)
async def chat_stream(
    payload: ChatRequest,
    request: Request,
) -> StreamingResponse:
    """Server-sent events: start, token*, then done (or error)."""
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    user_id = _user_id(payload.user_id)
    memory = gateway.coordinator.memory

    conversation, history = await run_in_threadpool(
        _open_conversation,
        payload,
        user_id,
    )

    route = gateway.coordinator.describe_route(payload.message)
    started = time.perf_counter()
    cancel = threading.Event()

    async def events():
        yield _sse(
            {
                "type": "start",
                "conversation_id": conversation.id,
                "route": route,
            }
        )

        iterator = gateway.stream(
            payload.message,
            user_id=user_id,
            source=payload.source,
            history=history,
            cancel=cancel,
        )
        finished = object()

        try:
            while True:
                event = await run_in_threadpool(next, iterator, finished)

                if event is finished:
                    break

                if await request.is_disconnected():
                    break

                if event["type"] == "token":
                    yield _sse(event)
                    continue

                response = event["response"]

                if response.answer:
                    await run_in_threadpool(
                        memory.add_conversation_message,
                        conversation.id,
                        role="assistant",
                        content=response.answer,
                    )

                yield _sse(
                    {
                        "type": "done",
                        "answer": response.answer,
                        "status": response.status.value,
                        "task_id": response.task_id,
                        "agent_name": response.agent_name,
                        "conversation_id": conversation.id,
                        "elapsed_ms": round(
                            (time.perf_counter() - started) * 1000,
                            2,
                        ),
                        "route": _route_label(response.trace, route),
                        "error": response.error,
                        "trace": response.trace,
                    }
                )
        except Exception as exc:
            yield _sse(
                {
                    "type": "error",
                    "detail": f"SALLY gateway error: {exc}",
                }
            )
        finally:
            # Tells the generation thread to stop at the next token, which
            # also releases the model lock for the next request.
            cancel.set()

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.post("/conversations", dependencies=_PROTECTED)
def create_conversation(
    user_id: str = "web",
    title: str = "New Conversation",
) -> dict[str, object]:
    conversation = gateway.coordinator.memory.create_conversation(
        _user_id(user_id),
        title=title,
    )
    return _conversation_dict(conversation)


@app.get("/conversations", dependencies=_PROTECTED)
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


@app.get("/conversations/{conversation_id}", dependencies=_PROTECTED)
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


@app.get("/help", dependencies=_PROTECTED)
def help_page() -> dict[str, object]:
    from core.help import help_topics

    return help_topics(gateway.coordinator.runtime.tools)


@app.get("/settings", dependencies=_PROTECTED)
def get_settings() -> dict[str, object]:
    from core import settings_schema
    from core.config import settings

    return settings_schema.describe(settings)


@app.put("/settings", dependencies=_PROTECTED)
def update_settings(payload: SettingsUpdate) -> dict[str, object]:
    from core import settings_schema
    from core.config import settings

    try:
        clean = settings_schema.validate(payload.values)
        settings_schema.write_env(clean)
    except settings_schema.SettingsError as exc:
        raise HTTPException(
            status_code=422,
            detail={"message": "Some settings were not saved.", "errors": exc.errors},
        ) from exc

    return {"saved": sorted(clean), **settings_schema.describe(settings)}


@app.post("/settings/restart", dependencies=_PROTECTED)
def restart_sally() -> dict[str, str]:
    from core import settings_schema

    settings_schema.restart_process()

    return {"status": "restarting"}


@app.delete("/conversations/{conversation_id}", dependencies=_PROTECTED)
def delete_conversation(
    conversation_id: str,
    user_id: str = "web",
) -> dict[str, str]:
    deleted = gateway.coordinator.memory.delete_conversation(
        conversation_id,
        _user_id(user_id),
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    return {"status": "deleted"}


@app.get("/tools", dependencies=_PROTECTED)
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


def _model_path_exists() -> bool:
    model_path = Path(settings.llm.model_path)
    if not model_path.is_absolute():
        model_path = ROOT / model_path
    return model_path.exists()


@app.get("/health")
def health() -> dict[str, object]:
    return {
        "status": "ok",
        "name": settings.sally.name,
        "version": settings.sally.version,
        "web_dist": WEB_DIST.exists(),
        "model_exists": _model_path_exists(),
        "web_port": WEB_PORT,
    }


@app.get("/system/stats", dependencies=_PROTECTED)
def system_stats() -> dict[str, object]:
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ROOT))
    db_path = Path(settings.memory.db_path)
    if not db_path.is_absolute():
        db_path = ROOT / db_path

    db_size_mb = (
        db_path.stat().st_size / (1024 * 1024)
        if db_path.exists()
        else 0.0
    )

    return {
        "ram_percent": vm.percent,
        "ram_used_gb": round(vm.used / (1024**3), 2),
        "ram_total_gb": round(vm.total / (1024**3), 2),
        "cpu_percent": psutil.cpu_percent(interval=None),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / (1024**3), 2),
        "disk_total_gb": round(disk.total / (1024**3), 2),
        "memory_db_mb": round(db_size_mb, 2),
        "memory_db_path": str(db_path),
        "model": settings.llm.model_path,
        "model_exists": _model_path_exists(),
        "context_tokens": settings.llm.n_ctx,
        "web_port": WEB_PORT,
    }


@app.get("/memory", dependencies=_PROTECTED)
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


@app.get("/memory/recent", dependencies=_PROTECTED)
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


@app.delete("/memory/{memory_id}", dependencies=_PROTECTED)
def forget_memory(memory_id: str) -> dict[str, str]:
    if not gateway.coordinator.memory.forget(memory_id):
        raise HTTPException(status_code=404, detail="Memory not found.")

    return {"status": "deleted"}


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

    _WEB_DIST_RESOLVED = WEB_DIST.resolve()

    @app.get("/{path:path}", response_class=HTMLResponse)
    def spa(path: str) -> FileResponse:
        candidate = (WEB_DIST / path).resolve()

        # Reject anything that escapes the web/dist directory (path
        # traversal via "..", absolute paths, symlink tricks, etc.)
        # instead of trusting the client-supplied path.
        is_contained = (
            candidate == _WEB_DIST_RESOLVED
            or _WEB_DIST_RESOLVED in candidate.parents
        )

        if is_contained and candidate.exists() and candidate.is_file():
            return FileResponse(candidate)

        return FileResponse(WEB_DIST / "index.html")
else:
    @app.get("/", response_class=HTMLResponse)
    def home() -> str:
        return (
            "<h1>SALLY Gateway</h1>"
            "<p>Web build not found. Run "
            "<code>cd web && npm run build</code>.</p>"
        )
