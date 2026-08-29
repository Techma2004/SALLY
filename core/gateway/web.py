from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from core.gateway.gateway import gateway_chat
from core.memory import search_memory
from core.learner import get_tool_stats

app = FastAPI(title="SALLY v0.49 React")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
ROOT = Path(__file__).parent.parent.parent
WEB_DIST = ROOT / "web" / "dist"

@app.post("/chat")
async def chat(req: Request):
    b = await req.json()
    return {"answer": gateway_chat(b.get("user_id","web"), b.get("message",""), "web")}

@app.get("/memory")
async def mem(q: str="Edima"):
    res = search_memory(q, 10)
    return {"results": [r.get("content","")[:200] if isinstance(r, dict) else str(r)[:200] for r in res]}

@app.get("/stats")
async def stats():
    try:
        s=get_tool_stats()
        return {"stats": [{"tool":t,"count":c} for t,c,_ in s]}
    except: return {"stats":[]}

@app.get("/daily")
async def daily_list():
    d = ROOT / "memory" / "daily"
    files = sorted([p.name for p in d.glob("*.md")], reverse=True)[:10] if d.exists() else []
    return {"files": files}

# Serve React build
if WEB_DIST.exists():
    app.mount("/assets", StaticFiles(directory=WEB_DIST / "assets"), name="assets")
    @app.get("/")
    async def home():
        return FileResponse(WEB_DIST / "index.html")
    @app.get("/{path:path}")
    async def serve_spa(path: str):
        # For SPA routing, serve index.html for unknown paths
        target = WEB_DIST / path
        if target.exists() and target.is_file():
            return FileResponse(target)
        return FileResponse(WEB_DIST / "index.html")
else:
    @app.get("/")
    async def home():
        return {"status":"Build not found. Run: cd web && npm run build"}
