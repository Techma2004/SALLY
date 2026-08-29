from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from core.gateway.gateway import gateway_chat
from core.memory import search_memory
import asyncio, psutil
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="SALLY v0.50 Streaming")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
executor = ThreadPoolExecutor(4)
ROOT = Path(__file__).parent.parent.parent
WEB_DIST = ROOT / "web" / "dist"

@app.post("/chat")
async def chat(req: Request):
    b = await req.json()
    loop = asyncio.get_event_loop()
    ans = await loop.run_in_executor(executor, lambda: gateway_chat(b.get("user_id","web"), b.get("message",""), "web"))
    return {"answer": ans}

@app.post("/chat/stream")
async def chat_stream(req: Request):
    b = await req.json()
    msg = b.get("message","")
    async def gen():
        loop = asyncio.get_event_loop()
        full = await loop.run_in_executor(executor, lambda: gateway_chat("web", msg, "web"))
        for chunk in full.split(" "):
            yield chunk + " "
            await asyncio.sleep(0.03)
    return StreamingResponse(gen(), media_type="text/plain")

@app.get("/system/stats")
async def sys_stats():
    vm = psutil.virtual_memory()
    disk = psutil.disk_usage(str(ROOT))
    db_path = ROOT / "memory" / "memory.db"
    db_size = db_path.stat().st_size / (1024**3) if db_path.exists() else 0
    return {
        "ram_percent": vm.percent,
        "ram_used_gb": round(vm.used / (1024**3), 1),
        "ram_total_gb": round(vm.total / (1024**3), 1),
        "cpu_percent": psutil.cpu_percent(interval=0.5),
        "disk_percent": disk.percent,
        "disk_used_gb": round(disk.used / (1024**3), 1),
        "disk_total_gb": round(disk.total / (1024**3), 1),
        "memory_used_gb": round(db_size, 2),
        "memory_percent": min(100, int(vm.percent * 0.68 + db_size*10)) or 68
    }

@app.get("/memory")
async def mem(q="Edima"):
    loop = asyncio.get_event_loop()
    res = await loop.run_in_executor(executor, lambda: search_memory(q,10))
    return {"results": res}

@app.get("/health")
async def health():
    return {"status":"ok","version":"v0.50","dist": WEB_DIST.exists()}

# === SERVE REACT DIST WITH ASSETS MOUNT ===
if WEB_DIST.exists():
    assets_dir = WEB_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")
    
    @app.get("/", response_class=HTMLResponse)
    async def home():
        return FileResponse(WEB_DIST / "index.html")
    
    @app.get("/{path:path}", response_class=HTMLResponse)
    async def spa(path: str):
        file_path = WEB_DIST / path
        if file_path.exists() and file_path.is_file():
            return FileResponse(file_path)
        return FileResponse(WEB_DIST / "index.html")
else:
    @app.get("/", response_class=HTMLResponse)
    async def home():
        return "<h1>SALLY v0.50 - Run: cd web && npm run build</h1>"
