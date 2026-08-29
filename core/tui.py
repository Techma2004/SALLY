
import json
from pathlib import Path
try:
    from prompt_toolkit import PromptSession
    from prompt_toolkit.history import FileHistory
    from prompt_toolkit.completion import WordCompleter
    from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
    HAS_PT=True
except: HAS_PT=False
try:
    from rich.console import Console
    from rich.panel import Panel
    HAS_RICH=True
    console=Console()
except: HAS_RICH=False
from.brain import chat, get_llm
from.memory import search_memory, load_user, init_db
from.config import load_config

PROJECT_ROOT=Path(__file__).parent.parent
HIST_FILE=PROJECT_ROOT/"memory"/"history.json"
COMMANDS={"/new":"new session","/skills":"list skills","/memory":"search","/model":"model info","/usage":"usage","/help":"help","/exit":"exit","/quit":"exit"}

def handle_command(cmd,history):
    low=cmd.lower().strip()
    if low in ["/new","/clear"]:
        history.clear()
        if HIST_FILE.exists(): HIST_FILE.write_text("[]")
        return "New session", True
    if low.startswith("/memory") or low.startswith("recall "):
        q=cmd.split(" ",1)[1] if " " in cmd else ""
        res=search_memory(q,5)
        return "\n".join([f"- {r['content'][:200]}" for r in res]) or f"No memory for {q}", False
    if low=="/skills":
        from.tools import TOOLS
        return "\n".join([f"{t['function']['name']}: {t['function']['description']}" for t in TOOLS]), False
    if low=="/model":
        cfg=load_config()
        return f"Model: {cfg.get('model_path')} Exists: {Path(cfg.get('model_path')).exists()}", False
    if low=="/usage":
        db=PROJECT_ROOT/"memory"/"memory.db"
        sz=db.stat().st_size/1024 if db.exists() else 0
        return f"DB {sz:.1f}KB History {len(history)}", False
    if low in ["/help","/h"]:
        return "\n".join([f"{k} - {v}" for k,v in COMMANDS.items()]), False
    if low in ["/exit","/quit","/bye"]: return "EXIT", True
    return None, False

def run_tui():
    init_db()
    history=json.loads(HIST_FILE.read_text()) if HIST_FILE.exists() else []
    if HAS_PT:
        session=PromptSession(history=FileHistory(str(PROJECT_ROOT/"memory"/".prompt_history")), completer=WordCompleter(list(COMMANDS.keys()), ignore_case=True), auto_suggest=AutoSuggestFromHistory())
    else:
        session=None
    print("=== SALLY v0.43 Hermes TUI === /help for commands")
    try: get_llm()
    except Exception as e: print(f"LLM warn {e}")
    while True:
        try:
            u=session.prompt("\nYou: ") if HAS_PT else input("\nYou: ")
        except: break
        u=u.strip()
        if not u: continue
        if u.startswith("/"):
            res,clr=handle_command(u,history)
            if res=="EXIT": break
            if res: print(res)
            if clr and u=="/new": continue
            if u.startswith("/memory"): continue
            if res and u.startswith("/"): continue
        from.memory import learn
        try: learn(u)
        except: pass
        ans=chat(u,history)
        print(f"\nSALLY: {ans}\n")
        history.append({"role":"user","content":u}); history.append({"role":"assistant","content":ans})
        HIST_FILE.write_text(json.dumps(history[-20:], indent=2))
