import json, os
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
from .brain import chat, get_llm
from .memory import search_memory, load_user, init_db
from .config import PROJECT_ROOT
PROJECT_ROOT=Path(__file__).parent.parent
HIST_FILE=PROJECT_ROOT/"memory"/"history.json"
PT_HISTORY=PROJECT_ROOT/"memory"/".prompt_history"
COMMANDS={"/new":"new session","/skills":"list skills","/memory":"search memory","/model":"model info","/usage":"usage","/learn":"learning stats","/help":"help","/exit":"exit","/quit":"exit"}

def handle_command(cmd, history):
    low=cmd.strip().lower()
    if low in ["/new","/clear"]:
        history.clear()
        if HIST_FILE.exists(): HIST_FILE.write_text("[]")
        return "[TUI] New session", True
    if low.startswith("/memory") or low.startswith("recall "):
        q=cmd.split(" ",1)[1] if " " in cmd else ""
        res=search_memory(q,5)
        if not res: return f"No memory for '{q}'", False
        return "\n".join([f"- {r[:180]}" for r in res]), False
    if low=="/skills":
        from .tools import TOOLS, get_tool_descriptions_for_prompt
        out="Available tools:\n" + get_tool_descriptions_for_prompt()
        try:
            from .learner import list_auto_skills
            autos=list_auto_skills()
            if autos: out+="\n\nAuto skills:\n" + "\n".join([f" - {p.name}" for p in autos])
        except: pass
        return out, False
    if low=="/model":
        mp=os.getenv("LLM_MODEL_PATH","models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf")
        return f"Model: {mp} Exists: {Path(mp).exists()}", False
    if low=="/usage":
        db=PROJECT_ROOT/"memory"/"memory.db"
        sz=db.stat().st_size/1024 if db.exists() else 0
        return f"DB {sz:.1f}KB History {len(history)}", False
    if low=="/learn":
        try:
            from .learner import get_tool_stats, list_auto_skills
            stats=get_tool_stats()
            out="Tool usage:\n" + "\n".join([f"{t} x{c}" for t,c,_ in stats]) if stats else "No usage yet"
            autos=list_auto_skills()
            out+=f"\n\nAuto skills {len(autos)}:\n" + "\n".join([p.name for p in autos])
            return out, False
        except Exception as e: return f"Learner error {e}", False
    if low in ["/help","/h"]:
        return "\n".join([f"{k} - {v}" for k,v in COMMANDS.items()]), False
    if low in ["/exit","/quit","/bye"]: return "EXIT", True
    return None, False

def run_tui():
    init_db()
    history=json.loads(HIST_FILE.read_text()) if HIST_FILE.exists() else []
    if HAS_PT:
        session=PromptSession(history=FileHistory(str(PT_HISTORY)), completer=WordCompleter(list(COMMANDS.keys()), ignore_case=True), auto_suggest=AutoSuggestFromHistory())
    else: session=None
    print("=== SALLY v0.44 Hermes TUI + Learner === /help /learn")
    try: get_llm()
    except Exception as e: print(f"LLM warn {e}")
    while True:
        try: u=session.prompt("\nYou: ") if HAS_PT else input("\nYou: ")
        except: break
        u=u.strip()
        if not u: continue
        if u.startswith("/") or u.lower().startswith("recall "):
            res,clr=handle_command(u,history)
            if res=="EXIT": break
            if res: print(f"\n{res}\n")
            if u.startswith("/memory") or u.lower().startswith("recall "): continue
            if res and u.startswith("/"): continue
        try:
            from .memory import learn
            learn(u)
        except: pass
        ans=chat("user", u)
        print(f"\nSALLY: {ans}\n")
        history.append({"role":"user","content":u}); history.append({"role":"assistant","content":ans})
        HIST_FILE.write_text(json.dumps(history[-20:], indent=2))
