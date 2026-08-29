import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
HIST_FILE = PROJECT_ROOT / "memory" / "history.json"
from core.brain import chat
from core.memory import search_memory, save_memory
from core.config import load_config
try:
    from core.learner import get_tool_stats, list_auto_skills
    HAS_LEARNER=True
except:
    HAS_LEARNER=False

def handle_gateway_command(text, history):
    low=text.strip().lower()
    if low in ["/new","/clear","/start"]:
        history.clear()
        if HIST_FILE.exists(): HIST_FILE.write_text("[]")
        return "New session - I'm SALLY", True
    if low.startswith("/memory") or low.startswith("recall "):
        q=text.split(" ",1)[1] if " " in text else ""
        res=search_memory(q,5)
        if not res: return f"No memory for '{q}'", False
        return "\n".join([f"- {r['content'][:180]}" for r in res]), False
    if low in ["/skills","/help"]:
        from core.tools import TOOLS
        out="Skills:\n" + "\n".join([f"• {t['function']['name']}" for t in TOOLS])
        return out, False
    if low=="/learn":
        if not HAS_LEARNER: return "Learner not ready", False
        stats=get_tool_stats()
        return "\n".join([f"{t} x{c}" for t,c,_ in stats]) if stats else "No usage yet", False
    return None, False

def gateway_chat(user_id, text, platform="telegram"):
    hist_path=PROJECT_ROOT / f"memory/history_{platform}.json"
    if platform=="tui": hist_path=HIST_FILE
    history=[]
    if hist_path.exists():
        try: history=json.loads(hist_path.read_text())[-20:]
        except: history=[]
    if text.startswith("/"):
        result,cleared=handle_gateway_command(text, history)
        if result:
            if cleared: hist_path.write_text("[]")
            return result
    try: save_memory(f"[{platform}:{user_id}] {text[:100]}", mtype="daily", importance=0.3)
    except: pass
    answer=chat(text, history)
    history.append({"role":"user","content":text}); history.append({"role":"assistant","content":answer})
    try:
        hist_path.write_text(json.dumps(history[-20:], indent=2))
        HIST_FILE.write_text(json.dumps(history[-20:], indent=2))
    except: pass
    return answer
