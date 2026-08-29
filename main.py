
from core.brain import think, get_llm, chat
from core.memory import learn, recall, get_context, search_memory
import json
from pathlib import Path
HIST = Path("memory/history.json")
history = json.loads(HIST.read_text()) if HIST.exists() else []
try:
    from core.skills import load_skills
    tools = load_skills()
except:
    tools = {"weather": True, "news": True, "time": True, "calc": True, "memory": True}
print("--- SALLY v0.42 Hermes 3-Layer ---")
print(f"[MEMORY] {get_context()}")
print(f"[TOOLS] {list(tools.keys()) if isinstance(tools, dict) else tools}")
try:
    get_llm()
except Exception as e:
    print(f"[WARN] LLM not loaded: {e}")
while True:
    try:
        u = input("\nYou: ").strip()
    except:
        break
    if not u: continue
    if u.lower() in ["exit","quit","bye"]:
        HIST.write_text(json.dumps(history[-20:], indent=2))
        break
    if u.lower().startswith("recall "):
        for r in search_memory(u[7:],5):
            print(f"- {r['content'][:200]}")
        continue
    try: learn(u)
    except: pass
    ans = think(u, history, tools)
    print(f"\nSALLY: {ans}")
    history.append({"role":"user","content":u}); history.append({"role":"assistant","content":ans})
    HIST.write_text(json.dumps(history[-20:], indent=2))
