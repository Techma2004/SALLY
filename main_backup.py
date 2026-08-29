from core.brain import think, get_llm
from core.memory import learn, recall, get_context
from core.skills import load_skills
import json
from pathlib import Path

HIST = Path("memory/history.json")
history = json.loads(HIST.read_text()) if HIST.exists() else []
tools = load_skills()

print("--- SALLY v1.1 BLANK + SKILLS (Hermes-inspired) ---")
print(f"[MEMORY] {get_context()}")
print(f"[TOOLS] {list(tools.keys())}")
get_llm()

while True:
    u = input("\nYou: ").strip()
    if not u: continue
    if u.lower() in ["exit","quit"]: HIST.write_text(json.dumps(history[-20:], indent=2)); break
    if u.lower().startswith("recall "):
        for h in recall(u[7:], k=5): print(f"- {h['id']}: {h['content'][:200]}")
        continue
    learn(u)
    ans = think(u, history, tools)
    print(f"\nSALLY: {ans}")
    history.append({"role":"user","content":u}); history.append({"role":"assistant","content":ans})
    HIST.write_text(json.dumps(history[-20:], indent=2))
