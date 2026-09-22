try:
    from core.tui import run_tui
    HAS_TUI=True
except Exception as e:
    HAS_TUI=False
    print(f"[MAIN] TUI fallback: {e}")

if HAS_TUI:
    if __name__ == "__main__":
        run_tui()
else:
    from core.brain import chat
    from core.memory import search_memory
    import json
    from pathlib import Path
    HIST = Path("memory/history.json")
    history = json.loads(HIST.read_text()) if HIST.exists() else []
    print("--- SALLY v0.42 fallback ---")
    while True:
        try:
            u=input("\nYou: ").strip()
        except: break
        if not u: continue
        if u.lower() in ["exit","quit","bye"]:
            HIST.write_text(json.dumps(history[-20:], indent=2)); break
        if u.lower().startswith("recall ") or u.lower().startswith("/memory"):
            q=u.split(" ",1)[1] if " " in u else ""
            for r in search_memory(q,5): print(f"- {r[:200]}")
            continue
        ans=chat("user", u)
        print(f"\nSALLY: {ans}")
        history.append({"role":"user","content":u}); history.append({"role":"assistant","content":ans})
        HIST.write_text(json.dumps(history[-20:], indent=2))
