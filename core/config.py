from pathlib import Path
import json, os
PROJECT_ROOT = Path(__file__).parent.parent.resolve()

def load_user():
    try:
        hp = PROJECT_ROOT / "memory" / "core" / "human.json"
        if hp.exists():
            d = json.loads(hp.read_text())
            return {"user_name": d.get("user_name") or d.get("name") or "Edima", "user_handle": d.get("user_handle") or "Edima", "user_location": d.get("user_location") or "Calabar, NG", "raw": d}
    except: pass
    return {"user_name": "Edima", "user_handle": "Edima", "user_location": "Calabar, NG", "raw": {}}

def load_or_create_facts():
    return load_user()
