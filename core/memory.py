# Compatibility wrapper — keeps main.py working, but uses Obsidian backend
from core.obsidian_memory import remember, recall, get_fact, list_facts, init_core_from_facts
import json
from core.config import PROJECT_ROOT

MEMORY_FILE = PROJECT_ROOT / "memory.json"

def load_memory():
    if MEMORY_FILE.exists():
        try: return json.loads(MEMORY_FILE.read_text())
        except: return []
    return []

def save_memory(h):
    MEMORY_FILE.write_text(json.dumps(h[-20:], indent=2))
