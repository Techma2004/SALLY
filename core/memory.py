"""
SALLY Memory v0.40 — Offline-first
- New: remember(), recall(), init_core_from_facts()
- Legacy: load_memory(), save_memory() for v0.35 compatibility
"""
import sqlite3
import json
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory"
CORE_DIR = MEMORY_DIR / "core"
EPISODES_DIR = MEMORY_DIR / "episodes"
DB_PATH = MEMORY_DIR / "sally.db"
LEGACY_JSON = PROJECT_ROOT / "memory.json"

MEMORY_DIR.mkdir(exist_ok=True)
CORE_DIR.mkdir(exist_ok=True)
EPISODES_DIR.mkdir(exist_ok=True)

# --- NEW SYSTEM (SQLite FTS5) ---
def _init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            type TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT
        )
    """)
    conn.execute("CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, content='memories', content_rowid='id')")
    conn.execute("CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN INSERT INTO memories_fts(rowid, content) VALUES (new.id, new.content); END;")
    conn.execute("CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.id, old.content); END;")
    conn.commit()
    return conn

def remember(content: str, type: str = "episodic", source: str = "chat"):
    if not content.strip(): return None
    conn = _init_db()
    ts = datetime.now().isoformat()
    cur = conn.cursor()
    cur.execute("INSERT INTO memories (timestamp, type, content, source) VALUES (?,?,?,?)", (ts, type, content, source))
    conn.commit()
    mem_id = cur.lastrowid
    # Markdown backup
    safe_ts = ts.replace(":", "-")
    if type == "semantic":
        human_md = CORE_DIR / "human.md"
        existing = human_md.read_text() if human_md.exists() else "# Techma Core Memory\n"
        human_md.write_text(existing + f"\n- {content}\n")
    else:
        (EPISODES_DIR / f"{safe_ts}_{type}.md").write_text(f"---\ntype: {type}\ntimestamp: {ts}\n---\n\n{content}\n")
    print(f"[MEMORY] +{type} #{mem_id}")
    return mem_id

def recall(query: str, k: int = 3, type_filter: str = None):
    conn = _init_db()
    sql = "SELECT m.id, m.timestamp, m.type, m.content FROM memories_fts f JOIN memories m ON m.id = f.rowid WHERE memories_fts MATCH?"
    params = [query]
    if type_filter:
        sql += " AND m.type =?"
        params.append(type_filter)
    sql += " LIMIT?"
    params.append(k)
    cur = conn.execute(sql, params)
    return [{"id": r[0], "timestamp": r[1], "type": r[2], "content": r[3]} for r in cur.fetchall()]

def init_core_from_facts():
    try:
        from core.config import USER_FACTS
        human_md = CORE_DIR / "human.md"
        if human_md.exists() and len(human_md.read_text()) > 200:
            return
        print("[MEMORY] Migrating USER_FACTS to human.md...")
        content = f"# Techma — Core Memory (migrated {datetime.now().isoformat()})\n"
        for k, v in USER_FACTS.items():
            line = f"{k}: {v}"
            remember(line, type="semantic", source="config_migration")
            content += f"- **{k}**: {v}\n"
        human_md.write_text(content)
        print(f"[MEMORY] Migrated {len(USER_FACTS)} facts")
    except Exception as e:
        print(f"[MEMORY] Migration failed: {e}")

# --- LEGACY SYSTEM (for your v0.35 main.py) ---
def load_memory():
    """Old JSON history for brain.py"""
    if LEGACY_JSON.exists():
        try:
            data = json.loads(LEGACY_JSON.read_text())
            return data if isinstance(data, list) else []
        except:
            return []
    return []

def save_memory(role, content):
    """Old JSON + New system"""
    # 1. Save to legacy JSON (for brain history)
    history = load_memory()
    history.append({"role": role, "content": content, "timestamp": datetime.now().isoformat()})
    # Keep last 100 only
    history = history[-100:]
    LEGACY_JSON.write_text(json.dumps(history, indent=2))

    # 2. Also save to new SQLite (for recall)
    if role == "user":
        remember(content, type="episodic", source="legacy_save")
    # semantic facts auto-detected
    if any(k in content.lower() for k in ["my name is", "i am", "i live", "i use", "debian", "techma"]):
        remember(content, type="semantic", source="auto_detect")

if __name__ == "__main__":
    init_core_from_facts()
    print(recall("Techma"))
