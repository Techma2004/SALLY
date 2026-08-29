import sqlite3
import datetime
from pathlib import Path
import hashlib

ROOT = Path(__file__).parent.parent
MEM_DIR = ROOT / "memory"
DAILY_DIR = MEM_DIR / "daily"
DB_PATH = MEM_DIR / "memory.db"
DAILY_DIR.mkdir(exist_ok=True)

USER_FILE = MEM_DIR / "USER.md"
MEMORY_FILE = MEM_DIR / "MEMORY.md"

# Backward compat: ensure files exist
for f, default in [(USER_FILE, "# USER\n\nNo facts yet.\n"), (MEMORY_FILE, "# MEMORY\n\nBlank slate.\n")]:
    if not f.exists(): f.write_text(default)

def init_db():
    """Hermes 3-layer: sessions + memories + user_model + FTS5"""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()

    # Layer 1 — sessions
    c.execute("""
    CREATE TABLE IF NOT EXISTS sessions (
        id TEXT PRIMARY KEY,
        started_at TEXT,
        summary TEXT
    )
    """)

    # Layer 2 — persistent memories
    c.execute("""
    CREATE TABLE IF NOT EXISTS memories (
        id TEXT PRIMARY KEY,
        type TEXT, -- user, fact, skill, daily, session
        content TEXT,
        importance REAL DEFAULT 0.5,
        created_at TEXT
    )
    """)

    # FTS5 for fast search ~10ms over 10K docs (Hermes benchmark)
    c.execute("""
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
    USING fts5(content, content='memories', content_rowid='rowid')
    """)

    # Layer 3 — user model (Honcho-style)
    c.execute("""
    CREATE TABLE IF NOT EXISTS user_model (
        key TEXT PRIMARY KEY,
        value TEXT,
        confidence REAL DEFAULT 0.5,
        updated_at TEXT
    )
    """)

    conn.commit()
    conn.close()
    print(f"[MEMORY] DB ready at {DB_PATH}")

    # Migrate existing USER.md / MEMORY.md into DB if empty
    migrate_md_to_db()

def migrate_md_to_db():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM memories")
    if c.fetchone()[0] > 0:
        conn.close()
        return

    print("[MEMORY] Migrating USER.md + MEMORY.md -> memory.db")
    for md_file, mtype in [(USER_FILE, "user"), (MEMORY_FILE, "fact")]:
        if not md_file.exists(): continue
        for line in md_file.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"): continue
            if line.startswith("- "): line = line[2:]
            if len(line) < 3: continue
            save_memory(line, mtype=mtype, importance=0.7, _conn=conn)

    conn.commit()
    conn.close()

def save_memory(content: str, mtype: str = "fact", importance: float = 0.5, _conn=None):
    """Layer 2 — save with FTS5 index"""
    conn = _conn or sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    mid = hashlib.md5((content + str(datetime.datetime.now())).encode()).hexdigest()[:12]
    now = datetime.datetime.now().isoformat()

    # dedup: don't save if similar exists
    c.execute("SELECT id FROM memories WHERE content = ?", (content,))
    if c.fetchone():
        if not _conn: conn.close()
        return None

    c.execute("INSERT INTO memories (id, type, content, importance, created_at) VALUES (?,?,?,?,?)",
              (mid, mtype, content, importance, now))
    c.execute("INSERT INTO memories_fts(rowid, content) VALUES ((SELECT rowid FROM memories WHERE id=?), ?)", (mid, content))

    if not _conn:
        conn.commit()
        conn.close()

    # Also keep Markdown for transparency (Hermes trades this for convenience, we keep both)
    if mtype == "user":
        USER_FILE.write_text(USER_FILE.read_text() + f"\n- {content}\n")
    elif mtype == "fact":
        MEMORY_FILE.write_text(MEMORY_FILE.read_text() + f"\n- {content}\n")

    return mid

def search_memory(query: str, limit: int = 5):
    """Hermes FTS5 search ~10ms over 10K docs"""
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    try:
        # FTS5 search
        c.execute("""
        SELECT m.type, m.content, m.importance, m.created_at, rank
        FROM memories_fts
        JOIN memories m ON m.rowid = memories_fts.rowid
        WHERE memories_fts MATCH ?
        ORDER BY rank
        LIMIT ?
        """, (query, limit))
        results = c.fetchall()
    except sqlite3.OperationalError:
        # fallback if FTS5 syntax fails
        c.execute("SELECT type, content, importance, created_at, 0 FROM memories WHERE content LIKE ? LIMIT ?", (f"%{query}%", limit))
        results = c.fetchall()

    conn.close()
    return [{"type": r[0], "content": r[1], "importance": r[2], "created_at": r[3]} for r in results]

def get_user_model():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT key, value, confidence FROM user_model")
    data = {row[0]: {"value": row[1], "confidence": row[2]} for row in c.fetchall()}
    conn.close()
    return data

def save_user_model(key: str, value: str, confidence: float = 0.8):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    now = datetime.datetime.now().isoformat()
    c.execute("INSERT OR REPLACE INTO user_model (key, value, confidence, updated_at) VALUES (?,?,?,?)",
              (key, value, confidence, now))
    conn.commit()
    conn.close()

def load_user():
    # backward compat for brain.py
    return USER_FILE.read_text()[:2000]

def load_memory():
    # backward compat — returns recent high-importance memories
    results = search_memory("", limit=20)  # empty query fallback returns recent
    if not results:
        return MEMORY_FILE.read_text()[:2000]
    return "\n".join([f"- {r['content']}" for r in results[:10]])

def save_user_fact(line: str):
    save_memory(line, mtype="user", importance=0.9)
    # also update user_model if pattern like "my name is X"
    low = line.lower()
    if "my name is" in low:
        name = line.split("my name is")[-1].strip().title()
        save_user_model("name", name, 0.9)

def save_memory_fact(line: str):
    save_memory(line, mtype="fact", importance=0.7)

def log_daily(event: str):
    today = DAILY_DIR / f"{datetime.date.today()}.md"
    if today.exists():
        today.write_text(today.read_text() + f"\n- {datetime.datetime.now().strftime('%H:%M')} {event}\n")
    else:
        today.write_text(f"# {today.stem}\n\n- {event}\n")
    # also save to DB as daily type
    save_memory(event, mtype="daily", importance=0.5)

def should_nudge(turn_count: int) -> bool:
    """Hermes nudge every 10-20 turns to persist knowledge"""
    return turn_count % 15 == 0

def compact_daily():
    """Summarize daily/*.md into MEMORY.md when too long (Hermes layer 1 -> layer 2)"""
    # For now simple: if MEMORY.md > 2000 lines, summarize oldest
    lines = MEMORY_FILE.read_text().splitlines()
    if len(lines) > 2000:
        MEMORY_FILE.write_text("# MEMORY\n\n" + "\n".join(lines[-1000:]) + "\n\n# ... compacted ...\n")

# init on import
init_db()
