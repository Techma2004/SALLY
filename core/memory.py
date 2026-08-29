import sqlite3, re, uuid, json
from pathlib import Path
from core.config import PROJECT_ROOT
DB_PATH=PROJECT_ROOT/"memory"/"memory.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)
def sanitize_fts5(q):
    q=re.sub(r'["*:\-]', ' ', q); q=re.sub(r'\s+', ' ', q).strip()
    return q[:200] or "SALLY"
def get_conn():
    c=sqlite3.connect(DB_PATH, timeout=10, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL;"); c.execute("PRAGMA busy_timeout=5000;")
    return c
def init_db():
    c=get_conn()
    c.executescript("""
    CREATE TABLE IF NOT EXISTS memories (id TEXT PRIMARY KEY, content TEXT NOT NULL, type TEXT NOT NULL, created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP);
    CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts USING fts5(content, content='memories', content_rowid='rowid');
    CREATE TRIGGER IF NOT EXISTS memories_ai AFTER INSERT ON memories BEGIN INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content); END;
    CREATE TRIGGER IF NOT EXISTS memories_ad AFTER DELETE ON memories BEGIN INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content); END;
    CREATE TRIGGER IF NOT EXISTS memories_au AFTER UPDATE ON memories BEGIN INSERT INTO memories_fts(memories_fts, rowid, content) VALUES('delete', old.rowid, old.content); INSERT INTO memories_fts(rowid, content) VALUES (new.rowid, new.content); END;
    """); c.commit(); c.close()
def save_memory(content,mtype="episodic"):
    if not content or len(content)>5000: return
    if ".." in mtype or "/" in mtype: mtype="episodic"
    init_db(); c=get_conn(); c.execute("INSERT INTO memories(id,content,type) VALUES(?,?,?)",(str(uuid.uuid4())[:12],content,mtype)); c.commit(); c.close()
    ed=PROJECT_ROOT/"memory"/"episodes"; ed.mkdir(parents=True, exist_ok=True)
    open(ed/f"{mtype}.md","a",encoding="utf-8").write(f"- {content}\n")
def search_memory(query,k=5):
    init_db(); c=get_conn(); q=sanitize_fts5(query)
    try: cur=c.execute("SELECT content FROM memories_fts WHERE memories_fts MATCH? ORDER BY rank LIMIT?",(q,k)); rows=[r[0] for r in cur.fetchall()]
    except: cur=c.execute("SELECT content FROM memories WHERE content LIKE? LIMIT?",(f"%{q}%",k)); rows=[r[0] for r in cur.fetchall()]
    c.close(); return rows
def load_user():
    try:
        hp=PROJECT_ROOT/"memory"/"core"/"human.json"
        if hp.exists(): return json.loads(hp.read_text())
    except: pass
    return {"user_name":"Edima"}
def load_memory(): return "\n".join(search_memory("SALLY",10))
