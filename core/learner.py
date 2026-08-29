import sqlite3, datetime, json
from pathlib import Path
ROOT = Path(__file__).parent.parent
MEM_DIR = ROOT / "memory"
DB_PATH = MEM_DIR / "memory.db"
AUTO_SKILLS = MEM_DIR / "skills" / "auto"
AUTO_SKILLS.mkdir(parents=True, exist_ok=True)

def _ensure_table():
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("CREATE TABLE IF NOT EXISTS tool_usage (tool TEXT PRIMARY KEY, count INTEGER DEFAULT 0, last_used TEXT, examples TEXT)")
    conn.commit()
    conn.close()

def log_tool_use(tool_name, args, output):
    _ensure_table()
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    now = datetime.datetime.now().isoformat()
    c.execute("SELECT count, examples FROM tool_usage WHERE tool=?", (tool_name,))
    row = c.fetchone()
    if row:
        count, ex_json = row
        count+=1
        try: examples=json.loads(ex_json) if ex_json else []
        except: examples=[]
        examples.append({"args": args, "output": str(output)[:200], "time": now})
        examples=examples[-5:]
        c.execute("UPDATE tool_usage SET count=?, last_used=?, examples=? WHERE tool=?", (count, now, json.dumps(examples), tool_name))
    else:
        count=1
        examples=[{"args": args, "output": str(output)[:200], "time": now}]
        c.execute("INSERT INTO tool_usage (tool, count, last_used, examples) VALUES (?,?,?,?)", (tool_name, count, now, json.dumps(examples)))
    conn.commit()
    conn.close()
    print(f"[LEARNER] {tool_name} x{count}")
    if count>=5 and count%5==0:
        create_skill_from_tool(tool_name)

def get_tool_stats():
    _ensure_table()
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT tool, count, last_used FROM tool_usage ORDER BY count DESC")
    rows=c.fetchall()
    conn.close()
    return rows

def create_skill_from_tool(tool_name):
    conn = sqlite3.connect(str(DB_PATH))
    c = conn.cursor()
    c.execute("SELECT examples FROM tool_usage WHERE tool=?", (tool_name,))
    row=c.fetchone()
    conn.close()
    examples=json.loads(row[0]) if row and row[0] else []
    code=f"""# Auto-generated skill for {tool_name}
# Created after 5+ uses - {datetime.datetime.now()}
from core.tools import execute_tool
def run_{tool_name}(**kwargs):
    return execute_tool("{tool_name}", kwargs)
# Examples seen: {len(examples)}
"""
    path=AUTO_SKILLS / f"{tool_name}_auto.py"
    if not path.exists():
        path.write_text(code)
        print(f"[LEARNER] Auto-created {path}")
        try:
            from core.memory import save_memory
            save_memory(f"Auto-created skill {tool_name}_auto", mtype="fact", importance=0.8)
        except: pass

def list_auto_skills():
    return list(AUTO_SKILLS.glob("*.py")) if AUTO_SKILLS.exists() else []
