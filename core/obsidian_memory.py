import re, json, hashlib
from pathlib import Path
from datetime import datetime, timezone
from core.config import PROJECT_ROOT, USER_FACTS

MEMORY_ROOT = PROJECT_ROOT / "memory"
FACTS_DIR = MEMORY_ROOT / "facts"
EVENTS_DIR = MEMORY_ROOT / "events"
PEOPLE_DIR = MEMORY_ROOT / "people"
PROJECTS_DIR = MEMORY_ROOT / "projects"
CONTEXT_DIR = MEMORY_ROOT / "context"
DECISIONS_DIR = MEMORY_ROOT / "decisions"
INSIGHTS_DIR = MEMORY_ROOT / "insights"
INDEXES_DIR = MEMORY_ROOT / "_indexes"
OPS_DIR = MEMORY_ROOT / "_ops" / "applied"

for d in [FACTS_DIR, EVENTS_DIR, PEOPLE_DIR, PROJECTS_DIR, CONTEXT_DIR, DECISIONS_DIR, INSIGHTS_DIR, INDEXES_DIR, OPS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

def _slugify(text: str, max_len=40) -> str:
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_-]+", "-", text).strip("-")
    return text[:max_len] or "note"

def _sanitize_q(q: str) -> list:
    """No FTS5 crash — pure python tokenization"""
    q = re.sub(r"[^\w\s]", " ", q.lower())
    toks = [t for t in q.split() if len(t) > 1][:10]
    return toks

def _frontmatter(entity, predicate, value, source="chat"):
    return f"""---
entity: {entity}
predicate: {predicate}
value: "{value.replace('"','\\"')}"
source: {source}
timestamp: {datetime.now(timezone.utc).isoformat()}
---

{value}
"""

def remember(content: str, type="episodic", entity=None, predicate=None, source="chat"):
    """
    type:
      - episodic -> memory/events/YYYY-MM-DD/slug.md
      - semantic/fact -> memory/facts/{entity}/{predicate}.md (one fact, one file)
      - people/project -> narrative
    """
    entity = entity or USER_FACTS.get("user_handle","user").lower()
    entity = _slugify(entity)

    if type in ["semantic", "fact"]:
        # One fact, one file — path is primary key
        pred = predicate or _slugify(content.split(":")[0]) or "note"
        # If content is "key: value" split it
        if ":" in content and not predicate:
            k, v = content.split(":", 1)
            pred = _slugify(k)
            value = v.strip()
        else:
            value = content
            pred = _slugify(pred or "note")

        fact_dir = FACTS_DIR / entity
        fact_dir.mkdir(parents=True, exist_ok=True)
        fact_path = fact_dir / f"{pred}.md"

        # Transactional write + receipt (v4 idea)
        tx_id = hashlib.sha256(f"{entity}/{pred}/{value}".encode()).hexdigest()[:12]
        fact_path.write_text(_frontmatter(entity, pred, value, source))
        (OPS_DIR / f"{datetime.now().strftime('%Y%m%d-%H%M%S')}_{tx_id}.md").write_text(
            f"---\nop: create_fact\nentity: {entity}\npredicate: {pred}\ntx: {tx_id}\n---\nApplied {fact_path}\n"
        )
        print(f"[MEMORY] +fact {entity}/{pred} #{tx_id}")
        return fact_path

    else: # episodic / event
        day_dir = EVENTS_DIR / datetime.now().strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        slug = _slugify(content) or "event"
        path = day_dir / f"{datetime.now().strftime('%H%M%S')}_{slug}.md"
        path.write_text(f"""---
type: episodic
entity: {entity}
timestamp: {datetime.now(timezone.utc).isoformat()}
source: {source}
---

{content}
""")
        print(f"[MEMORY] +event {path.relative_to(MEMORY_ROOT)}")
        return path

def _read_fact_file(p: Path):
    try:
        text = p.read_text()
        # Parse frontmatter value
        m = re.search(r'value:\s*"?(.*?)"?\s*\n', text)
        value = m.group(1) if m else text
        return value
    except: return ""

def recall(query: str, k=3, entity=None):
    """Lexical search over Markdown — no FTS5, no crash on can't"""
    tokens = _sanitize_q(query)
    if not tokens:
        return []

    candidates = []
    search_roots = [FACTS_DIR, EVENTS_DIR]
    if entity:
        search_roots = [FACTS_DIR / _slugify(entity)]

    for root in search_roots:
        if not root.exists(): continue
        for md_file in root.rglob("*.md"):
            if "_ops" in str(md_file) or "_indexes" in str(md_file):
                continue
            try:
                content = md_file.read_text().lower()
                score = sum(1 for t in tokens if t in content)
                if score > 0:
                    # Boost facts over events
                    if "facts" in str(md_file): score += 1
                    candidates.append((score, md_file, content[:500]))
            except: continue

    candidates.sort(key=lambda x: x[0], reverse=True)
    results = []
    for score, path, snippet in candidates[:k]:
        rel = path.relative_to(MEMORY_ROOT)
        results.append({
            "id": str(rel),
            "type": "fact" if "facts" in str(path) else "episodic",
            "content": snippet.strip(),
            "path": str(path),
            "score": score
        })
    return results

def get_fact(entity, predicate):
    p = FACTS_DIR / _slugify(entity) / f"{_slugify(predicate)}.md"
    if p.exists():
        return _read_fact_file(p)
    return None

def list_facts(entity):
    d = FACTS_DIR / _slugify(entity)
    if not d.exists(): return {}
    facts = {}
    for f in d.glob("*.md"):
        facts[f.stem] = _read_fact_file(f)
    return facts

def init_core_from_facts():
    """Migrate USER_FACTS to facts/ on first run — no more human.json confusion"""
    handle = _slugify(USER_FACTS.get("user_handle","user"))
    fact_dir = FACTS_DIR / handle
    if not any(fact_dir.glob("*.md")):
        print(f"[MEMORY] Migrating USER_FACTS to facts/{handle}/...")
        for k,v in USER_FACTS.items():
            # k like user_name -> predicate name
            pred = k.replace("user_","")
            remember(f"{v}", type="fact", entity=handle, predicate=pred, source="facts")
        # Also create people narrative
        people_file = PEOPLE_DIR / f"{handle}.md"
        if not people_file.exists():
            people_file.write_text(f"""---
entity: {handle}
type: people
---

# {USER_FACTS.get('user_name')} aka {USER_FACTS.get('user_handle')}

- **Location:** {USER_FACTS.get('user_location')}
- **Hardware:** {USER_FACTS.get('user_hardware')}
- **Interests:** {USER_FACTS.get('user_interests')}
- **Dream:** {USER_FACTS.get('user_dream_project')}

Related facts: [[facts/{handle}/name]] [[facts/{handle}/hardware]]
""")
