#!/usr/bin/env python3
import sys
sys.path.insert(0, ".")
from core.obsidian_memory import recall, list_facts, get_fact

if len(sys.argv) < 2:
    print("Usage: python3 tools/query.py <query> OR python3 tools/query.py facts --entity techma")
    sys.exit(1)

if sys.argv[1] == "facts":
    entity = sys.argv[sys.argv.index("--entity")+1] if "--entity" in sys.argv else "techma"
    if "--predicate" in sys.argv:
        pred = sys.argv[sys.argv.index("--predicate")+1]
        print(get_fact(entity, pred))
    else:
        for k,v in list_facts(entity).items():
            print(f"{k}: {v}")
else:
    q = " ".join(sys.argv[1:])
    for r in recall(q, k=5):
        print(f"[{r['type']}] {r['id']} (score {r['score']})\n{r['content'][:300]}\n---")
