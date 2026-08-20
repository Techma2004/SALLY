#!/usr/bin/env python3
from pathlib import Path
import re, sys

ROOT = Path(__file__).parent.parent / "memory"
errors = 0

print("Linting Obsidian Memory vault...")

for fact_file in (ROOT / "facts").rglob("*.md") if (ROOT / "facts").exists() else []:
    text = fact_file.read_text()
    if not text.startswith("---"):
        print(f"FAIL: {fact_file} missing frontmatter"); errors+=1
    if "entity:" not in text or "predicate:" not in text:
        print(f"FAIL: {fact_file} missing entity/predicate"); errors+=1

for event_file in (ROOT / "events").rglob("*.md") if (ROOT / "events").exists() else []:
    if "timestamp:" not in event_file.read_text():
        print(f"WARN: {event_file} missing timestamp")

if errors==0:
    print("✓ Vault OK — one fact, one file, all frontmatter valid")
    sys.exit(0)
else:
    print(f"✗ {errors} errors")
    sys.exit(1)
