import json, os, datetime
MEMORY_FILE = "memory.json"

def load_memory():
    if not os.path.exists(MEMORY_FILE):
        return []
    with open(MEMORY_FILE, 'r') as f:
        return json.load(f)

def save_memory(role, content):
    data = load_memory()
    data.append({"role": role, "content": content, "time": str(datetime.datetime.now())})
    # keep last 50
    data = data[-50:]
    with open(MEMORY_FILE, 'w') as f:
        json.dump(data, f, indent=2)
