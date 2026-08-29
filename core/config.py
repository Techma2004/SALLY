
import os
import json
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_ROOT / ".env")
except:
    pass
SETTINGS_PATH = PROJECT_ROOT / "settings.json"
EXAMPLE_PATH = PROJECT_ROOT / "example.settings.json"
def _load_settings_file():
    path = SETTINGS_PATH if SETTINGS_PATH.exists() else EXAMPLE_PATH
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except:
        return {}
def load_config():
    file_cfg = _load_settings_file()
    model_path = os.getenv("LLM_MODEL_PATH") or file_cfg.get("LLM_MODEL_PATH") or file_cfg.get("model_path") or "models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf"
    if model_path and "mmproj" in str(model_path).lower():
        print(f"[CONFIG] WARNING: {model_path} looks like mmproj (vision), ignoring")
        model_path = file_cfg.get("LLM_MODEL_PATH", "models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf")
    cfg = {
        "model_path": str(PROJECT_ROOT / model_path) if not os.path.isabs(model_path) else model_path,
        "n_ctx": int(os.getenv("LLM_N_CTX") or file_cfg.get("LLM_N_CTX", 4096)),
        "n_threads": int(os.getenv("LLM_N_THREADS") or file_cfg.get("LLM_N_THREADS", 8)),
        "temperature": float(os.getenv("LLM_TEMPERATURE") or file_cfg.get("LLM_TEMPERATURE", 0.7)),
        "city": os.getenv("DEFAULT_CITY") or file_cfg.get("DEFAULT_CITY", "Calabar"),
        "raw": file_cfg
    }
    return cfg
def load_facts():
    user_file = PROJECT_ROOT / "memory" / "USER.md"
    if user_file.exists():
        return user_file.read_text()
    return "No user facts yet."
def load_user_facts():
    return load_facts()
def get_default_city():
    return load_config().get("city", "Calabar")
