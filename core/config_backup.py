import os
import json
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory"
FACTS_DIR = MEMORY_DIR / "facts"
EVENTS_DIR = MEMORY_DIR / "events"
FACTS_DIR.mkdir(parents=True, exist_ok=True)
EVENTS_DIR.mkdir(parents=True, exist_ok=True)

SETTINGS_PATH = PROJECT_ROOT / "settings.json"
EXAMPLE_PATH = PROJECT_ROOT / "example.settings.json"

# Load settings.json, fallback to example.settings.json
if SETTINGS_PATH.exists():
    SETTINGS = json.loads(SETTINGS_PATH.read_text())
elif EXAMPLE_PATH.exists():
    SETTINGS = json.loads(EXAMPLE_PATH.read_text())
else:
    raise FileNotFoundError("Missing settings.json and example.settings.json")

# --- Secrets from.env (must exist, no fallback) ---
LLM_MODEL_PATH = Path(os.environ["LLM_MODEL_PATH"])

# --- Config from settings.json (no fallback in code) ---
LLM_N_CTX = int(SETTINGS["LLM_N_CTX"])
LLM_N_THREADS = int(SETTINGS["LLM_N_THREADS"])
LLM_TEMPERATURE = float(SETTINGS["LLM_TEMPERATURE"])
DEFAULT_CITY = SETTINGS["DEFAULT_CITY"]
DEFAULT_COUNTRY = SETTINGS["DEFAULT_COUNTRY"]
VOICE_MODEL_PATH = SETTINGS["VOICE_MODEL_PATH"]
WHISPER_MODEL_SIZE = SETTINGS["WHISPER_MODEL_SIZE"]

# API keys still from.env (secrets)
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
NEWS_API_KEY = os.getenv("NEWS_API_KEY", "")

PERSONALITY = f"You are SALLY, private offline assistant in {DEFAULT_CITY}. You start blank and learn from user. Use [MEMORY] if given."
