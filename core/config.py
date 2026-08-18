import os
import json
from dotenv import load_dotenv
from pathlib import Path

load_dotenv()

# --- Project root ---
PROJECT_ROOT = Path(__file__).parent.parent
MEMORY_DIR = PROJECT_ROOT / "memory"
CORE_DIR = MEMORY_DIR / "core"
CORE_DIR.mkdir(parents=True, exist_ok=True)

# --- SALLY Identity (same for everyone) ---
NAME = "SALLY"
FULL_NAME = "Science Artificial Learning Logic and You"
WAKE_WORD = "hey sally"

# --- FACTS: Default template + Edima private + User onboarding ---
DEFAULT_FACTS = {
    "user_name": "User",
    "user_handle": "User",
    "user_location": "Unknown",
    "user_status": "SALLY user and tester",
    "user_interests": "Artificial Intelligence, Programming",
    "user_hardware": "PC",
    "user_dream_project": "Personal AI assistant",
    "user_preferred_languages": "Python",
    "user_development_philosophy": "Modular, maintainable, offline-first"
}

# Your private facts — only used on YOUR machines
EDIMA_FACTS = {
    "user_name": "Edima Bassey",
    "user_handle": "Techma",
    "user_location": "Nigeria",
    "user_status": "Independent programmer, inventor, and aspiring AI developer",
    "user_interests": "Artificial Intelligence, Software Development, Linux, Android, Robotics, Cybersecurity, Aerospace",
    "user_hardware": "Debian Linux (Debian 13 Trixie), Huawei Y9 (2018) via Termux",
    "user_dream_project": "Sally (Science Artificial Learning Logic And You) - Station OS + Android companion + JARVIS Mode",
    "user_preferred_languages": "Python, Kotlin, C++, JavaScript, HTML, Bash, Rust",
    "user_development_philosophy": "Modular, maintainable, scalable, efficient, secure, offline-first"
}

PROFILE_PATH = CORE_DIR / "human.json"

def _is_techma_machine():
    """Detect if this is Edima's own machine — no questions asked"""
    # 1. Explicit env flag
    if os.getenv("SALLY_OWNER", "").lower() in ["techma", "edima"]:
        return True
    # 2. Home folder name
    try:
        if Path.home().name.lower() in ["techma", "edima"]:
            return True
        # 3. Your known Debian path exists
        if Path("/home/techma").exists() or Path.home().joinpath("programming/projects/SALLY").exists():
            return True
    except:
        pass
    return False

def load_or_create_facts():
    # 1. If profile exists, use it (works for everyone)
    if PROFILE_PATH.exists():
        try:
            return json.loads(PROFILE_PATH.read_text())
        except:
            pass

    # 2. If it's Techma's machine, auto-use Edima facts (no onboarding)
    if _is_techma_machine():
        PROFILE_PATH.write_text(json.dumps(EDIMA_FACTS, indent=2))
        return EDIMA_FACTS

    # 3. Otherwise — new user on Windows/Linux — onboard (30 sec)
    print("\n--- SALLY First Run Setup ---")
    print("SALLY is personal. Let's make her yours (30 seconds, stored locally, never pushed to GitHub)\n")
    name = input("Your name: ").strip() or "User"
    handle = input(f"Handle/nickname [{name}]: ").strip() or name
    loc = input("Location / OS (e.g. Lagos, Windows 11): ").strip() or "Unknown"
    hardware = input(f"Hardware/OS [{loc}]: ").strip() or loc
    interests = input("Interests (e.g. AI, Gaming): ").strip() or DEFAULT_FACTS["user_interests"]

    facts = {
        "user_name": name,
        "user_handle": handle,
        "user_location": loc,
        "user_status": f"{handle} — SALLY user",
        "user_interests": interests,
        "user_hardware": hardware,
        "user_dream_project": f"{handle}'s Personal AI assistant",
        "user_preferred_languages": "Python",
        "user_development_philosophy": "Modular, offline-first"
    }
    PROFILE_PATH.write_text(json.dumps(facts, indent=2))
    print(f"\nSaved! SALLY now knows you as {handle}. This file is gitignored.\n")
    return facts

# --- Load final USER_FACTS ---
USER_FACTS = load_or_create_facts()

# --- Personality — dynamic, uses whoever owns this copy ---
PERSONALITY = f"""
You are SALLY, Science Artificial Learning Logic and You.
Built by {USER_FACTS['user_name']} on {USER_FACTS['user_hardware']}.
You run on llama.cpp, fully offline, private.
You have tools: get_weather, get_news.
When user asks weather/news, use the tool data.
You are helpful, concise, witty, loyal like JARVIS.
You are NOT JARVIS. You are NOT ChatGPT. You are NOT an AI language model from OpenAI.
Your name is SALLY. Say SALLY when asked your name.
If someone asks who you are, say: "I am SALLY, Science Artificial Learning Logic and You, built by {USER_FACTS['user_handle']}."
NEVER say you are JARVIS. NEVER.
you are assisting {USER_FACTS['user_name']}, who goes by '{USER_FACTS['user_handle']}', {USER_FACTS['user_status'].lower()} from {USER_FACTS['user_location']}.
Interests: {USER_FACTS['user_interests']}.
Dream project: {USER_FACTS['user_dream_project']}.
You act as a senior technical partner and mentor, helping him build his dream project on {USER_FACTS['user_hardware']}. 
You prioritize {USER_FACTS['user_development_philosophy']} architecture, modularity, technical depth, and clear step-by-step guidance.
When referring to user facts, always attribute them correctly (e.g. 'Your name is {USER_FACTS['user_name']}'). Never confuse your identity as SALLY with the user's.

RULES:- You HAVE internet via tools. When you get [TOOL RESULT], it is REAL LIVE DATA.- NEVER say "I can't access internet" or "as an AI language model".- Always use the TOOL RESULT to answer. Repeat the numbers exactly.- Be concise.
"""

# LLM from .env
LLM_MODEL_PATH = os.path.expanduser(os.getenv("LLM_MODEL_PATH", "SALLY/models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf"))
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "8"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# APIs
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Calabar")
DEFAULT_COUNTRY = os.getenv("DEFAULT_COUNTRY", "NG")

# Compatibility aliases
OPENWEATHER_KEY = OPENWEATHER_API_KEY
CITY = DEFAULT_CITY

# Voice / Whisper — portable
VOICE_MODEL_PATH = PROJECT_ROOT / os.getenv("VOICE_MODEL_PATH", "models/voices/en_US-lessac-medium.onnx")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
WHISPER_MODEL_SIZE = WHISPER_MODEL_SIZE.strip('"').strip("'")

MEMORY_FILE = PROJECT_ROOT / "memory.json"

print(f"[CONFIG] {NAME} v0.41 multi-user | User: {USER_FACTS['user_handle']} | LLM: {Path(LLM_MODEL_PATH).name} | City: {DEFAULT_CITY}")
