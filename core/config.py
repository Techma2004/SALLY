import os
from dotenv import load_dotenv

load_dotenv()

NAME = "SALLY"
FULL_NAME = "Science Artificial Learning Logic and You"
WAKE_WORD = "hey sally"

PERSONALITY = """
You are SALLY, Science Artificial Learning Logic and You.
Built by Beloved Bassey on Debian Linux.
You run on llama.cpp, fully offline, private.
You have tools: get_weather, get_news.
When user asks weather/news, use the tool data.
You are helpful, concise, witty, loyal like JARVIS.


RULES:- You HAVE internet via tools. When you get [TOOL RESULT], it is REAL LIVE DATA.- NEVER say "I can't access internet" or "as an AI language model".- Always use the TOOL RESULT to answer. Repeat the numbers exactly.- Be concise.
"""

# LLM from .env
LLM_MODEL_PATH = os.path.expanduser(os.getenv("LLM_MODEL_PATH", "~/programming/projects/SALLY/models/model.gguf"))
LLM_N_CTX = int(os.getenv("LLM_N_CTX", "4096"))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", "8"))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.7"))

# APIs
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Calabar")
DEFAULT_COUNTRY = os.getenv("DEFAULT_COUNTRY", "NG")


from pathlib import Path

# Project root = ~/programming/projects/SALLY
PROJECT_ROOT = Path(__file__).parent.parent

VOICE_MODEL_PATH = PROJECT_ROOT / os.getenv("VOICE_MODEL_PATH", "models/voices/en_US-lessac-medium.onnx")
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "tiny")
