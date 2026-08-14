import os
from dotenv import load_dotenv

load_dotenv() # loads.env automatically

NAME = "SALLY"
FULL_NAME = "Science Artificial Learning Logic and You"

# LLM from.env
LLM_MODEL_PATH = os.path.expanduser(os.getenv("LLM_MODEL_PATH"))
LLM_N_CTX = int(os.getenv("LLM_N_CTX", 4096))
LLM_N_THREADS = int(os.getenv("LLM_N_THREADS", 8))
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", 0.7))

# APIs from.env
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
DEFAULT_CITY = os.getenv("DEFAULT_CITY", "Calabar")
