from core.brain import chat
from core.memory import search_memory, init_db

init_db()
print("Test chat:", chat("user", "hello"))
print("Test tool:", chat("user", "time"))
print("Test tool:", chat("user", "calc 25*40"))
print("Test tool:", chat("user", "weather in Lagos"))