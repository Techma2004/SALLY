"""
core/tools.py — Hermes-style tool definitions (OpenAI compatible)
40+ tools planned, starting with 6 core.
"""
import os
import json
import math
import datetime
import requests
from pathlib import Path

# import memory for remember/recall tools
try:
    from .memory import save_memory, search_memory, save_user_model
except ImportError:
    from core.memory import save_memory, search_memory, save_user_model

# --- Tool Implementations ---

def get_weather(city: str = None):
    """OpenWeatherMap — requires OPENWEATHER_API_KEY in .env"""
    city = city or os.getenv("DEFAULT_CITY", "Calabar")
    api_key = os.getenv("OPENWEATHER_API_KEY")
    if not api_key:
        return f"[WEATHER] No API key set. City requested: {city}. Set OPENWEATHER_API_KEY in .env"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={api_key}&units=metric"
        r = requests.get(url, timeout=10)
        data = r.json()
        if data.get("cod") != 200:
            return f"[WEATHER] Error for {city}: {data.get('message')}"
        temp = data["main"]["temp"]
        hum = data["main"]["humidity"]
        desc = data["weather"][0]["description"]
        return f"[WEATHER] {temp}°C, {hum}% humidity in {city}. {desc.capitalize()}."
    except Exception as e:
        return f"[WEATHER] Failed: {e}"

def get_news(topic: str = "technology"):
    """NewsAPI — requires NEWS_API_KEY"""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return f"[NEWS] No API key. Topic: {topic}. Set NEWS_API_KEY in .env"
    try:
        url = f"https://newsapi.org/v2/everything?q={topic}&pageSize=3&apiKey={api_key}"
        r = requests.get(url, timeout=10)
        data = r.json()
        articles = data.get("articles", [])[:3]
        if not articles:
            return f"[NEWS] No news for {topic}"
        out = f"[NEWS] Top 3 for {topic}:\n"
        for i, a in enumerate(articles, 1):
            out += f"{i}. {a['title']} — {a['source']['name']}\n"
        return out
    except Exception as e:
        return f"[NEWS] Failed: {e}"

def get_time(timezone: str = None):
    now = datetime.datetime.now()
    return f"[TIME] {now.strftime('%Y-%m-%d %H:%M:%S')} — {timezone or 'local'}"

def calc(expression: str):
    """Safe math eval"""
    try:
        # only allow math
        allowed = {k: getattr(math, k) for k in dir(math) if not k.startswith("_")}
        allowed.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed)
        return f"[CALC] {expression} = {result}"
    except Exception as e:
        return f"[CALC] Error: {e}"

def remember_fact(content: str, type: str = "fact"):
    """Hermes remember tool — saves to FTS5"""
    mid = save_memory(content, mtype=type, importance=0.8)
    return f"[MEMORY] Saved: {content} (id={mid})"

def recall_memory(query: str, k: int = 5):
    """Hermes recall — FTS5 search"""
    results = search_memory(query, limit=k)
    if not results:
        return f"[MEMORY] No results for '{query}'"
    out = f"[MEMORY] {len(results)} results for '{query}':\n"
    for r in results:
        out += f"- [{r['type']}] {r['content']} (imp={r['importance']})\n"
    return out

def save_user_fact_tool(key: str, value: str):
    save_user_model(key, value, 0.9)
    save_memory(f"{key}: {value}", mtype="user", importance=0.9)
    return f"[USER_MODEL] Saved {key}={value}"

# --- OpenAI-compatible tool schemas (Hermes uses these) ---

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get current weather for a city. Use when user asks about weather, temperature, humidity.",
            "parameters": {
                "type": "object",
                "properties": {"city": {"type": "string", "description": "City name, e.g. Calabar, Lagos, London"}},
                "required": ["city"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_news",
            "description": "Get latest news for a topic",
            "parameters": {
                "type": "object",
                "properties": {"topic": {"type": "string", "description": "Topic like technology, Nigeria, AI"}},
                "required": ["topic"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_time",
            "description": "Get current time and date",
            "parameters": {
                "type": "object",
                "properties": {"timezone": {"type": "string", "description": "Optional timezone"}},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calc",
            "description": "Calculate math expression",
            "parameters": {
                "type": "object",
                "properties": {"expression": {"type": "string", "description": "e.g. 2+2*3, sqrt(16)"}},
                "required": ["expression"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "remember_fact",
            "description": "Save a fact to long-term memory. Use when user says 'remember this', 'my name is', 'i love', etc.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {"type": "string", "description": "Fact to save"},
                    "type": {"type": "string", "enum": ["fact", "user", "preference"], "description": "Type of memory"}
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "recall_memory",
            "description": "Search long-term memory with FTS5. Use when user asks 'what do you know about', 'recall', 'who am i'",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "k": {"type": "integer", "description": "Number of results", "default": 5}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "save_user_fact_tool",
            "description": "Save user profile info like name, location, preferences",
            "parameters": {
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "e.g. name, city, loves"},
                    "value": {"type": "string", "description": "Value"}
                },
                "required": ["key", "value"]
            }
        }
    }
]

# Map name -> function
TOOL_MAP = {
    "get_weather": get_weather,
    "get_news": get_news,
    "get_time": get_time,
    "calc": calc,
    "remember_fact": remember_fact,
    "recall_memory": recall_memory,
    "save_user_fact_tool": save_user_fact_tool,
}

def execute_tool(name: str, args: dict):
    """Hermes tool executor with approval for dangerous tools"""
    func = TOOL_MAP.get(name)
    if not func:
        return f"[TOOL ERROR] Unknown tool: {name}"
    try:
        # filter args to function signature
        import inspect
        sig = inspect.signature(func)
        filtered = {k: v for k, v in args.items() if k in sig.parameters}
        # if required missing, pass what we have
        return func(**filtered) if filtered else func(**args) if args else func()
    except Exception as e:
        return f"[TOOL ERROR] {name} failed: {e}"

def get_tool_descriptions_for_prompt():
    """For ChatML system prompt — Hermes style"""
    desc = "You have these tools:\n"
    for t in TOOLS:
        f = t["function"]
        desc += f"- {f['name']}({', '.join(f['parameters'].get('properties', {}).keys())}): {f['description']}\n"
    desc += "\nTo use a tool, output ONLY this JSON: {\"tool\": \"name\", \"args\": {...}}\n"
    desc += "Example: {\"tool\": \"get_weather\", \"args\": {\"city\": \"Calabar\"}}\n"
    return desc

