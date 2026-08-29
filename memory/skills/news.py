"""skill: get headlines. Trigger: news, headlines"""
import os, requests
from core.config import NEWS_API_KEY

def run(query):
    if not NEWS_API_KEY or "YOUR_" in NEWS_API_KEY:
        return "[TOOL] No NEWS_API_KEY set in.env — set it to get live headlines"

    try:
        r = requests.get(f"https://newsapi.org/v2/top-headlines?country=us&pageSize=5&apiKey={NEWS_API_KEY}", timeout=6)
        arts = r.json().get("articles", [])[:3]
        return "\n".join([f"- {a['title']}" for a in arts]) or "No news found"
    except Exception as e:
        return f"news error: {e}"
