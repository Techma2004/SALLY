"""skill: get headlines. Trigger: news, headlines"""
import requests

from core.config import settings


def run(query):
    api_key = settings.integrations.news_api_key

    if not api_key or "YOUR_" in api_key:
        return "[TOOL] No NEWS_API_KEY set in .env — set it to get live headlines"

    try:
        r = requests.get(
            "https://newsapi.org/v2/top-headlines",
            params={
                "country": settings.integrations.default_country.lower(),
                "pageSize": 5,
                "apiKey": api_key,
            },
            timeout=6,
        )
        r.raise_for_status()
        arts = r.json().get("articles", [])[:3]
        return "\n".join([f"- {a['title']}" for a in arts]) or "No news found"
    except Exception as e:
        return f"news error: {e}"
