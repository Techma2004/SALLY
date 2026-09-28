"""skill: get weather for a city. Trigger: weather, temperature, rain"""
import re

import requests

from core.config import settings


def run(query):
    api_key = settings.integrations.openweather_api_key
    city = settings.integrations.default_city

    m = re.search(r"in ([a-zA-Z ]+)", query.lower())
    if m:
        city = m.group(1).strip().title()

    if not api_key or "YOUR_" in api_key:
        return (
            f"[TOOL] No OPENWEATHER_API_KEY set in .env — can't fetch live "
            f"weather for {city}. Get a free key at openweathermap.org/api"
        )

    try:
        r = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": api_key, "units": "metric"},
            timeout=6,
        )
        r.raise_for_status()
        d = r.json()
        return (
            f"{city}: {d['weather'][0]['description']}, "
            f"{d['main']['temp']}°C, humidity {d['main']['humidity']}%"
        )
    except Exception as e:
        return f"weather error: {e}"
