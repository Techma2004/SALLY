"""skill: get weather for a city. Trigger: weather, temperature, rain"""
import os, re, requests
from core.config import DEFAULT_CITY, OPENWEATHER_API_KEY

def run(query):
    city = DEFAULT_CITY
    m = re.search(r"in ([a-zA-Z ]+)", query.lower())
    if m: city = m.group(1).strip().title()

    if not OPENWEATHER_API_KEY or "YOUR_" in OPENWEATHER_API_KEY:
        return f"[TOOL] No OPENWEATHER_API_KEY set in.env — can't fetch live weather for {city}. Get free key at openweathermap.org/api"

    try:
        r = requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric", timeout=6)
        d = r.json()
        return f"{city}: {d['weather'][0]['description']}, {d['main']['temp']}°C, humidity {d['main']['humidity']}%"
    except Exception as e:
        return f"weather error: {e}"
