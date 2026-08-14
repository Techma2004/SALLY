import requests
from. import config

def get_weather(city=None):
    city = city or config.DEFAULT_CITY
    if not config.OPENWEATHER_API_KEY:
        return "Weather API key missing in.env"
    url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={config.OPENWEATHER_API_KEY}&units=metric"
    r = requests.get(url).json()
    return f"{city}: {r['weather'][0]['description']}, {r['main']['temp']}°C, humidity {r['main']['humidity']}%"

def get_news(topic="technology"):
    if not config.NEWS_API_KEY:
        return "News API key missing in.env"
    url = f"https://newsapi.org/v2/everything?q={topic}&pageSize=5&apiKey={config.NEWS_API_KEY}"
    r = requests.get(url).json()
    return "\n".join([f"- {a['title']}" for a in r.get('articles', [])])
