import requests
from. import config

def get_weather(city=None):
    """Get real-time weather"""
    city = city or config.DEFAULT_CITY
    if not config.OPENWEATHER_API_KEY or "YOUR_" in config.OPENWEATHER_API_KEY:
        return f"Weather API key not set in.env. Can't fetch weather for {city}."

    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={config.OPENWEATHER_API_KEY}&units=metric"
        r = requests.get(url, timeout=10)
        data = r.json()

        if r.status_code!= 200:
            return f"Weather error for {city}: {data.get('message')}"

        temp = data['main']['temp']
        desc = data['weather'][0]['description']
        humidity = data['main']['humidity']

        return f"Weather in {city}: {desc}, {temp}°C, humidity {humidity}%"
    except Exception as e:
        return f"Failed to fetch weather: {e}"

def get_news(topic="technology", count=5):
    """Get latest news"""
    if not config.NEWS_API_KEY or "YOUR_" in config.NEWS_API_KEY:
        return f"News API key not set in.env. Can't fetch news for {topic}."

    try:
        url = f"https://newsapi.org/v2/everything?q={topic}&sortBy=publishedAt&pageSize={count}&apiKey={config.NEWS_API_KEY}"
        r = requests.get(url, timeout=10)
        data = r.json()

        if r.status_code!= 200:
            return f"News error: {data.get('message')}"

        articles = data.get('articles', [])
        if not articles:
            return f"No news found for {topic}"

        result = ""
        for i, a in enumerate(articles, 1):
            result += f"{i}. {a['title']} ({a['source']['name']})\n"
        return result.strip()
    except Exception as e:
        return f"Failed to fetch news: {e}"

# THIS WAS MISSING - This is what brain.py imports
AVAILABLE_TOOLS = {
    "get_weather": get_weather,
    "get_news": get_news
}
