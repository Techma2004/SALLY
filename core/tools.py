import os, requests, datetime
def get_time(): return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z")
def get_weather(city="Calabar"):
    city=(city or "Calabar").strip()[:100]
    if ".." in city or "/" in city: city="Calabar"
    k=os.getenv("OPENWEATHER_API_KEY")
    if not k: return f"[TOOL] Weather key missing. Time: {get_time()} City: {city}"
    try:
        r=requests.get(f"https://api.openweathermap.org/data/2.5/weather?q={city}&units=metric&appid={k}",timeout=8); r.raise_for_status(); d=r.json()
        return f"[TOOL] {d['main']['temp']}C, {d['main']['humidity']}% in {city}. {d['weather'][0]['description']}"
    except Exception as e: return f"[TOOL] Weather err {city}: {e}"[:200]
def get_news(topic="AI"):
    k=os.getenv("NEWS_API_KEY")
    if not k: return "[TOOL] News key missing"
    try:
        r=requests.get(f"https://newsapi.org/v2/everything?q={topic[:50]}&pageSize=3&apiKey={k}",timeout=8); r.raise_for_status()
        return "\n".join([f"- {a['title']}" for a in r.json().get("articles",[])[:3]])
    except Exception as e: return f"News error: {e}"
def calc(expression): return expression
TOOLS={"get_weather":get_weather,"get_time":get_time,"get_news":get_news,"calc":calc}
def execute_tool(name, args=None):
    args=args or {}
    if name=="get_weather": return get_weather(args.get("city","Calabar"))
    if name=="get_time": return get_time()
    if name=="get_news": return get_news(args.get("topic","AI"))
    if name=="calc":
        import ast,operator
        try:
            allowed={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv}
            def ev(n):
                if isinstance(n,ast.Constant): return n.value
                if isinstance(n,ast.BinOp): return allowed[type(n.op)](ev(n.left),ev(n.right))
                if isinstance(n,ast.UnaryOp): return -ev(n.operand)
                raise ValueError("unsafe")
            return str(ev(ast.parse(args.get("expression",""),mode='eval').body))
        except Exception as e: return f"calc error: {e}"
    return f"Unknown {name}"
def get_tool_descriptions_for_prompt(): return "get_weather(city), get_time(), calc(expression)"
