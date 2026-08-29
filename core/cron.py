"""
core/cron.py — Phase 6 Cron v0.46
No dependency on handle_tool_call — uses direct APIs + memory
"""
import os, json, requests
from pathlib import Path
from datetime import datetime
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).parent.parent
load_dotenv(PROJECT_ROOT / ".env", override=True)

DAILY_DIR = PROJECT_ROOT / "memory" / "daily"
DAILY_DIR.mkdir(parents=True, exist_ok=True)
CHAT_IDS_FILE = PROJECT_ROOT / "memory" / "telegram_chat_ids.json"

def get_weather():
    key = os.getenv("OPENWEATHER_API_KEY")
    city = os.getenv("DEFAULT_CITY","Calabar")
    if not key:
        return "Weather: no OPENWEATHER_API_KEY in.env"
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={key}&units=metric"
        r = requests.get(url, timeout=10).json()
        if "main" in r:
            return f"{city}: {r['main']['temp']}C, {r['weather'][0]['description']}, humidity {r['main']['humidity']}%"
        return str(r)[:300]
    except Exception as e:
        return f"Weather error: {e}"

def get_news():
    key = os.getenv("NEWS_API_KEY")
    if not key:
        return "News: no NEWS_API_KEY"
    try:
        url = f"https://newsapi.org/v2/everything?q=Nigeria&language=en&sortBy=publishedAt&pageSize=3&apiKey={key}"
        r = requests.get(url, timeout=10).json()
        arts = r.get("articles",[])
        if not arts:
            return "No news"
        return "\n".join([f"• {a['title']} ({a['source']['name']})" for a in arts])
    except Exception as e:
        return f"News error: {e}"

def get_memory_recap():
    try:
        from core.memory import search_memory
        res = search_memory("Edima", 5)
        if not res:
            return "No memories yet"
        return "\n".join([f"- {c['content'][:120]}" if isinstance(c, dict) else f"- {str(c)[:120]}" for c in res])
    except Exception as e:
        return f"Memory error: {e}"

def get_learner_stats():
    try:
        from core.learner import get_tool_stats
        stats = get_tool_stats()
        if not stats:
            return "No tool usage yet"
        return ", ".join([f"{t} x{c}" for t,c,_ in stats[:6]])
    except Exception as e:
        return f"Learner: {e}"

def build_brief():
    now = datetime.now().strftime("%Y-%m-%d %H:%M WAT")
    return f"""# SALLY Daily Brief — {now}
Location: Calabar, NG

## Weather
{get_weather()}

## Memory Recap
{get_memory_recap()}

## News — Nigeria
{get_news()}

## Learning
{get_learner_stats()}

— SALLY v0.46 Cron
"""

def save_daily(brief: str):
    date = datetime.now().strftime("%Y-%m-%d")
    path = DAILY_DIR / f"{date}.md"
    path.write_text(brief)
    print(f"[CRON] Saved {path}")
    try:
        from core.memory import save_memory
        save_memory(f"Daily brief {date}: {brief[:400]}", mtype="daily", importance=0.8)
    except: pass
    return path

def send_telegram(brief: str):
    token = os.getenv("TELEGRAM_BOT_TOKEN","")
    if not token or "PUT" in token:
        envf = PROJECT_ROOT / ".env"
        if envf.exists():
            for line in envf.read_text().splitlines():
                if line.startswith("TELEGRAM_BOT_TOKEN="):
                    v = line.split("=",1)[1].strip().strip('"').strip("'")
                    if "=" in v: v = v.split("=")[-1]
                    if v and "PUT" not in v and ":" in v:
                        token = v
                        break
    if not token:
        print("[CRON] No Telegram token")
        return
    if not CHAT_IDS_FILE.exists():
        print(f"[CRON] No {CHAT_IDS_FILE} — message @Sally_12345_bot first")
        return
    try:
        chat_ids = json.loads(CHAT_IDS_FILE.read_text())
    except:
        chat_ids = []
    if not chat_ids:
        print("[CRON] No chat_ids saved")
        return
    short = brief[:3500]
    for cid in chat_ids:
        try:
            r = requests.post(
                f"https://api.telegram.org/bot{token}/sendMessage",
                json={"chat_id": cid, "text": f"Good morning Edima ☀️\n\n{short}\n\nFull: memory/daily/"}
            )
            print(f"[CRON] Telegram push {cid}: {r.status_code} {r.text[:100]}")
        except Exception as e:
            print(f"[CRON] Telegram fail: {e}")

def run_brief_job():
    print(f"[CRON] Running brief at {datetime.now()}")
    brief = build_brief()
    print(brief)
    save_daily(brief)
    send_telegram(brief)

def main():
    import sys
    if "--now" in sys.argv:
        run_brief_job()
        return
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
    except ImportError:
        print("pip install apscheduler")
        return
    scheduler = BlockingScheduler(timezone="Africa/Lagos")
    scheduler.add_job(run_brief_job, 'cron', hour=7, minute=0, id='daily_brief')
    print("[CRON] SALLY Cron v0.46 — daily 7:00 AM Africa/Lagos")
    print("[CRON] --now to test now")
    scheduler.start()

if __name__ == "__main__":
    main()
