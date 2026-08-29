import os, json, re
from pathlib import Path
try:
    from llama_cpp import Llama
except ImportError:
    Llama=None
from.config import load_config, load_facts
from.memory import search_memory, load_user, load_memory
from.tools import TOOLS, execute_tool, get_tool_descriptions_for_prompt
try:
    from.learner import log_tool_use
    HAS_LEARNER=True
except:
    HAS_LEARNER=False
    def log_tool_use(*a,**k): pass

PERSONALITY="""You are SALLY - Science Artificial Learning Logic and You. Offline, private, helpful. Built by Edima Bassey in Calabar. You NEVER say you are JARVIS. You are SALLY."""

def build_system_prompt():
    return f"{PERSONALITY}\n\nUSER: {load_user()[:800]}\nMEMORY: {load_memory()[:800]}\n\n{get_tool_descriptions_for_prompt()}"

_llm=None
def get_llm():
    global _llm
    if _llm is not None: return _llm
    if Llama is None: return None
    cfg=load_config()
    mp=cfg.get("model_path")
    if not mp or not Path(mp).exists(): return None
    if "mmproj" in str(mp).lower(): return None
    print(f"[BRAIN] Loaded {mp}")
    _llm=Llama(model_path=str(mp), n_ctx=cfg.get("n_ctx",4096), n_threads=cfg.get("n_threads",8), verbose=False)
    return _llm

def parse_tool_call(text):
    m=re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if m: text=m.group(1)
    jm=re.search(r"\{.*\}", text, re.DOTALL)
    if jm:
        try:
            data=json.loads(jm.group(0))
            if "tool" in data: return data["tool"], data.get("args") or {}
            if "name" in data: return data["name"], data.get("args") or {}
        except: pass
    return None,None

def detect_intent(text):
    low=text.lower()
    if "weather" in low:
        m=re.search(r"weather in ([a-z ]+)", low)
        city=m.group(1).strip() if m else "Calabar"
        city=city.replace(" now","").replace(" today","").split()[0]
        return "get_weather", {"city": city}
    if ("time" in low or "date" in low) and "what" in low:
        return "get_time", {}
    if "news" in low:
        m=re.search(r"news (?:about|on|for|in) ([a-z ]+)", low)
        topic=m.group(1).strip() if m else "technology"
        return "get_news", {"topic": topic}
    if "calc" in low or re.search(r"[0-9]+\s*[+\-*/]\s*[0-9]+", text):
        expr=re.sub(r"(?i)calculate|calc|what is|what's|equals|=", "", text)
        expr=re.sub(r"[^0-9+\-*/().% ]", "", expr).strip()
        if expr and len(expr)>=3:
            return "calc", {"expression": expr}
    if low.startswith("recall ") or low.startswith("/memory"):
        q=text.split(" ",1)[1] if " " in text else ""
        return "recall_memory", {"query": q}
    return None,None

def filter_identity(t):
    return t.replace("JARVIS","SALLY").replace("Meta AI","SALLY")

def chat(user_input, history=None):
    iname,iargs=detect_intent(user_input)
    llm=get_llm()
    if iname:
        print(f"[BRAIN] Forced tool: {iname} {iargs}")
        out=execute_tool(iname, iargs)
        try: log_tool_use(iname, iargs, out)
        except: pass
        if llm is None: return out
        msgs=[{"role":"system","content":f"Tool {iname} returned: {out}. Answer naturally."},{"role":"user","content":user_input}]
        try:
            r=llm.create_chat_completion(messages=msgs, temperature=0.7, max_tokens=256)
            return filter_identity(r["choices"][0]["message"]["content"])
        except: return out
    if llm is None: return f"[offline] {user_input}"
    sys_prompt=build_system_prompt()
    msgs=[{"role":"system","content":sys_prompt}]
    for h in (history or [])[-6:]: msgs.append(h)
    msgs.append({"role":"user","content":user_input})
    try:
        resp=llm.create_chat_completion(messages=msgs, temperature=0.7, max_tokens=512)
        txt=resp["choices"][0]["message"]["content"].strip()
    except Exception as e: return f"[ERROR] {e}"
    name,args=parse_tool_call(txt)
    if name:
        out=execute_tool(name, args or {})
        try: log_tool_use(name, args or {}, out)
        except: pass
        msgs.append({"role":"assistant","content":txt})
        msgs.append({"role":"user","content":f"Tool returned: {out}. Answer."})
        try:
            r2=llm.create_chat_completion(messages=msgs, temperature=0.7, max_tokens=256)
            return filter_identity(r2["choices"][0]["message"]["content"])
        except: return out
    return filter_identity(txt)

def respond(p,h=None): return chat(p,h)
def think(p,h=None,t=None): return chat(p,h)
