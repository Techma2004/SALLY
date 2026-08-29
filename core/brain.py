
import os, json, re
from pathlib import Path
try:
    from llama_cpp import Llama
except ImportError:
    Llama=None
from.config import load_config, load_facts
from.memory import search_memory, load_user, load_memory, save_memory, log_daily, should_nudge
from.tools import TOOLS, execute_tool, get_tool_descriptions_for_prompt
PROJECT_ROOT=Path(__file__).parent.parent
PERSONALITY="""You are SALLY — Science Artificial Learning Logic and You. Offline, private, helpful. Built by Edima Bassey in Calabar. You NEVER say you are JARVIS. You are SALLY."""
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
    _llm=Llama(model_path=str(mp), n_ctx=cfg.get("n_ctx",4096), n_threads=cfg.get("n_threads",8), verbose=False)
    print(f"[BRAIN] Loaded {mp}")
    return _llm
def parse_tool_call(text):
    text=text.strip()
    m=re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if m: text=m.group(1)
    for cand in [text]:
        try:
            data=json.loads(re.search(r"\{.*\}", text, re.DOTALL).group(0) if "{" in text else text)
            if "tool" in data: return data["tool"], data.get("args") or data.get("arguments") or {}
            if "name" in data: return data["name"], data.get("arguments") or data.get("args") or {}
        except: continue
    return None,None
def detect_intent(text):
    low=text.lower()
    if "weather" in low:
        m=re.search(r"weather in ([a-zA-Z ]+)", low)
        city=m.group(1).strip() if m else "Calabar"
        return "get_weather", {"city": city}
    if "time" in low or "what time" in low or "date" in low:
        return "get_time", {}
    if "news" in low:
        m=re.search(r"news (?:about|on|for) ([a-zA-Z ]+)", low)
        topic=m.group(1).strip() if m else "technology"
        return "get_news", {"topic": topic}
    if "calc" in low or re.search(r"[0-9]+\s*[+\-*/]+\s*[0-9]+", text):
        m=re.search(r"calc (.+)", low)
        expr=m.group(1) if m else text
        # clean
        expr=re.sub(r"[^0-9+\-*/(). sqrt]", "", expr)
        return "calc", {"expression": expr}
    if low.startswith("recall ") or "remember" in low and "?" in text:
        q=text[7:] if low.startswith("recall ") else text
        return "recall_memory", {"query": q}
    return None,None
def filter_identity(t):
    for a,b in [("JARVIS","SALLY"),("Meta AI","SALLY")]: t=t.replace(a,b)
    return t
def chat(user_input, history=None):
    # 1. Force tools for small models
    iname,iargs=detect_intent(user_input)
    llm=get_llm()
    if iname:
        print(f"[BRAIN] Forced tool: {iname} {iargs}")
        out=execute_tool(iname, iargs)
        if llm is None:
            return out
        # Let LLM rephrase tool output nicely
        msgs=[{"role":"system","content":f"You are SALLY. Tool {iname} returned: {out}. Answer naturally."},{"role":"user","content":user_input}]
        try:
            r=llm.create_chat_completion(messages=msgs, temperature=0.7, max_tokens=256)
            return filter_identity(r["choices"][0]["message"]["content"])
        except:
            return out
    if llm is None:
        return f"[offline] You: {user_input}"
    sys_prompt=build_system_prompt()
    msgs=[{"role":"system","content":sys_prompt}]
    for h in (history or [])[-6:]: msgs.append(h)
    msgs.append({"role":"user","content":user_input})
    try:
        resp=llm.create_chat_completion(messages=msgs, temperature=0.7, max_tokens=512)
        txt=resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[ERROR] {e}"
    name,args=parse_tool_call(txt)
    if name:
        out=execute_tool(name, args or {})
        msgs.append({"role":"assistant","content":txt})
        msgs.append({"role":"user","content":f"Tool returned: {out}. Answer."})
        try:
            r2=llm.create_chat_completion(messages=msgs, temperature=0.7, max_tokens=256)
            return filter_identity(r2["choices"][0]["message"]["content"])
        except:
            return out
    return filter_identity(txt)
def respond(p,h=None): return chat(p,h)
def think(p,h=None,t=None): return chat(p,h)
