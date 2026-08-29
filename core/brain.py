import re, json, os, ast, operator
from pathlib import Path
try: from llama_cpp import Llama
except: Llama=None
from core.config import PROJECT_ROOT, load_user
from core.memory import search_memory, save_memory
from core.tools import get_weather, get_time, get_news
_llm=None
def get_llm():
    global _llm
    if _llm is not None: return _llm
    if Llama is None: return None
    mp=os.getenv("LLM_MODEL_PATH","models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf")
    full=PROJECT_ROOT/mp
    if not full.exists():
        for p in (PROJECT_ROOT/"models").glob("*.gguf"): full=p; break
    try: _llm=Llama(model_path=str(full),n_ctx=4096,n_threads=8,verbose=False)
    except Exception as e: print(f"[BRAIN] {e}"); _llm=None
    return _llm
def detect_tool(text):
    t=text.lower()
    if any(w in t for w in ["weather","forecast","temperature"]):
        m=re.search(r"in ([A-Za-z\s]+)",text,re.I); city=m.group(1).strip() if m else "Calabar"
        return "get_weather",{"city":city}
    if any(w in t for w in ["time","clock","date"]): return "get_time",{}
    if "calc" in t or re.search(r"\d+\s*[\+\-\*/]",t):
        m=re.search(r"([0-9+\-*/().%\s]+)",text); return "calc",{"expression":m.group(1) if m else text}
    return None
def execute_tool(name,args):
    if name=="get_weather": return get_weather(args.get("city","Calabar"))
    if name=="get_time": return get_time()
    if name=="calc":
        try:
            allowed={ast.Add:operator.add,ast.Sub:operator.sub,ast.Mult:operator.mul,ast.Div:operator.truediv,ast.Pow:operator.pow,ast.Mod:operator.mod}
            def ev(n):
                if isinstance(n,ast.Constant): return n.value
                if isinstance(n,ast.BinOp): return allowed[type(n.op)](ev(n.left),ev(n.right))
                if isinstance(n,ast.UnaryOp): return -ev(n.operand)
                raise ValueError("unsafe")
            return str(ev(ast.parse(args.get("expression",""),mode='eval').body))
        except Exception as e: return f"calc error {e}"
    if name=="get_news": return get_news(args.get("topic","AI"))
    return f"Unknown {name}"
def chat(user_id,message):
    mem=""
    try:
        h=search_memory(message,3)
        if h: mem="[MEMORY]\n"+"\n".join(h)+"\n"
    except: pass
    forced=detect_tool(message)
    if forced:
        n,a=forced; r=execute_tool(n,a); save_memory(f"User:{message}|{n}->{r}","episodic"); return r
    llm=get_llm()
    if llm is None: return f"SALLY offline tools only: try 'weather in Calabar' or 'time' or 'calc 25*40'"
    user=load_user(); sys=f"You are SALLY for {user.get('user_name','Edima')} in Calabar. Tools: get_weather(city), get_time(), calc(expr)."
    prompt=f"<|im_start|>system\n{sys}\n{mem}<|im_end|>\n<|im_start|>user\n{message}<|im_end|>\n<|im_start|>assistant\n"
    try:
        out=llm.create_completion(prompt,max_tokens=512,temperature=0.7,stop=["<|im_end|>"])
        text=out["choices"][0]["text"].strip()
        text=re.sub(r"JARVIS","SALLY",text,flags=re.I)
        save_memory(f"User:{message}|SALLY:{text[:500]}","episodic")
        return text
    except Exception as e: return f"Error {e}"
