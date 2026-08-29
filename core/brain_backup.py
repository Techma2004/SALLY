"""
core/brain.py v2 — Hermes-style real function calling
Replaces TRIGGERS hack with JSON tool parser + tool loop
"""
import os
import json
import re
from pathlib import Path

# Lazy imports to avoid crash if models missing
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None

from .config import load_config, load_facts
from .memory import search_memory, load_user, load_memory, save_memory, log_daily, should_nudge
from .tools import TOOLS, execute_tool, get_tool_descriptions_for_prompt

PROJECT_ROOT = Path(__file__).parent.parent

# --- Hermes ChatML Prompt ---
PERSONALITY = """You are SALLY — Science Artificial Learning Logic and You.
You are an offline-first, private, helpful assistant built by Edima Bassey in Calabar.
You NEVER say you are JARVIS, Meta AI, or any other assistant. You are SALLY.
You are concise, warm, and helpful. You remember user facts across sessions.
You have tools — use them when needed, don't hallucinate weather/news/time.
If you need to save something important, use remember_fact tool.
"""

def build_system_prompt():
    """Hermes 3-layer system prompt"""
    user_ctx = load_user()[:1000]
    mem_ctx = load_memory()[:1000]

    tool_desc = get_tool_descriptions_for_prompt()

    sys = f"""{PERSONALITY}

[USER CONTEXT]
{user_ctx}

[MEMORY CONTEXT - FTS5 recent]
{mem_ctx}

[TOOLS]
{tool_desc}

RULES:
- If user asks weather/news/time/calc, you MUST use tool, don't invent.
- If user says "remember" or tells you personal fact, use remember_fact.
- If user asks "who am i" or "recall", use recall_memory.
- After tool output, answer naturally.
- Keep answers short unless asked for detail.
"""
    return sys

# Global LLM instance
_llm = None
_turn_count = 0

def get_llm():
    global _llm
    if _llm is not None:
        return _llm
    if Llama is None:
        print("[BRAIN] llama_cpp not installed")
        return None

    cfg = load_config()
    model_path = cfg.get("model_path") or os.getenv("LLM_MODEL_PATH")

    # Safety: reject mmproj (vision projection) — Hermes fix from Phase 0
    if model_path and "mmproj" in str(model_path).lower():
        print(f"[BRAIN] ERROR: {model_path} is mmproj (vision), not LLM. Check settings.json")
        return None

    if not model_path or not Path(model_path).exists():
        print(f"[BRAIN] Model not found: {model_path}")
        return None

    print(f"[BRAIN] Loading LLM: {model_path}")
    _llm = Llama(
        model_path=str(model_path),
        n_ctx=cfg.get("n_ctx", 4096),
        n_threads=cfg.get("n_threads", 8),
        verbose=False
    )
    return _llm

def parse_tool_call(text: str):
    """
    Hermes parser — handles multiple formats:
    1. {"tool": "name", "args": {...}}  <- our Hermes format
    2. {"name": "name", "arguments": {...}} <- OpenAI
    3. <tool_call>{"name":...}</tool_call> <- Hermes Agent format
    """
    text = text.strip()

    # Try to find JSON block
    # Pattern 1: <tool_call>...</tool_call>
    m = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
    if m:
        text = m.group(1)

    # Find all JSON objects
    json_blocks = re.findall(r"\{[^{}]*\"(?:tool|name)\"[^{}]*\{[^{}]*\}[^{}]*\}|{[^{}]*\"(?:tool|name)\"[^{}]*\}", text, re.DOTALL)
    # Better: try to parse whole text as JSON
    candidates = [text]
    # Also try to extract first {...} block
    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for cand in candidates:
        try:
            data = json.loads(cand)
            # Format 1: {"tool": "weather", "args": {...}}
            if "tool" in data and "args" in data:
                return data["tool"], data["args"]
            if "tool" in data and "arguments" in data:
                return data["tool"], data["arguments"]
            # Format 2: {"name": "weather", "arguments": {...}}
            if "name" in data:
                args = data.get("arguments") or data.get("args") or {}
                if isinstance(args, str):
                    args = json.loads(args)
                return data["name"], args
            # Format 3: OpenAI function call wrapper
            if "function" in data:
                fn = data["function"]
                return fn["name"], json.loads(fn.get("arguments", "{}")) if isinstance(fn.get("arguments"), str) else fn.get("arguments", {})
        except:
            continue

    return None, None

def filter_identity(text: str) -> str:
    """JARVIS -> SALLY filter"""
    # Prevent identity leak
    replacements = [
        ("I am JARVIS", "I am SALLY"),
        ("I'm JARVIS", "I'm SALLY"),
        ("JARVIS", "SALLY"),
        ("I am Meta AI", "I am SALLY"),
        ("I'm Meta AI", "I'm SALLY"),
    ]
    for old, new in replacements:
        text = text.replace(old, new)
    return text

def chat(user_input: str, history: list = None) -> str:
    """Hermes tool loop: LLM -> tool -> LLM -> answer"""
    global _turn_count
    _turn_count += 1

    llm = get_llm()
    if llm is None:
        # Fallback without LLM — still run tools
        # Try direct tool detection for offline testing
        low = user_input.lower()
        if "weather" in low:
            city = "Calabar"
            m = re.search(r"weather in (\w+)", low)
            if m: city = m.group(1)
            return execute_tool("get_weather", {"city": city})
        if "time" in low:
            return execute_tool("get_time", {})
        return f"[SALLY offline] LLM not loaded. But I can still run tools. You said: {user_input}"

    sys_prompt = build_system_prompt()
    history = history or []

    # Build ChatML messages
    messages = [{"role": "system", "content": sys_prompt}]
    for h in history[-6:]:  # last 6 turns
        messages.append(h)
    messages.append({"role": "user", "content": user_input})

    # First LLM call — may output tool call
    try:
        resp = llm.create_chat_completion(
            messages=messages,
            temperature=0.7,
            max_tokens=512,
            stop=["<|im_end|>", "</tool_call>"]
        )
        assistant_text = resp["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"[BRAIN ERROR] {e}"

    # Check if it's a tool call
    tool_name, tool_args = parse_tool_call(assistant_text)

    if tool_name:
        print(f"[BRAIN] Tool call: {tool_name} {tool_args}")
        tool_output = execute_tool(tool_name, tool_args or {})
        print(f"[BRAIN] Tool output: {tool_output[:200]}")

        # Log tool use for learning loop (Phase 4)
        log_daily(f"Used tool {tool_name} with {tool_args} -> {tool_output[:100]}")

        # Second LLM call — feed tool output back
        messages.append({"role": "assistant", "content": assistant_text})
        messages.append({"role": "user", "content": f"Tool {tool_name} returned: {tool_output}\nNow answer the user naturally using this result. Don't mention tool JSON."})

        try:
            resp2 = llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                max_tokens=512
            )
            final = resp2["choices"][0]["message"]["content"].strip()
        except Exception as e:
            final = tool_output  # fallback to raw tool output

        final = filter_identity(final)

        # Nudge check — Hermes periodic save
        if should_nudge(_turn_count):
            final += "\n\n[Memory nudge: Should I save something from this chat?]"

        return final
    else:
        # No tool, direct answer
        assistant_text = filter_identity(assistant_text)
        log_daily(f"User: {user_input[:100]} -> SALLY: {assistant_text[:100]}")
        return assistant_text

# Backward compat for old main.py that calls brain.respond()
def respond(prompt: str, history=None):
    return chat(prompt, history)

# For testing
if __name__ == "__main__":
    print(build_system_prompt())
    print("\n--- Test tool parser ---")
    tests = [
        '{"tool": "get_weather", "args": {"city": "Calabar"}}',
        '{"name": "get_weather", "arguments": {"city": "Lagos"}}',
        '<tool_call>{"name": "get_time", "arguments": {}}</tool_call>'
    ]
    for t in tests:
        print(t, "->", parse_tool_call(t))

