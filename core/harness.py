"""
core/harness.py — DeepSeek-style tool parser (Plan B)
Borrowed logic from DeepSeek-V3 function calling harness:
- Extract JSON from <tool_call>, ```json, or raw {tool:..}
- Validate against known tools
- Forced intent fallback via keywords (Qwen 1.5B fix)
"""
import json, re

TOOL_KEYWORDS = {
    "get_weather": ["weather", "temperature", "forecast", "rain", "humidity", "calabar", "lagos"],
    "get_time": ["time", "date", "clock", "what.*day"],
    "get_news": ["news", "headline", "nigeria", "fg", "eagles"],
    "calc": ["calculate", "math", "what is", r"\d+\s*[\+\-\*/]"],
    "recall_memory": ["remember", "recall", "memory", "what.*my"],
    "save_memory": ["save this", "remember that", "my name is"],
    "search_memory": ["search", "find in memory"],
}

KNOWN_TOOLS = set(TOOL_KEYWORDS.keys()) | {"read_file","write_file","list_files","exit"}

def extract_json_blocks(text: str):
    """DeepSeek harness: try multiple formats"""
    candidates = []

    # 1. <tool_call>{"name":...}</tool_call> or <tool_call>...</tool_call>
    for m in re.finditer(r'<tool_call>(.*?)</tool_call>', text, re.DOTALL):
        candidates.append(m.group(1).strip())

    # 2. ```json {...} ```
    for m in re.finditer(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL):
        candidates.append(m.group(1).strip())

    # 3. Raw JSON objects that look like tool calls
    for m in re.finditer(r'\{\s*"tool"\s*:\s*"[^"]+".*?\}', text, re.DOTALL):
        candidates.append(m.group(0))
    for m in re.finditer(r'\{\s*"name"\s*:\s*"[^"]+".*?\}', text, re.DOTALL):
        candidates.append(m.group(0))

    # 4. The whole text if it starts with {
    stripped = text.strip()
    if stripped.startswith('{') and stripped.endswith('}'):
        candidates.append(stripped)

    return candidates

def normalize_tool_call(obj: dict):
    """Normalize DeepSeek / OpenAI / Qwen formats to {tool, args}"""
    if not isinstance(obj, dict):
        return None

    # Format 1: {"tool": "get_weather", "args": {"city":"Calabar"}}
    if "tool" in obj:
        tool = obj["tool"]
        args = obj.get("args") or obj.get("arguments") or obj.get("parameters") or {}
        return {"tool": tool, "args": args}

    # Format 2: {"name": "get_weather", "arguments": {...}}
    if "name" in obj:
        tool = obj["name"]
        args = obj.get("arguments") or obj.get("args") or {}
        if isinstance(args, str):
            try: args = json.loads(args)
            except: args = {}
        return {"tool": tool, "args": args}

    return None

def forced_intent(text: str):
    """Fallback: keyword -> tool (for Qwen 1.5B)"""
    t = text.lower()
    for tool, kws in TOOL_KEYWORDS.items():
        for kw in kws:
            if re.search(kw, t):
                # extract city for weather
                if tool == "get_weather":
                    city = "Calabar"
                    m = re.search(r'in (\w+)', t)
                    if m: city = m.group(1).capitalize()
                    return {"tool": tool, "args": {"city": city}}
                if tool == "calc":
                    # extract expression
                    m = re.search(r'([0-9\+\-\*/\.\(\) ]+)', text)
                    if m:
                        return {"tool": tool, "args": {"expression": m.group(1).strip()}}
                return {"tool": tool, "args": {}}
    return None

def parse_tool_calls(llm_output: str):
    """
    Main entry — DeepSeek harness style
    Returns list of {"tool":..., "args":...} or []
    """
    if not llm_output:
        return []

    # Try extract JSON blocks
    for cand in extract_json_blocks(llm_output):
        try:
            obj = json.loads(cand)
            # Could be {"tool_calls": [..]}
            if "tool_calls" in obj:
                results = []
                for tc in obj["tool_calls"]:
                    norm = normalize_tool_call(tc)
                    if norm and norm["tool"] in KNOWN_TOOLS:
                        results.append(norm)
                if results:
                    return results

            norm = normalize_tool_call(obj)
            if norm and norm["tool"] in KNOWN_TOOLS:
                return [norm]

            # Direct: {"get_weather": {"city":..}} style
            for k,v in obj.items():
                if k in KNOWN_TOOLS:
                    return [{"tool": k, "args": v if isinstance(v, dict) else {}}]

        except:
            continue

    # Fallback forced
    forced = forced_intent(llm_output)
    if forced:
        return [forced]

    return []

# Test
if __name__ == "__main__":
    tests = [
        '<tool_call>{"tool": "get_weather", "args": {"city":"Calabar"}}</tool_call>',
        '```json\n{"name":"calc","arguments":{"expression":"2+2"}}\n```',
        'I need weather in Lagos { "tool": "get_weather", "args": {"city":"Lagos"}}',
        'what time is it?',
    ]
    for t in tests:
        print(t, "=>", parse_tool_calls(t))
