from llama_cpp import Llama
from.config import PERSONALITY, LLM_MODEL_PATH, LLM_N_CTX, LLM_N_THREADS, LLM_TEMPERATURE
from.tools import AVAILABLE_TOOLS
import re

print(f"[SALLY] Loading {LLM_MODEL_PATH}...")
llm = Llama(
    model_path=LLM_MODEL_PATH,
    n_ctx=LLM_N_CTX,
    n_threads=LLM_N_THREADS,
    verbose=False
)
print("[SALLY] Ready.")

def detect_intent(prompt: str):
    prompt_lower = prompt.lower()
    if any(k in prompt_lower for k in ["weather", "temperature", "forecast"]):
        city_match = re.search(r"in ([a-zA-Z ]+)", prompt_lower)
        city = city_match.group(1).strip() if city_match else None
        return "get_weather", {"city": city}
    if any(k in prompt_lower for k in ["news", "headlines"]):
        topic_match = re.search(r"news (?:about|on|for)? ([a-zA-Z ]+)", prompt_lower)
        topic = topic_match.group(1).strip() if topic_match else "technology"
        return "get_news", {"topic": topic, "count": 5}
    return None, {}

def think(prompt, history=[]):
    tool_name, tool_args = detect_intent(prompt)
    tool_context = ""

    if tool_name and tool_name in AVAILABLE_TOOLS:
        print(f"[SALLY] Tool -> {tool_name} {tool_args}")
        func = AVAILABLE_TOOLS[tool_name]
        clean_args = {k: v for k, v in tool_args.items() if v}
        try:
            result = func(**clean_args)
            tool_context = f"\nSYSTEM: LIVE DATA FROM TOOL {tool_name}: {result}\nYou MUST use this data in your answer. Do not say you can't access internet."
        except Exception as e:
            tool_context = f"\n[TOOL ERROR]: {e}"

    full_prompt = prompt + tool_context

    messages = [{"role": "system", "content": PERSONALITY}]
    messages.extend(history)
    messages.append({"role": "user", "content": full_prompt})

    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=LLM_TEMPERATURE,
        stop=["<|eot_id|>", "<|im_end|>"]
    )

    return output["choices"][0]["message"]["content"]
