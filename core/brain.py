import time

_llm = None

def get_llm():
    global _llm
    if _llm is not None:
        return _llm
    from llama_cpp import Llama
    from core.config import LLM_MODEL_PATH, LLM_N_CTX, LLM_N_THREADS

    print(f"[SALLY] Loading {LLM_MODEL_PATH.name}...")
    t0 = time.time()
    _llm = Llama(
        model_path=str(LLM_MODEL_PATH),
        n_ctx=LLM_N_CTX,
        n_threads=LLM_N_THREADS,
        use_mmap=True,
        use_mlock=False,
        verbose=False
    )
    print(f"[SALLY] Ready in {time.time()-t0:.1f}s")
    return _llm

def think(prompt, history, tools):
    from core.config import PERSONALITY, LLM_TEMPERATURE
    from core.memory import get_context

    mem = get_context()
    low = prompt.lower()

    # --- Hermes: check tools first ---
    tool_outputs = []
    if "weather" in low or "temperature" in low:
        if "weather" in tools:
            tool_outputs.append(tools["weather"](prompt))
    if "news" in low or "headline" in low:
        if "news" in tools:
            tool_outputs.append(tools["news"](prompt))

    # If it's a weather/news query, return tool result DIRECTLY — no LLM hallucination
    if tool_outputs:
        # If tool says no API key, show it clearly
        return "\n".join(tool_outputs)

    llm = get_llm()
    msgs = [
        {"role": "system", "content": f"{PERSONALITY}\n[MEMORY]\n{mem}"},
        *history[-4:],
        {"role": "user", "content": prompt}
    ]
    out = llm.create_chat_completion(messages=msgs, temperature=LLM_TEMPERATURE, max_tokens=350)
    ans = out["choices"][0]["message"]["content"].strip()
    if ans.lower().startswith("sally:"):
        ans = ans[6:].strip()
    return ans
