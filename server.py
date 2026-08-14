from fastapi import FastAPI
from core.brain import think, llm
from core.memory import load_memory, save_memory

app = FastAPI(title="SALLY Bridge")

@app.get("/")
def status():
    return {"status": "SALLY online", "engine": "llama.cpp", "model": llm.model_path}

@app.post("/chat")
def chat(prompt: str):
    history = load_memory()
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]
    response = think(prompt, chat_history)
    save_memory("user", prompt)
    save_memory("assistant", response)
    return {"sally": response}
