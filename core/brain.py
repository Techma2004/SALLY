from llama_cpp import Llama
from.config import PERSONALITY
import os

# Path to your model
MODEL_PATH = os.path.expanduser("/home/edima/programming/projects/SALLY/models/Qwen2.5-Coder-1.5B-Instruct-Q3_K_L.gguf")

# Load model once at startup
# n_ctx = context window, n_threads = use your CPU cores
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=4096,
    n_threads=8, # change to your CPU cores
    verbose=False
)

def think(prompt, history=[]):
    # Build chat format that llama.cpp understands
    messages = [{"role": "system", "content": PERSONALITY}]
    messages.extend(history)
    messages.append({"role": "user", "content": prompt})

    # llama.cpp chat completion
    output = llm.create_chat_completion(
        messages=messages,
        max_tokens=512,
        temperature=0.7,
        stop=["<|eot_id|>", "You:"]
    )

    return output["choices"][0]["message"]["content"]
