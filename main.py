from core.brain import think
from core.voice import speak
from core.memory import load_memory, save_memory

print("--- SALLY v0.1 Online on Debian ---")
print("Type 'exit' to quit")

history = load_memory()

while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ['exit', 'quit', 'bye']:
        speak("Goodbye. SALLY going offline.")
        break

    save_memory("user", user_input)
    # convert memory to ollama format
    chat_history = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]

    response = think(user_input, chat_history)

    save_memory("assistant", response)
    history.append({"role": "user", "content": user_input})
    history.append({"role": "assistant", "content": response})

    speak(response)
