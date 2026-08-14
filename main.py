from core.brain import think
from core.voice import speak, listen
from core.memory import load_memory, save_memory

print("--- SALLY v0.3 Voice (no wake word) ---")
print("Commands: [Enter]=voice input, type=text input, 'exit'=quit")
history = load_memory()

while True:
    try:
        mode = input("\nPress Enter for VOICE or type your message: ")

        if mode.lower() in ['exit', 'quit', 'bye']:
            speak("Goodbye. SALLY going offline.")
            break

        if mode.strip() == "":
            # VOICE MODE
            user_input = listen(duration=6)
            if not user_input:
                print("Didn't catch that.")
                continue
        else:
            # TEXT MODE
            user_input = mode

        save_memory("user", user_input)
        chat_history = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]

        response = think(user_input, chat_history)

        save_memory("assistant", response)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        speak(response)

    except KeyboardInterrupt:
        speak("Going offline")
        break
