from core.brain import think
from core.voice import speak, listen
from core.memory import load_memory, save_memory, recall, init_core_from_facts, remember

print("--- SALLY v0.40 Memory + Voice ---")
print("Commands: [Enter]=voice, type=text, 'recall X'=search memory, 'exit'=quit")

# Migrate facts on first boot
init_core_from_facts()
history = load_memory()

while True:
    try:
        mode = input("\nPress Enter for VOICE or type your message: ").strip()

        if mode.lower() in ['exit', 'quit', 'bye']:
            speak("Goodbye. SALLY going offline.")
            break

        # Special: recall command
        if mode.lower().startswith("recall "):
            q = mode[7:]
            hits = recall(q, k=5)
            if not hits:
                print("No memories found.")
            else:
                for h in hits:
                    print(f"[{h['type']}] {h['timestamp'][:16]}: {h['content'][:120]}")
            continue

        if mode == "":
            user_input = listen(duration=6)
            if not user_input:
                print("Didn't catch that.")
                continue
        else:
            user_input = mode

        save_memory("user", user_input)

        # --- MEMORY-AUGMENTED THINKING ---
        # 1. Search relevant memories
        mem_hits = recall(user_input, k=3)
        mem_context = "\n".join([f"[{h['type']}] {h['content']}" for h in mem_hits]) if mem_hits else ""

        # 2. Build prompt with memory
        chat_history = [{"role": m["role"], "content": m["content"]} for m in history[-10:]]

        # If we have memory hits, prepend them as system context
        if mem_context:
            augmented_input = f"[MEMORY CONTEXT]\n{mem_context}\n\n[USER]\n{user_input}"
        else:
            augmented_input = user_input

        response = think(augmented_input, chat_history)

        save_memory("assistant", response)
        history.append({"role": "user", "content": user_input})
        history.append({"role": "assistant", "content": response})

        print(f"\nSALLY: {response}\n")
        speak(response)

    except KeyboardInterrupt:
        speak("Going offline")
        break
    except Exception as e:
        print(f"Error: {e}")
        import traceback; traceback.print_exc()
