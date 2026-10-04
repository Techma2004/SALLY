"""User-facing help: what SALLY can do and how to ask for it.

Every example here is checked by tests/test_help.py: tool examples must route
to their tool and run, agent examples must route to their agent. The help can
therefore never advertise something SALLY does not do.
"""

from __future__ import annotations

TOOL_EXAMPLES: dict[str, tuple[str, ...]] = {
    "calculator": (
        "Calculate 25 * 40",
        "What is 144 divided by 12?",
    ),
    "datetime": (
        "What time is it?",
        "What is today's date?",
    ),
    "machine_status": (
        "How is my machine doing?",
        "How much RAM am I using?",
        "Disk space left",
    ),
    "unit_convert": (
        "Convert 5 miles to kilometers",
        "How many meters are in 3 miles?",
    ),
    "scientific_constant": (
        "What is the speed of light?",
        "Planck constant",
    ),
    "science_calculate": (
        "sqrt(144)",
        "sin(0.5)",
    ),
}

AGENT_GUIDE: tuple[dict[str, object], ...] = (
    {
        "name": "planning",
        "title": "Planning",
        "description": "Break a goal into steps or a roadmap.",
        "examples": ("Plan SALLY's next milestone",),
    },
    {
        "name": "coding",
        "title": "Coding",
        "description": "Help with code, bugs, and scripts.",
        "examples": ("Write a Python function that reverses a string",),
    },
    {
        "name": "testing",
        "title": "Testing and review",
        "description": "Review work and look for edge cases.",
        "examples": ("Review the edge cases of a login form",),
    },
    {
        "name": "research",
        "title": "Research",
        "description": "Compare options and weigh trade-offs.",
        "examples": ("Compare SQLite and PostgreSQL",),
    },
)

MEMORY_GUIDE: dict[str, object] = {
    "description": (
        "Tell SALLY about yourself in plain sentences and she remembers it. "
        "She only saves what you say outright, never guesses, and you can "
        "delete anything on the Memory page."
    ),
    # Safe to click: these only read memory.
    "examples": ("What's my name?", "What do you know about me?"),
    # Templates to say in your own words (clicking them would overwrite facts).
    "say": (
        "My name is …",
        "I live in …",
        "I work as …",
        "I love …",
        "Remember that my sister's birthday is May 3",
    ),
}

TIPS: tuple[str, ...] = (
    "No commands needed: just write the way you would to a person. SALLY "
    "recognises calculations, times, conversions, machine questions and "
    "things worth remembering on her own.",
    "Tools give exact answers instantly. Agents take longer because they "
    "work in several steps.",
    "Press Enter to send and Shift + Enter for a new line. The stop button "
    "ends a reply early.",
    "Type / in the message box for quick commands.",
    "Your conversations and memories are stored on this machine only. "
    "Delete any of them from the History drawer or the Memory page.",
    "SALLY runs a small local model. It can make mistakes in free "
    "conversation, so use tools for numbers, units, time and machine facts.",
)


def help_topics(registry) -> dict[str, object]:
    """Assemble the help page from the live tool registry."""
    tools = []

    for name in registry.names():
        tools.append(
            {
                "name": name,
                "description": registry.metadata(name).description,
                "examples": list(TOOL_EXAMPLES.get(name, ())),
            }
        )

    return {
        "memory": {
            **MEMORY_GUIDE,
            "examples": list(MEMORY_GUIDE["examples"]),
            "say": list(MEMORY_GUIDE["say"]),
        },
        "tools": tools,
        "agents": [
            {**agent, "examples": list(agent["examples"])}
            for agent in AGENT_GUIDE
        ],
        "tips": list(TIPS),
    }
