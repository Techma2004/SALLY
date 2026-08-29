import importlib.util, pathlib
SKILLS_DIR = pathlib.Path("memory/skills")
SKILLS_DIR.mkdir(exist_ok=True)

def load_skills():
    tools = {}
    for py in SKILLS_DIR.glob("*.py"):
        try:
            spec = importlib.util.spec_from_file_location(py.stem, py)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            if hasattr(mod, "run"):
                tools[py.stem] = mod.run
                print(f"[SKILL] loaded {py.stem}")
        except Exception as e:
            print(f"[SKILL] fail {py.stem}: {e}")
    return tools
