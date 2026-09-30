from __future__ import annotations

from typing import Any


def machine_status() -> dict[str, Any]:
    """Return a live snapshot of the machine's current state."""
    import sally_native

    raw = sally_native.system_status()

    memory_total = int(raw["memory_total_bytes"])
    memory_used = int(raw["memory_used_bytes"])

    battery_raw = raw["battery"]

    battery: dict[str, Any] = {
        "available": bool(battery_raw["available"]),
    }

    if battery["available"]:
        battery["percent"] = round(float(battery_raw["percent"]))
        battery["status"] = str(battery_raw["status"])
        battery["remaining_minutes"] = battery_raw["remaining_minutes"]

    return {
        "time": str(raw["time"]),
        "platform": str(raw["platform"]),
        "architecture": str(raw["architecture"]),
        "cpu": {
            "cores": int(raw["cpu_cores"]),
        },
        "memory": {
            "total_mb": round(memory_total / (1024 * 1024)),
            "used_mb": round(memory_used / (1024 * 1024)),
        },
        "battery": battery,
        "uptime_seconds": int(raw["uptime_seconds"]),
    }
