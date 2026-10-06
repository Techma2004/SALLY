from __future__ import annotations

from typing import Any

# Pseudo or read-only filesystems that say nothing about free space.
_SKIP_FILESYSTEMS = {"squashfs", "overlay", "tmpfs", "devtmpfs", "ramfs", "iso9660"}

_MB = 1024 * 1024
_GB = 1024 * _MB


def cpu_percent(sample_ms: int = 250) -> float:
    """Overall CPU usage in percent, sampled over a short window."""
    import sally_native

    return round(float(sally_native.cpu_percent(sample_ms)), 1)


def disk_status(limit: int = 4) -> list[dict[str, Any]]:
    """Real mounted filesystems, root first, then largest."""
    import sally_native

    seen: set[str] = set()
    disks: list[dict[str, Any]] = []

    for raw in sally_native.disk_status():
        mount = str(raw["mount"])
        filesystem = str(raw["filesystem"])
        total = int(raw["total_bytes"])

        if filesystem in _SKIP_FILESYSTEMS or mount in seen or total <= 0:
            continue

        seen.add(mount)
        used = max(0, total - int(raw["available_bytes"]))

        disks.append(
            {
                "mount": mount,
                "filesystem": filesystem,
                "total_gb": round(total / _GB, 1),
                "used_gb": round(used / _GB, 1),
                "percent": round(100 * used / total),
            }
        )

    disks.sort(key=lambda disk: (disk["mount"] != "/", -disk["total_gb"]))

    return disks[: max(1, limit)]


def network_status() -> dict[str, Any]:
    """Traffic totals since boot (loopback excluded)."""
    import sally_native

    raw = sally_native.network_status()

    return {
        "received_mb": round(int(raw["received_bytes"]) / _MB, 1),
        "transmitted_mb": round(int(raw["transmitted_bytes"]) / _MB, 1),
        "interfaces": sorted(
            str(item["name"]) for item in raw["interfaces"] if item["name"] != "lo"
        ),
    }


def top_processes(limit: int = 5) -> list[dict[str, Any]]:
    """The processes using the most memory."""
    import sally_native

    return [
        {
            "pid": int(raw["pid"]),
            "name": str(raw["name"]),
            "memory_mb": round(int(raw["memory_bytes"]) / _MB, 1),
        }
        for raw in sally_native.top_processes(limit)
    ]


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
            "percent": cpu_percent(),
        },
        "memory": {
            "total_mb": round(memory_total / (1024 * 1024)),
            "used_mb": round(memory_used / (1024 * 1024)),
        },
        "disks": disk_status(),
        "network": network_status(),
        "battery": battery,
        "uptime_seconds": int(raw["uptime_seconds"]),
    }


def _duration(seconds: int) -> str:
    days, rest = divmod(max(0, seconds), 86400)
    hours, rest = divmod(rest, 3600)
    minutes = rest // 60

    parts = []
    if days:
        parts.append(f"{days} d")
    if hours:
        parts.append(f"{hours} h")
    parts.append(f"{minutes} min")

    return " ".join(parts)


def _describe_focus(status: dict[str, Any], focus: str) -> str | None:
    memory = status["memory"]

    if focus == "memory":
        percent = round(100 * memory["used_mb"] / memory["total_mb"])
        return (
            f"You're using {memory['used_mb']} of {memory['total_mb']} MB "
            f"of RAM ({percent}%)."
        )

    if focus == "cpu":
        return (
            f"The CPU is about {status['cpu']['percent']}% busy across "
            f"{status['cpu']['cores']} cores."
        )

    if focus == "disk" and status.get("disks"):
        disk = status["disks"][0]
        free = round(disk["total_gb"] - disk["used_gb"], 1)
        return (
            f"You have {free} GB free on {disk['mount']} "
            f"({disk['percent']}% of {disk['total_gb']} GB used)."
        )

    if focus == "battery":
        battery = status["battery"]
        if not battery["available"]:
            return "This machine doesn't report a battery."
        return f"Battery is at {battery['percent']}% ({battery['status']})."

    if focus == "uptime":
        return f"This machine has been up for {_duration(status['uptime_seconds'])}."

    if focus == "network" and status.get("network"):
        network = status["network"]
        return (
            f"Since boot: {network['received_mb']} MB received and "
            f"{network['transmitted_mb']} MB sent."
        )

    return None


def describe_machine(status: dict[str, Any], focus: str | None = None) -> str:
    """Turn machine_status() into a short, accurate, human-readable report.

    With a focus ("disk", "memory", ...) only that reading is described.
    """
    if focus:
        focused = _describe_focus(status, focus)

        if focused:
            return focused

    memory = status["memory"]
    percent = round(100 * memory["used_mb"] / memory["total_mb"])

    lines = [
        f"Machine status at {status['time']} "
        f"({status['platform']} {status['architecture']})",
        f"CPU: {status['cpu']['percent']}% busy across "
        f"{status['cpu']['cores']} cores",
        f"Memory: {memory['used_mb']} / {memory['total_mb']} MB ({percent}%)",
    ]

    for disk in status.get("disks", []):
        free = round(disk["total_gb"] - disk["used_gb"], 1)
        lines.append(
            f"Disk {disk['mount']}: {disk['percent']}% of "
            f"{disk['total_gb']} GB used ({free} GB free)"
        )

    network = status.get("network")
    if network:
        lines.append(
            f"Network since boot: {network['received_mb']} MB received, "
            f"{network['transmitted_mb']} MB sent"
        )

    battery = status["battery"]
    if battery["available"]:
        remaining = battery.get("remaining_minutes")
        extra = f", about {remaining} min left" if remaining else ""
        lines.append(
            f"Battery: {battery['percent']}% ({battery['status']}{extra})"
        )
    else:
        lines.append("Battery: not available on this machine")

    lines.append(f"Uptime: {_duration(status['uptime_seconds'])}")

    return "\n".join(lines)
