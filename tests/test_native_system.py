import sys
from types import SimpleNamespace

import pytest

from core import native

GB = 1024**3
MB = 1024**2


def _fake_native():
    return SimpleNamespace(
        cpu_percent=lambda sample_ms=250: 12.345,
        disk_status=lambda: [
            {"mount": "/data", "filesystem": "ext4",
             "total_bytes": 500 * GB, "available_bytes": 100 * GB},
            {"mount": "/", "filesystem": "ext4",
             "total_bytes": 100 * GB, "available_bytes": 25 * GB},
            {"mount": "/snap/core", "filesystem": "squashfs",
             "total_bytes": 1 * GB, "available_bytes": 0},
            {"mount": "/", "filesystem": "ext4",  # duplicate mount
             "total_bytes": 100 * GB, "available_bytes": 25 * GB},
        ],
        network_status=lambda: {
            "received_bytes": 5 * MB,
            "transmitted_bytes": 2 * MB,
            "interfaces": [
                {"name": "wlan0", "received_bytes": 5 * MB, "transmitted_bytes": 2 * MB},
                {"name": "lo", "received_bytes": 9, "transmitted_bytes": 9},
            ],
        },
        top_processes=lambda limit=5: [
            {"pid": 7, "name": "llama", "memory_bytes": 900 * MB},
        ][:limit],
        system_status=lambda: {
            "time": "10:00:00", "platform": "linux", "architecture": "x86_64",
            "cpu_cores": 4, "memory_total_bytes": 4 * GB,
            "memory_used_bytes": 2 * GB, "uptime_seconds": 60,
            "battery": {"available": False},
        },
    )


@pytest.fixture
def fake(monkeypatch):
    module = _fake_native()
    monkeypatch.setitem(sys.modules, "sally_native", module)
    return module


def test_disk_status_filters_dedupes_and_puts_root_first(fake):
    disks = native.disk_status()

    assert [d["mount"] for d in disks] == ["/", "/data"]
    assert disks[0] == {
        "mount": "/", "filesystem": "ext4",
        "total_gb": 100.0, "used_gb": 75.0, "percent": 75,
    }


def test_network_status_converts_units_and_hides_loopback(fake):
    assert native.network_status() == {
        "received_mb": 5.0, "transmitted_mb": 2.0, "interfaces": ["wlan0"],
    }


def test_top_processes_and_cpu_percent(fake):
    assert native.top_processes(1) == [
        {"pid": 7, "name": "llama", "memory_mb": 900.0}
    ]
    assert native.cpu_percent() == 12.3


def test_machine_status_includes_new_readings(fake):
    status = native.machine_status()

    assert status["cpu"] == {"cores": 4, "percent": 12.3}
    assert status["disks"][0]["mount"] == "/"
    assert status["network"]["interfaces"] == ["wlan0"]
    assert status["memory"] == {"total_mb": 4096, "used_mb": 2048}


def test_real_native_module_returns_sane_values(monkeypatch):
    real = pytest.importorskip("sally_native")
    monkeypatch.setitem(sys.modules, "sally_native", real)

    disks = native.disk_status()
    assert disks and all(0 <= d["percent"] <= 100 for d in disks)

    assert 0 <= native.cpu_percent() <= 100
    assert native.network_status()["received_mb"] >= 0
    assert all(p["memory_mb"] >= 0 for p in native.top_processes(3))
