use battery::Manager;
use chrono::Local;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sysinfo::{
    CpuRefreshKind, Disks, MemoryRefreshKind, Networks, ProcessesToUpdate, RefreshKind, System,
    MINIMUM_CPU_UPDATE_INTERVAL,
};
use std::time::Duration;

#[pyfunction]
fn platform() -> &'static str {
    std::env::consts::OS
}

#[pyfunction]
fn architecture() -> &'static str {
    std::env::consts::ARCH
}

#[pyfunction]
fn current_time() -> String {
    Local::now().format("%H:%M:%S").to_string()
}

#[pyfunction]
fn system_info() -> PyResult<(u64, u64, u64, u64)> {
    let mut system = System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything()),
    );

    system.refresh_cpu_all();
    system.refresh_memory();

    let cpu_count = system.cpus().len() as u64;
    let total_memory = system.total_memory();
    let used_memory = system.used_memory();
    let uptime = System::uptime();

    Ok((cpu_count, total_memory, used_memory, uptime))
}

#[pyfunction]
fn system_status(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let mut system = System::new_with_specifics(
        RefreshKind::nothing()
            .with_cpu(CpuRefreshKind::everything())
            .with_memory(MemoryRefreshKind::everything()),
    );

    system.refresh_cpu_all();
    system.refresh_memory();

    let status = PyDict::new(py);
    status.set_item("time", current_time())?;
    status.set_item("platform", platform())?;
    status.set_item("architecture", architecture())?;
    status.set_item("cpu_cores", system.cpus().len())?;
    status.set_item("memory_total_bytes", system.total_memory())?;
    status.set_item("memory_used_bytes", system.used_memory())?;
    status.set_item("uptime_seconds", System::uptime())?;

    let battery = PyDict::new(py);

    let manager = Manager::new();

    if let Ok(manager) = manager {
        if let Ok(mut batteries) = manager.batteries() {
            if let Some(Ok(battery_data)) = batteries.next() {
                let percent = battery_data.state_of_charge().value * 100.0;
                let state = format!("{:?}", battery_data.state());

                battery.set_item("available", true)?;
                battery.set_item("percent", percent)?;
                battery.set_item("status", state)?;

                let remaining_minutes = battery_data
                    .time_to_empty()
                    .map(|duration| {
                        duration
                            .get::<battery::units::time::minute>()
                            as u64
                    });

                battery.set_item("remaining_minutes", remaining_minutes)?;
            } else {
                battery.set_item("available", false)?;
            }
        } else {
            battery.set_item("available", false)?;
        }
    } else {
        battery.set_item("available", false)?;
    }

    status.set_item("battery", battery)?;

    Ok(status.into())
}

#[pyfunction]
fn battery_info() -> PyResult<Option<(f32, String, Option<u64>)>> {
    let manager = match Manager::new() {
        Ok(manager) => manager,
        Err(_) => return Ok(None),
    };

    let mut batteries = match manager.batteries() {
        Ok(batteries) => batteries,
        Err(_) => return Ok(None),
    };

    let battery = match batteries.next() {
        Some(Ok(battery)) => battery,
        _ => return Ok(None),
    };

    let percent = battery.state_of_charge().value * 100.0;
    let status = format!("{:?}", battery.state());

    let remaining_minutes = battery
        .time_to_empty()
        .map(|duration| duration.get::<battery::units::time::minute>() as u64);

    Ok(Some((percent, status, remaining_minutes)))
}

/// Mounted filesystems with a real size (pseudo filesystems are skipped).
#[pyfunction]
fn disk_status(py: Python<'_>) -> PyResult<Vec<Py<PyAny>>> {
    let disks = Disks::new_with_refreshed_list();
    let mut result = Vec::new();

    for disk in disks.list() {
        let total = disk.total_space();

        if total == 0 {
            continue;
        }

        let item = PyDict::new(py);
        item.set_item("mount", disk.mount_point().to_string_lossy().to_string())?;
        item.set_item("filesystem", disk.file_system().to_string_lossy().to_string())?;
        item.set_item("total_bytes", total)?;
        item.set_item("available_bytes", disk.available_space())?;

        result.push(item.into_any().unbind());
    }

    Ok(result)
}

/// Cumulative network counters since boot (loopback excluded from totals).
#[pyfunction]
fn network_status(py: Python<'_>) -> PyResult<Py<PyAny>> {
    let networks = Networks::new_with_refreshed_list();

    let mut received: u64 = 0;
    let mut transmitted: u64 = 0;
    let mut interfaces = Vec::new();

    for (name, data) in networks.iter() {
        let rx = data.total_received();
        let tx = data.total_transmitted();

        if name != "lo" {
            received += rx;
            transmitted += tx;
        }

        let item = PyDict::new(py);
        item.set_item("name", name)?;
        item.set_item("received_bytes", rx)?;
        item.set_item("transmitted_bytes", tx)?;
        interfaces.push(item);
    }

    let status = PyDict::new(py);
    status.set_item("received_bytes", received)?;
    status.set_item("transmitted_bytes", transmitted)?;
    status.set_item("interfaces", interfaces)?;

    Ok(status.into())
}

/// The processes using the most memory (empty if the platform hides them).
#[pyfunction]
#[pyo3(signature = (limit=5))]
fn top_processes(py: Python<'_>, limit: usize) -> PyResult<Vec<Py<PyAny>>> {
    let mut system = System::new();
    system.refresh_processes(ProcessesToUpdate::All, true);

    let mut processes: Vec<_> = system.processes().values().collect();
    processes.sort_by_key(|process| std::cmp::Reverse(process.memory()));

    let mut result = Vec::new();

    for process in processes.into_iter().take(limit.min(25)) {
        let item = PyDict::new(py);
        item.set_item("pid", process.pid().as_u32())?;
        item.set_item("name", process.name().to_string_lossy().to_string())?;
        item.set_item("memory_bytes", process.memory())?;

        result.push(item.into_any().unbind());
    }

    Ok(result)
}

/// Overall CPU usage in percent, measured over a short sampling window.
/// The GIL is released while sampling so other Python threads keep running.
#[pyfunction]
#[pyo3(signature = (sample_ms=250))]
fn cpu_percent(py: Python<'_>, sample_ms: u64) -> f32 {
    let window = Duration::from_millis(sample_ms).max(MINIMUM_CPU_UPDATE_INTERVAL);

    py.allow_threads(|| {
        let mut system = System::new_with_specifics(
            RefreshKind::nothing().with_cpu(CpuRefreshKind::nothing().with_cpu_usage()),
        );

        system.refresh_cpu_usage();
        std::thread::sleep(window);
        system.refresh_cpu_usage();

        system.global_cpu_usage()
    })
}

#[pymodule]
fn sally_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(platform, m)?)?;
    m.add_function(wrap_pyfunction!(architecture, m)?)?;
    m.add_function(wrap_pyfunction!(current_time, m)?)?;
    m.add_function(wrap_pyfunction!(system_info, m)?)?;
    m.add_function(wrap_pyfunction!(battery_info, m)?)?;
    m.add_function(wrap_pyfunction!(system_status, m)?)?;
    m.add_function(wrap_pyfunction!(disk_status, m)?)?;
    m.add_function(wrap_pyfunction!(network_status, m)?)?;
    m.add_function(wrap_pyfunction!(top_processes, m)?)?;
    m.add_function(wrap_pyfunction!(cpu_percent, m)?)?;

    Ok(())
}
