use battery::Manager;
use chrono::Local;
use pyo3::prelude::*;
use pyo3::types::PyDict;
use sysinfo::{CpuRefreshKind, MemoryRefreshKind, RefreshKind, System};

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

#[pymodule]
fn sally_native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(platform, m)?)?;
    m.add_function(wrap_pyfunction!(architecture, m)?)?;
    m.add_function(wrap_pyfunction!(current_time, m)?)?;
    m.add_function(wrap_pyfunction!(system_info, m)?)?;
    m.add_function(wrap_pyfunction!(battery_info, m)?)?;
    m.add_function(wrap_pyfunction!(system_status, m)?)?;

    Ok(())
}
