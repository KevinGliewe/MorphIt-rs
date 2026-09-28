//! Python bindings of MorphIt (`morphit_rs._native`; the public package is
//! `morphit_rs`, see `python/morphit_rs`). Heavy work runs with the GIL
//! released; sessions follow the C API's locking scheme.

mod config;
mod convert;
mod errors;
mod export;
mod logging;
mod mesh;
mod result;
mod robot;
mod session;

use std::sync::Arc;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::config::PyConfig;
use crate::mesh::PyMesh;
use crate::result::PyPackResult;

/// Version of the library (the workspace version).
#[pyfunction]
fn version() -> &'static str {
    env!("CARGO_PKG_VERSION")
}

/// The preset names: MorphIt-V, -S, -B, -Obj, -Obj-mass.
#[pyfunction]
fn presets() -> Vec<&'static str> {
    morphit::Preset::ALL.iter().map(|p| p.name()).collect()
}

/// The GPU adapters (`index` is the `N` of `device="gpu:N"`).
#[pyfunction]
fn devices(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    let list = py.detach(morphit::list_devices);
    list.into_iter()
        .map(|g| {
            let d = PyDict::new(py);
            d.set_item("index", g.index)?;
            d.set_item("name", g.name)?;
            d.set_item("backend", g.backend)?;
            d.set_item("kind", g.kind)?;
            d.set_item("software", g.software)?;
            Ok(d)
        })
        .collect()
}

/// Size the worker pool (default: all cores). Only before the first pack.
#[pyfunction]
fn set_num_threads(n: usize) -> PyResult<()> {
    rayon::ThreadPoolBuilder::new()
        .num_threads(n)
        .build_global()
        .map_err(|e| errors::StateError::new_err(format!("the thread pool is already running: {e}")))
}

/// Pack `mesh` with `config` (all iterations, then the final prune). The GIL
/// is released, so other Python threads keep running.
#[pyfunction]
#[pyo3(signature = (mesh, config = None))]
fn pack(py: Python<'_>, mesh: &PyMesh, config: Option<&PyConfig>) -> PyResult<PyPackResult> {
    let config = match config {
        Some(c) => c.inner.clone(),
        None => morphit::Config::from_preset(morphit::Preset::B),
    };
    let mesh = Arc::clone(&mesh.inner);
    let inner = py.detach(move || morphit::pack(config, mesh)).map_err(errors::core)?;
    Ok(PyPackResult { inner })
}

/// The bundled example objects: `[{"name", "label", "filename", "default"}]`
/// (files in the repository's `web/examples`).
#[pyfunction]
fn example_objects(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    morphit_robot::examples::EXAMPLE_OBJECTS
        .iter()
        .map(|e| {
            let d = PyDict::new(py);
            d.set_item("name", e.name)?;
            d.set_item("label", e.label)?;
            d.set_item("filename", e.filename)?;
            d.set_item("default", e.default)?;
            Ok(d)
        })
        .collect()
}

/// The bundled example robots: `[{"name", "label", "folder", "urdf", "default"}]`.
#[pyfunction]
fn example_robots(py: Python<'_>) -> PyResult<Vec<Bound<'_, PyDict>>> {
    morphit_robot::examples::EXAMPLE_ROBOTS
        .iter()
        .map(|e| {
            let d = PyDict::new(py);
            d.set_item("name", e.name)?;
            d.set_item("label", e.label)?;
            d.set_item("folder", e.folder)?;
            d.set_item("urdf", e.urdf)?;
            d.set_item("default", e.default)?;
            Ok(d)
        })
        .collect()
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    errors::register(m)?;
    m.add_class::<PyConfig>()?;
    m.add_class::<PyMesh>()?;
    m.add_class::<PyPackResult>()?;
    m.add_class::<session::PySession>()?;
    m.add_class::<session::PyStepInfo>()?;
    m.add_class::<robot::PyRobotPackage>()?;
    m.add_function(wrap_pyfunction!(version, m)?)?;
    m.add_function(wrap_pyfunction!(presets, m)?)?;
    m.add_function(wrap_pyfunction!(devices, m)?)?;
    m.add_function(wrap_pyfunction!(set_num_threads, m)?)?;
    m.add_function(wrap_pyfunction!(pack, m)?)?;
    m.add_function(wrap_pyfunction!(example_objects, m)?)?;
    m.add_function(wrap_pyfunction!(example_robots, m)?)?;
    m.add_function(wrap_pyfunction!(export::object_urdf, m)?)?;
    m.add_function(wrap_pyfunction!(export::object_mjcf, m)?)?;
    m.add_function(wrap_pyfunction!(export::spheres_from_urdf, m)?)?;
    m.add_function(wrap_pyfunction!(export::evaluate, m)?)?;
    m.add_function(wrap_pyfunction!(export::link_quality, m)?)?;
    m.add_function(wrap_pyfunction!(export::aggregate, m)?)?;
    m.add_function(wrap_pyfunction!(logging::_install_logging, m)?)?;
    m.add_function(wrap_pyfunction!(logging::_drain_logs, m)?)?;
    Ok(())
}
