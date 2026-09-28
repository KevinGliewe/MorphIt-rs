//! Python exceptions for `morphit::Error` and `morphit_robot::Error`.

use pyo3::exceptions::{PyException, PyOSError, PyValueError};
use pyo3::prelude::*;
use pyo3::{PyErr, create_exception};

create_exception!(morphit_rs, MorphItError, PyException, "Base class of every MorphIt error.");
create_exception!(morphit_rs, MeshError, MorphItError, "A mesh could not be loaded or used.");
create_exception!(morphit_rs, StateError, MorphItError, "The operation is not valid in the session's state.");
create_exception!(morphit_rs, BusyError, StateError, "The session is running on another thread.");
create_exception!(morphit_rs, CancelledError, MorphItError, "The run was cancelled.");

/// `MorphItIOError(MorphItError, OSError)`, `ConfigError(MorphItError,
/// ValueError)` and `RobotError(MorphItError, ValueError)` have two bases,
/// which `create_exception!` cannot express; they are made in Python.
pub(crate) struct Classes {
    pub io: Py<PyAny>,
    pub config: Py<PyAny>,
    pub robot: Py<PyAny>,
}

static CLASSES: std::sync::OnceLock<Classes> = std::sync::OnceLock::new();

fn two_base(py: Python<'_>, name: &str, second: &Bound<'_, PyAny>, doc: &str) -> PyResult<Py<PyAny>> {
    let builtins = py.import("builtins")?;
    let bases = (py.get_type::<MorphItError>(), second.clone());
    let ns = pyo3::types::PyDict::new(py);
    ns.set_item("__doc__", doc)?;
    ns.set_item("__module__", "morphit_rs")?;
    Ok(builtins.getattr("type")?.call1((name, bases, ns))?.unbind())
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    let py = m.py();
    m.add("MorphItError", py.get_type::<MorphItError>())?;
    m.add("MeshError", py.get_type::<MeshError>())?;
    m.add("StateError", py.get_type::<StateError>())?;
    m.add("BusyError", py.get_type::<BusyError>())?;
    m.add("CancelledError", py.get_type::<CancelledError>())?;
    let classes = Classes {
        io: two_base(
            py,
            "MorphItIOError",
            py.get_type::<PyOSError>().as_any(),
            "A file could not be read or written.",
        )?,
        config: two_base(
            py,
            "ConfigError",
            py.get_type::<PyValueError>().as_any(),
            "An invalid configuration key or value (the key is in `.key`).",
        )?,
        robot: two_base(
            py,
            "RobotError",
            py.get_type::<PyValueError>().as_any(),
            "An invalid robot package, URDF or pack request.",
        )?,
    };
    m.add("MorphItIOError", classes.io.bind(py))?;
    m.add("ConfigError", classes.config.bind(py))?;
    m.add("RobotError", classes.robot.bind(py))?;
    let _ = CLASSES.set(classes);
    Ok(())
}

fn instance(py: Python<'_>, class: fn(&Classes) -> &Py<PyAny>, msg: String) -> PyErr {
    match CLASSES.get() {
        Some(c) => match class(c).bind(py).call1((msg.clone(),)) {
            Ok(v) => PyErr::from_value(v),
            Err(e) => e,
        },
        None => MorphItError::new_err(msg),
    }
}

/// Convert a core error.
pub(crate) fn core(e: morphit::Error) -> PyErr {
    Python::attach(|py| match e {
        morphit::Error::Io(m) => instance(py, |c| &c.io, m),
        morphit::Error::Mesh(m) => MeshError::new_err(m),
        morphit::Error::Config { ref key, .. } => {
            let key = key.clone();
            let err = instance(py, |c| &c.config, e.to_string());
            let _ = err.value(py).setattr("key", key);
            err
        }
        morphit::Error::State(m) => StateError::new_err(m),
        morphit::Error::Cancelled => CancelledError::new_err("cancelled"),
    })
}

/// Convert a robot-pipeline error.
pub(crate) fn robot(e: morphit_robot::Error) -> PyErr {
    match e {
        morphit_robot::Error::Morphit(e) => core(e),
        morphit_robot::Error::Io(m) => Python::attach(|py| instance(py, |c| &c.io, m)),
        morphit_robot::Error::Invalid(m) => Python::attach(|py| instance(py, |c| &c.robot, m)),
    }
}

/// A `ValueError` for bad arguments that never reach the library.
pub(crate) fn value(msg: impl Into<String>) -> PyErr {
    PyValueError::new_err(msg.into())
}
