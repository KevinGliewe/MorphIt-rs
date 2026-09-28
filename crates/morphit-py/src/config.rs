//! `Config`: preset + dotted-key overrides, the same keys as the CLI's `--set`.

use morphit::{Config, Preset};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use serde_json::Value;

use crate::convert::{from_py, to_py};
use crate::errors;

/// Optimizer configuration. Start from a preset, then change any value by its
/// dotted key: `cfg["training.center_lr"] = 0.001`.
#[pyclass(name = "Config", module = "morphit_rs", from_py_object)]
#[derive(Clone)]
pub struct PyConfig {
    pub(crate) inner: Config,
}

impl PyConfig {
    fn set_value(&mut self, key: &str, v: Value) -> PyResult<()> {
        self.inner.set(key, v).map_err(errors::core)
    }
}

#[pymethods]
impl PyConfig {
    /// `Config()` is MorphIt-B; `Config("MorphIt-S")` another preset.
    #[new]
    #[pyo3(signature = (preset = "MorphIt-B"))]
    fn new(preset: &str) -> PyResult<Self> {
        Ok(PyConfig { inner: Config::from_preset_name(preset).map_err(errors::core)? })
    }

    /// A preset with the most common settings; everything else via `cfg[key] = value`.
    #[staticmethod]
    #[pyo3(signature = (name = "MorphIt-B", *, num_spheres = None, iterations = None, seed = None, device = None))]
    fn preset(
        name: &str,
        num_spheres: Option<usize>,
        iterations: Option<usize>,
        seed: Option<u64>,
        device: Option<String>,
    ) -> PyResult<Self> {
        let mut c = Self::new(name)?;
        if let Some(n) = num_spheres {
            c.set_value("model.num_spheres", n.into())?;
        }
        if let Some(n) = iterations {
            c.set_value("training.iterations", n.into())?;
        }
        if let Some(s) = seed {
            c.set_value("random_seed", s.into())?;
        }
        if let Some(d) = device {
            c.set_value("model.device", d.into())?;
        }
        Ok(c)
    }

    /// A config from a (partial) nested dict, as written under `"config"` in a result.
    #[staticmethod]
    fn from_dict(d: &Bound<'_, PyDict>) -> PyResult<Self> {
        let v: Value = from_py(d.as_any())?;
        Ok(PyConfig { inner: Config::from_json_str(&v.to_string()).map_err(errors::core)? })
    }

    /// A config from nested JSON text.
    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(PyConfig { inner: Config::from_json_str(text).map_err(errors::core)? })
    }

    /// The preset names.
    #[staticmethod]
    fn presets() -> Vec<&'static str> {
        Preset::ALL.iter().map(|p| p.name()).collect()
    }

    fn __getitem__<'py>(&self, py: Python<'py>, key: &str) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner.get(key).map_err(errors::core)?)
    }

    fn __setitem__(&mut self, key: &str, value: &Bound<'_, PyAny>) -> PyResult<()> {
        let v: Value = from_py(value)?;
        self.set_value(key, v)
    }

    /// Apply `{dotted_key: value}` updates; all or nothing.
    fn update(&mut self, updates: &Bound<'_, PyDict>) -> PyResult<()> {
        let v: Value = from_py(updates.as_any())?;
        self.inner.apply_updates_json(&v.to_string()).map_err(errors::core)
    }

    /// Check that a session can run with this config (raises `ConfigError`).
    fn validate(&self) -> PyResult<()> {
        self.inner.validate().map_err(errors::core)
    }

    /// The nested config as a dict (`model`, `training`, `visualization`, ...).
    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner.to_json_value())
    }

    fn to_json(&self) -> String {
        serde_json::to_string_pretty(&self.inner.to_json_value()).expect("config serializes")
    }

    fn copy(&self) -> Self {
        self.clone()
    }

    fn __copy__(&self) -> Self {
        self.clone()
    }

    fn __deepcopy__(&self, _memo: &Bound<'_, PyAny>) -> Self {
        self.clone()
    }

    fn __eq__(&self, other: &Self) -> bool {
        self.inner.to_json_value() == other.inner.to_json_value()
    }

    fn __repr__(&self) -> String {
        let m = &self.inner.model;
        format!(
            "Config(num_spheres={}, iterations={}, seed={}, device={:?})",
            m.num_spheres,
            self.inner.training.iterations,
            self.inner.random_seed.map_or("None".into(), |s| s.to_string()),
            m.device
        )
    }
}
