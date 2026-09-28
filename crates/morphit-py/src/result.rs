//! `PackResult`: packed spheres in the Python MorphIt JSON schema.

use std::path::PathBuf;

use morphit::PackResult;
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;

use crate::config::PyConfig;
use crate::convert::{rows3, to_py, vec1};
use crate::errors;

/// Result of a pack: `centers` (n, 3), `radii`, `masses`, the config and the
/// mesh preparation report. `save` writes the JSON the Python scripts read.
#[pyclass(name = "PackResult", module = "morphit_rs", frozen)]
pub struct PyPackResult {
    pub(crate) inner: PackResult,
}

#[pymethods]
impl PyPackResult {
    /// Read a result JSON file (this library's or the Python code's).
    #[staticmethod]
    fn load(path: PathBuf) -> PyResult<Self> {
        Ok(PyPackResult { inner: PackResult::load(&path).map_err(errors::core)? })
    }

    #[staticmethod]
    fn from_json(text: &str) -> PyResult<Self> {
        Ok(PyPackResult { inner: PackResult::from_json_str(text).map_err(errors::core)? })
    }

    /// Sphere centers, (n, 3) float64.
    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows3(py, self.inner.centers.iter().copied())
    }

    #[getter]
    fn radii<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.inner.radii.clone())
    }

    #[getter]
    fn masses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.inner.masses.clone())
    }

    #[getter]
    fn num_spheres(&self) -> usize {
        self.inner.num_spheres
    }

    #[getter]
    fn mesh_path(&self) -> &str {
        &self.inner.mesh_path
    }

    #[getter]
    fn per_sphere_mass(&self) -> bool {
        self.inner.per_sphere_mass
    }

    /// What mesh preparation did, as a dict (None for old files without it).
    #[getter]
    fn mesh_prep<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner.mesh_prep)
    }

    #[getter]
    fn config(&self) -> PyConfig {
        PyConfig { inner: self.inner.config.clone() }
    }

    fn to_json(&self) -> String {
        self.inner.to_json_string()
    }

    fn to_dict<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.inner)
    }

    /// Write the result JSON (parent folders are created).
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(&path).map_err(errors::core)
    }

    fn __len__(&self) -> usize {
        self.inner.radii.len()
    }

    fn __repr__(&self) -> String {
        format!("PackResult(num_spheres={}, mesh_path={:?})", self.inner.radii.len(), self.inner.mesh_path)
    }
}
