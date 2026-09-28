//! `Mesh`: an immutable triangle mesh, shared cheaply between sessions.

use std::path::PathBuf;
use std::sync::Arc;

use morphit::glam::DVec3;
use morphit::{Mesh, MeshPrepOptions};
use numpy::ndarray::Array2;
use numpy::{AllowTypeChange, IntoPyArray, PyArray1, PyArray2, PyArrayLike2};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::convert::{dvec3s, read_rows3, to_py};
use crate::errors;

/// A closed triangle mesh (OBJ, STL, PLY or COLLADA), or one built from arrays.
#[pyclass(name = "Mesh", module = "morphit_rs", frozen)]
pub struct PyMesh {
    pub(crate) inner: Arc<Mesh>,
}

impl PyMesh {
    pub(crate) fn new(mesh: Arc<Mesh>) -> Self {
        PyMesh { inner: mesh }
    }
}

pub(crate) fn prep_options(union: bool, convex_hull: bool) -> MeshPrepOptions {
    MeshPrepOptions { union_overlapping_bodies: union, convex_hull }
}

#[pymethods]
impl PyMesh {
    /// Load a mesh file; the format follows the extension.
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let mesh = py.detach(|| Mesh::load(&path)).map_err(errors::core)?;
        Ok(PyMesh::new(Arc::new(mesh)))
    }

    /// A mesh from file contents; `ext` is `"obj"`, `"stl"`, `"ply"` or `"dae"`.
    #[staticmethod]
    #[pyo3(signature = (data, ext, name = None))]
    fn from_bytes(py: Python<'_>, data: Vec<u8>, ext: String, name: Option<String>) -> PyResult<Self> {
        let mesh = py.detach(|| Mesh::load_from_bytes(&data, &ext, name)).map_err(errors::core)?;
        Ok(PyMesh::new(Arc::new(mesh)))
    }

    /// A mesh from `vertices` (n, 3) and triangle `faces` (m, 3) of vertex indices.
    #[staticmethod]
    fn from_arrays(
        vertices: PyArrayLike2<'_, f64, AllowTypeChange>,
        faces: PyArrayLike2<'_, i64, AllowTypeChange>,
    ) -> PyResult<Self> {
        let v = read_rows3(vertices, "vertices")?;
        let f = faces.as_array();
        if f.ncols() != 3 {
            return Err(errors::value(format!("faces must have shape (m, 3), got {:?}", f.shape())));
        }
        let mut tris = Vec::with_capacity(f.nrows());
        for r in f.rows() {
            let idx = |i: usize| {
                u32::try_from(r[i]).map_err(|_| errors::value(format!("face index {} out of range", r[i])))
            };
            tris.push([idx(0)?, idx(1)?, idx(2)?]);
        }
        Ok(PyMesh::new(Arc::new(Mesh::from_arrays(&v, &tris).map_err(errors::core)?)))
    }

    /// File extensions `load` understands.
    #[staticmethod]
    fn supported_extensions() -> Vec<&'static str> {
        Mesh::supported_extensions().to_vec()
    }

    /// Vertex positions, (n, 3) float64.
    #[getter]
    fn vertices<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        dvec3s(py, self.inner.vertices())
    }

    /// Triangles as vertex indices, (m, 3) uint32.
    #[getter]
    fn faces<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<u32>> {
        let f = self.inner.faces();
        let flat: Vec<u32> = f.iter().flatten().copied().collect();
        Array2::from_shape_vec((f.len(), 3), flat).expect("m x 3").into_pyarray(py)
    }

    #[getter]
    fn volume(&self) -> f64 {
        self.inner.volume()
    }

    #[getter]
    fn area(&self) -> f64 {
        self.inner.area()
    }

    /// `(min, max)` corners of the bounding box.
    #[getter]
    fn bounds(&self) -> ([f64; 3], [f64; 3]) {
        let (lo, hi) = self.inner.bounds();
        (lo.to_array(), hi.to_array())
    }

    /// Center of mass of the solid (uniform density).
    #[getter]
    fn center_mass(&self) -> [f64; 3] {
        self.inner.center_mass().to_array()
    }

    /// Inertia tensor about the center of mass for unit density, (3, 3).
    #[getter]
    fn inertia<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let m = self.inner.moment_inertia();
        let rows = (0..3).map(|r| [m.col(0)[r], m.col(1)[r], m.col(2)[r]]);
        crate::convert::rows3(py, rows)
    }

    /// True when the file was wound inward and has been flipped.
    #[getter]
    fn winding_flipped(&self) -> bool {
        self.inner.winding_flipped()
    }

    #[getter]
    fn source_path(&self) -> Option<&str> {
        self.inner.source_path()
    }

    /// Summary: counts, volume, area, bounds, center of mass, ...
    fn info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let m = &self.inner;
        let (lo, hi) = m.bounds();
        let d = PyDict::new(py);
        d.set_item("vertices", m.vertices().len())?;
        d.set_item("faces", m.faces().len())?;
        d.set_item("volume", m.volume())?;
        d.set_item("area", m.area())?;
        d.set_item("scale", m.scale())?;
        d.set_item("bounds", (lo.to_array(), hi.to_array()))?;
        d.set_item("center_mass", m.center_mass().to_array())?;
        d.set_item("winding_flipped", m.winding_flipped())?;
        d.set_item("source_path", m.source_path())?;
        Ok(d)
    }

    /// Which of the points (k, 3) lie inside the mesh.
    fn contains<'py>(
        &self,
        py: Python<'py>,
        points: PyArrayLike2<'_, f64, AllowTypeChange>,
    ) -> PyResult<Bound<'py, PyArray1<bool>>> {
        let pts: Vec<DVec3> = read_rows3(points, "points")?.into_iter().map(DVec3::from_array).collect();
        let mesh = Arc::clone(&self.inner);
        let inside = py.detach(move || mesh.contains_many(&pts));
        Ok(inside.into_pyarray(py))
    }

    /// The mesh as packing prepares it: convex hull of each body (optional),
    /// then the union of overlapping closed bodies (optional). Cached.
    #[pyo3(signature = (union = true, convex_hull = false))]
    fn prepared(&self, py: Python<'_>, union: bool, convex_hull: bool) -> Self {
        let mesh = Arc::clone(&self.inner);
        let (m, _) = py.detach(move || mesh.prepared_with(prep_options(union, convex_hull)));
        PyMesh::new(m)
    }

    /// What `prepared` does (or did): action, reason, bodies, volumes, warnings.
    #[pyo3(signature = (union = true, convex_hull = false))]
    fn prep_report<'py>(
        &self,
        py: Python<'py>,
        union: bool,
        convex_hull: bool,
    ) -> PyResult<Bound<'py, PyAny>> {
        let mesh = Arc::clone(&self.inner);
        let (_, report) = py.detach(move || mesh.prepared_with(prep_options(union, convex_hull)));
        to_py(py, &report)
    }

    /// Wavefront OBJ text (coordinates round-trip exactly).
    fn to_obj(&self) -> String {
        self.inner.to_obj()
    }

    /// Binary STL.
    fn to_stl<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.inner.to_stl())
    }

    /// Write `.obj` or `.stl` (by extension).
    fn save(&self, path: PathBuf) -> PyResult<()> {
        self.inner.save(&path).map_err(errors::core)
    }

    fn __repr__(&self) -> String {
        format!(
            "Mesh(vertices={}, faces={}, volume={:.6e}{})",
            self.inner.vertices().len(),
            self.inner.faces().len(),
            self.inner.volume(),
            self.inner.source_path().map(|p| format!(", source={p:?}")).unwrap_or_default()
        )
    }
}
