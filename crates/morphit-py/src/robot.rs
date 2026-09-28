//! `RobotPackage`: a URDF package in memory, inspected, packed link by link
//! and assembled into a spherical URDF (the web service's robot pipeline).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use morphit::{Config, Session};
use morphit_robot::assemble::{MemSpheres, load_spheres_str, rewrite_urdf_text};
use morphit_robot::color::safe_color_rgba;
use morphit_robot::config::{PackParams, parse_advanced};
use morphit_robot::inspect::{CollisionItem, InspectionReport, inspect_urdf_in, select_urdf_in};
use morphit_robot::kinematics::zero_config_link_poses;
use morphit_robot::pack::{json_filename, link_config, pack_mesh_path};
use morphit_robot::vfs::MemPackage;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyDict};

use crate::convert::{from_py, to_py};
use crate::errors;
use crate::mesh::PyMesh;
use crate::result::PyPackResult;
use crate::session::{PySession, PyStepInfo};

/// A robot description package (URDF plus meshes) held in memory.
#[pyclass(name = "RobotPackage", module = "morphit_rs")]
pub struct PyRobotPackage {
    pkg: Arc<MemPackage>,
    spheres: Mutex<MemSpheres>,
}

impl PyRobotPackage {
    fn with(pkg: MemPackage) -> Self {
        PyRobotPackage { pkg: Arc::new(pkg), spheres: Mutex::default() }
    }

    fn report(&self, urdf: Option<&str>) -> PyResult<InspectionReport> {
        let key = select_urdf_in(&self.pkg, urdf).map_err(errors::robot)?;
        inspect_urdf_in(&self.pkg, &key).map_err(errors::robot)
    }

    fn spheres(&self) -> std::sync::MutexGuard<'_, MemSpheres> {
        self.spheres.lock().unwrap_or_else(|p| p.into_inner())
    }

    fn record(&self, link: &str, index: usize, json: &str) -> PyResult<()> {
        let set = load_spheres_str(json, &json_filename(link, index)).map_err(errors::robot)?;
        self.spheres().0.insert((link.to_string(), index), set);
        Ok(())
    }
}

/// The web API's pack parameters, with its defaults and limits.
#[allow(clippy::too_many_arguments)]
fn pack_params(
    variant: &str,
    num_spheres: usize,
    iterations: usize,
    seed: Option<u64>,
    advanced: Option<&Bound<'_, PyDict>>,
    union_overlapping_bodies: bool,
    convex_hull: bool,
) -> PyResult<PackParams> {
    let advanced = match advanced {
        Some(d) => {
            let v: serde_json::Value = from_py(d.as_any())?;
            parse_advanced(&v.to_string()).map_err(errors::robot)?
        }
        None => Vec::new(),
    };
    let p = PackParams {
        variant: variant.to_string(),
        num_spheres,
        iterations,
        seed,
        advanced,
        union_overlapping_bodies,
        convex_hull,
    };
    p.validate().map_err(errors::robot)?;
    Ok(p)
}

fn link_session(
    pkg: &MemPackage,
    item: &CollisionItem,
    params: &PackParams,
    device: &str,
) -> PyResult<(Config, Arc<morphit::Mesh>)> {
    let path = pack_mesh_path(item).map_err(errors::robot)?;
    let (config, _) = link_config(item, params, device, "spheres").map_err(errors::robot)?;
    let mesh = Arc::new(pkg.load_mesh(path).map_err(errors::robot)?);
    Ok((config, mesh))
}

#[pymethods]
impl PyRobotPackage {
    /// An empty package; add files with `add_file`.
    #[new]
    fn new() -> Self {
        Self::with(MemPackage::new())
    }

    /// Every file under `path` (the package root).
    #[staticmethod]
    fn from_folder(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let pkg = py.detach(move || MemPackage::from_dir(&path)).map_err(errors::robot)?;
        Ok(Self::with(pkg))
    }

    /// A package from a `.zip` archive: its bytes or a file path.
    #[staticmethod]
    fn from_zip(py: Python<'_>, archive: &Bound<'_, PyAny>) -> PyResult<Self> {
        let bytes: Vec<u8> = match archive.cast::<PyBytes>() {
            Ok(b) => b.as_bytes().to_vec(),
            Err(_) => {
                let path: PathBuf = archive.extract()?;
                std::fs::read(&path).map_err(|e| {
                    errors::robot(morphit_robot::Error::Io(format!("cannot read {}: {e}", path.display())))
                })?
            }
        };
        let pkg = py.detach(move || MemPackage::from_zip(&bytes)).map_err(errors::robot)?;
        Ok(Self::with(pkg))
    }

    /// A package from `{relative_path: bytes}`.
    #[staticmethod]
    fn from_files(files: BTreeMap<String, Vec<u8>>) -> PyResult<Self> {
        let mut pkg = MemPackage::new();
        for (path, bytes) in files {
            pkg.insert(&path, bytes).map_err(errors::robot)?;
        }
        Ok(Self::with(pkg))
    }

    /// Add a file under its path relative to the package root; returns the normalized path.
    fn add_file(&mut self, path: &str, data: Vec<u8>) -> PyResult<String> {
        Arc::make_mut(&mut self.pkg).insert(path, data).map_err(errors::robot)
    }

    /// Every file path, sorted.
    fn files(&self) -> Vec<String> {
        self.pkg.files().map(str::to_string).collect()
    }

    /// Every `.urdf` in the package, sorted.
    fn urdfs(&self) -> Vec<String> {
        self.pkg.urdfs()
    }

    /// Classify every `<collision>`: which to pack, which to remove. The
    /// items with `action == "pack"` go to `pack_link`.
    #[pyo3(signature = (urdf = None))]
    fn inspect<'py>(&self, py: Python<'py>, urdf: Option<&str>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &self.report(urdf)?)
    }

    /// World pose of every link with all joints at zero:
    /// `{link: {"rotation": [9, column-major], "translation": [3]}}`.
    #[pyo3(signature = (urdf = None))]
    fn link_poses<'py>(&self, py: Python<'py>, urdf: Option<&str>) -> PyResult<Bound<'py, PyDict>> {
        let key = select_urdf_in(&self.pkg, urdf).map_err(errors::robot)?;
        let text = String::from_utf8_lossy(self.pkg.read(&key).unwrap_or_default()).into_owned();
        let poses = zero_config_link_poses(&text).map_err(errors::robot)?;
        let out = PyDict::new(py);
        for (link, p) in &poses {
            let d = PyDict::new(py);
            d.set_item("rotation", p.rotation.to_cols_array())?;
            d.set_item("translation", p.translation.to_array())?;
            out.set_item(link, d)?;
        }
        Ok(out)
    }

    /// Load a mesh of the package (e.g. an item's `mesh_path`).
    fn mesh(&self, py: Python<'_>, path: &str) -> PyResult<PyMesh> {
        let pkg = Arc::clone(&self.pkg);
        let path = path.to_string();
        let mesh = py.detach(move || pkg.load_mesh(&path)).map_err(errors::robot)?;
        Ok(PyMesh::new(Arc::new(mesh)))
    }

    /// A session packing one `pack` item of `inspect()` (the web API's
    /// defaults and limits). Record the finished result with `set_link_result`.
    #[pyo3(signature = (item, *, variant = "MorphIt-B", num_spheres = 20, iterations = 200, seed = None,
                        advanced = None, union_overlapping_bodies = true, convex_hull = false, device = "auto"))]
    #[allow(clippy::too_many_arguments)]
    fn pack_link(
        &self,
        py: Python<'_>,
        item: &Bound<'_, PyAny>,
        variant: &str,
        num_spheres: usize,
        iterations: usize,
        seed: Option<u64>,
        advanced: Option<&Bound<'_, PyDict>>,
        union_overlapping_bodies: bool,
        convex_hull: bool,
        device: &str,
    ) -> PyResult<PySession> {
        let item: CollisionItem = from_py(item)?;
        let params = pack_params(
            variant,
            num_spheres,
            iterations,
            seed,
            advanced,
            union_overlapping_bodies,
            convex_hull,
        )?;
        let (config, mesh) = link_session(&self.pkg, &item, &params, device)?;
        let s = py.detach(move || Session::new(config, mesh)).map_err(errors::core)?;
        Ok(PySession::from_session(s))
    }

    /// Pack every `pack` item and record the results. `callback(link, index,
    /// step)` is called after each step; returning `False` stops after the
    /// links finished so far. Returns `{(link, index): PackResult}`.
    #[pyo3(signature = (*, urdf = None, variant = "MorphIt-B", num_spheres = 20, iterations = 200, seed = None,
                        advanced = None, union_overlapping_bodies = true, convex_hull = false, device = "auto",
                        callback = None))]
    #[allow(clippy::too_many_arguments)]
    fn pack_all<'py>(
        &self,
        py: Python<'py>,
        urdf: Option<&str>,
        variant: &str,
        num_spheres: usize,
        iterations: usize,
        seed: Option<u64>,
        advanced: Option<&Bound<'py, PyDict>>,
        union_overlapping_bodies: bool,
        convex_hull: bool,
        device: &str,
        callback: Option<Bound<'py, PyAny>>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let report = self.report(urdf)?;
        let params = pack_params(
            variant,
            num_spheres,
            iterations,
            seed,
            advanced,
            union_overlapping_bodies,
            convex_hull,
        )?;
        let out = PyDict::new(py);
        for item in report.to_pack() {
            let (config, mesh) = link_session(&self.pkg, item, &params, device)?;
            let session = py.detach(move || Session::new(config, mesh)).map_err(errors::core)?;
            let session = PySession::from_session(session);
            let link = item.link_name.as_str();
            let index = item.collision_index;
            let outcome = session.run_with(py, |py, info| match &callback {
                Some(cb) => {
                    let ret = cb.call1((link, index, Bound::new(py, PyStepInfo::new(info))?))?;
                    Ok(!matches!(ret.extract::<bool>(), Ok(false)))
                }
                None => Ok(true),
            })?;
            if outcome == "cancelled" {
                // The callback returned False: stop, keeping the links packed so far.
                break;
            }
            let result = session.result_inner();
            self.record(link, index, &result.to_json_string())?;
            out.set_item((link, index), Bound::new(py, PyPackResult { inner: result })?)?;
        }
        Ok(out)
    }

    /// Record the spheres of `link[index]` (a `PackResult`).
    fn set_link_result(&self, link: &str, index: usize, result: &PyPackResult) -> PyResult<()> {
        self.record(link, index, &result.inner.to_json_string())
    }

    /// The recorded `(link, index)` pairs.
    fn link_results(&self) -> Vec<(String, usize)> {
        let mut keys: Vec<_> = self.spheres().0.keys().cloned().collect();
        keys.sort();
        keys
    }

    /// Forget every recorded link result.
    fn clear_link_results(&self) {
        self.spheres().0.clear();
    }

    /// The URDF with every packed collision replaced by sphere links.
    /// `base_color` is `#rrggbb`; `color_variation` (0..1) spreads the hue
    /// over the links. Returns `(urdf_text, stats)`.
    #[pyo3(signature = (urdf = None, *, base_color = None, color_variation = 0.0))]
    fn assemble<'py>(
        &self,
        py: Python<'py>,
        urdf: Option<&str>,
        base_color: Option<&str>,
        color_variation: f64,
    ) -> PyResult<(String, Bound<'py, PyDict>)> {
        let report = self.report(urdf)?;
        let text = String::from_utf8_lossy(self.pkg.read(&report.urdf_path).unwrap_or_default()).into_owned();
        let spheres = self.spheres();
        let (out, stats) = rewrite_urdf_text(
            &text,
            &report,
            &*spheres,
            safe_color_rgba(base_color),
            color_variation.clamp(0.0, 1.0),
        )
        .map_err(errors::robot)?;
        let d = PyDict::new(py);
        d.set_item("links_with_collisions_replaced", stats.links_with_collisions_replaced)?;
        d.set_item("mesh_collisions_replaced", stats.mesh_collisions_replaced)?;
        d.set_item("primitive_collisions_removed", stats.primitive_collisions_removed)?;
        d.set_item("sphere_collisions_removed", stats.sphere_collisions_removed)?;
        d.set_item("sphere_children_added", stats.sphere_children_added)?;
        d.set_item("skipped", stats.skipped_summary())?;
        Ok((out, d))
    }

    fn __repr__(&self) -> String {
        format!("RobotPackage(files={}, urdfs={:?})", self.pkg.files().count(), self.pkg.urdfs())
    }
}
