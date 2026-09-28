//! Simulator models from spheres, and packing quality metrics.

use std::sync::Arc;

use morphit::glam::DVec3;
use morphit::{QualityOptions, evaluate_packing};
use morphit_robot::color::{DEFAULT_SPHERE_RGBA, hex_to_rgba};
use morphit_robot::object_model::{
    ObjectModel, ObjectModelOptions, spheres_from_object_urdf, write_object_mjcf, write_object_urdf,
};
use morphit_robot::quality::{LinkQuality, aggregate_overall, quality_metrics};
use numpy::{AllowTypeChange, PyArrayLike1, PyArrayLike2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::convert::{CentersRadii, from_py, read_rows3, read_vec1, rows3, to_py, vec1};
use crate::errors;
use crate::mesh::PyMesh;
use crate::result::PyPackResult;

fn model_options(
    name: Option<String>,
    color: Option<String>,
    total_mass: Option<f64>,
    anchored: bool,
    decimals: Option<usize>,
) -> PyResult<ObjectModelOptions> {
    let color_rgba = match color {
        Some(c) => hex_to_rgba(&c).ok_or_else(|| errors::value(format!("color must be #rrggbb, got {c}")))?,
        None => DEFAULT_SPHERE_RGBA,
    };
    let d = ObjectModelOptions::default();
    Ok(ObjectModelOptions {
        robot_name: name.unwrap_or(d.robot_name.clone()),
        color_rgba,
        decimals: decimals.unwrap_or(d.decimals),
        total_mass: total_mass.unwrap_or(d.total_mass),
        anchored,
        ..d
    })
}

type Writer = fn(&[[f64; 3]], &[f64], &ObjectModelOptions) -> morphit_robot::Result<ObjectModel>;

#[allow(clippy::too_many_arguments)]
fn write(
    write: Writer,
    centers: PyArrayLike2<'_, f64, AllowTypeChange>,
    radii: PyArrayLike1<'_, f64, AllowTypeChange>,
    name: Option<String>,
    color: Option<String>,
    total_mass: Option<f64>,
    anchored: bool,
    decimals: Option<usize>,
) -> PyResult<(String, [f64; 3])> {
    let c = read_rows3(centers, "centers")?;
    let r = read_vec1(radii);
    let m =
        write(&c, &r, &model_options(name, color, total_mass, anchored, decimals)?).map_err(errors::robot)?;
    Ok((m.text, m.centroid))
}

/// The spheres as a URDF, one link per sphere on fixed joints with masses
/// split by volume (as the Python `create_object_urdf.py`). Returns
/// `(text, centroid)`; `anchored` welds the object to the world.
#[pyfunction]
#[pyo3(signature = (centers, radii, *, name = None, color = None, total_mass = None, anchored = false, decimals = None))]
pub(crate) fn object_urdf(
    centers: PyArrayLike2<'_, f64, AllowTypeChange>,
    radii: PyArrayLike1<'_, f64, AllowTypeChange>,
    name: Option<String>,
    color: Option<String>,
    total_mass: Option<f64>,
    anchored: bool,
    decimals: Option<usize>,
) -> PyResult<(String, [f64; 3])> {
    write(write_object_urdf, centers, radii, name, color, total_mass, anchored, decimals)
}

/// The spheres as MJCF for MuJoCo: a body with one sphere geom per sphere.
/// Returns `(text, centroid)`.
#[pyfunction]
#[pyo3(signature = (centers, radii, *, name = None, color = None, total_mass = None, anchored = false, decimals = None))]
pub(crate) fn object_mjcf(
    centers: PyArrayLike2<'_, f64, AllowTypeChange>,
    radii: PyArrayLike1<'_, f64, AllowTypeChange>,
    name: Option<String>,
    color: Option<String>,
    total_mass: Option<f64>,
    anchored: bool,
    decimals: Option<usize>,
) -> PyResult<(String, [f64; 3])> {
    write(write_object_mjcf, centers, radii, name, color, total_mass, anchored, decimals)
}

/// Read the spheres back from an object URDF: `(centers (n, 3), radii (n,))`.
#[pyfunction(name = "spheres_from_object_urdf")]
pub(crate) fn spheres_from_urdf<'py>(py: Python<'py>, text: &str) -> PyResult<CentersRadii<'py>> {
    let (c, r) = spheres_from_object_urdf(text).map_err(errors::robot)?;
    Ok((rows3(py, c.into_iter()), vec1(py, r)))
}

/// Score a result against its mesh (the `debug_quick_eval` metrics: coverage
/// ratios, surface distances in mm, mass/COM/inertia errors). Options:
/// `seed`, `surface_samples`, `volume_samples`, `bounds_expand`, `density`.
#[pyfunction(name = "evaluate_packing")]
#[pyo3(signature = (mesh, result, **options))]
pub(crate) fn evaluate<'py>(
    py: Python<'py>,
    mesh: &PyMesh,
    result: &PyPackResult,
    options: Option<&Bound<'py, PyDict>>,
) -> PyResult<Bound<'py, PyAny>> {
    let mut opts = serde_json::to_value(QualityOptions::default()).expect("options serialize");
    if let Some(o) = options {
        let over: serde_json::Value = from_py(o.as_any())?;
        for (k, v) in over.as_object().into_iter().flatten() {
            let slot = opts.get_mut(k).ok_or_else(|| errors::value(format!("unknown option {k:?}")))?;
            *slot = v.clone();
        }
    }
    let opts: QualityOptions = serde_json::from_value(opts).map_err(|e| errors::value(e.to_string()))?;
    let r = &result.inner;
    let centers: Vec<DVec3> = r.centers.iter().map(|c| DVec3::from_array(*c)).collect();
    let radii = r.radii.clone();
    let masses = r.per_sphere_mass.then(|| r.masses.clone());
    let m = Arc::clone(&mesh.inner);
    let metrics = py.detach(move || evaluate_packing(&m, &centers, &radii, masses.as_deref(), &opts));
    to_py(py, &metrics)
}

/// The web API's per-link metrics against `mesh` (pass the prepared mesh):
/// surface distances, coverage, area and volume.
#[pyfunction]
pub(crate) fn link_quality<'py>(
    py: Python<'py>,
    mesh: &PyMesh,
    link_name: &str,
    collision_index: usize,
    centers: PyArrayLike2<'_, f64, AllowTypeChange>,
    radii: PyArrayLike1<'_, f64, AllowTypeChange>,
) -> PyResult<Bound<'py, PyAny>> {
    let (c, r) = (read_rows3(centers, "centers")?, read_vec1(radii));
    let (m, link) = (Arc::clone(&mesh.inner), link_name.to_string());
    let q = py.detach(move || quality_metrics(&m, &link, collision_index, &c, &r));
    to_py(py, &q)
}

/// Combine per-link metrics (dicts from `link_quality`), area- and volume-weighted.
#[pyfunction(name = "aggregate_overall")]
pub(crate) fn aggregate<'py>(py: Python<'py>, links: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyAny>> {
    let links: Vec<LinkQuality> = from_py(links)?;
    to_py(py, &aggregate_overall(&links))
}
