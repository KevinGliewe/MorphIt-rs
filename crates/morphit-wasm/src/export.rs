//! Turning spheres into simulator models, and scoring a packing.

use morphit::glam::DVec3;
use morphit::{PackResult, QualityOptions, evaluate_packing};
use morphit_robot::color::{DEFAULT_SPHERE_RGBA, hex_to_rgba};
use morphit_robot::object_model::{
    ObjectModel, ObjectModelOptions, spheres_from_object_urdf, write_object_mjcf, write_object_urdf,
};
use morphit_robot::quality::{LinkQuality, aggregate_overall, quality_metrics};
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

use crate::dto::to_js;
use crate::err;
use crate::mesh::JsMesh;

#[derive(Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct ModelOptions {
    name: Option<String>,
    color: Option<String>,
    total_mass: Option<f64>,
    anchored: Option<bool>,
    decimals: Option<usize>,
}

#[derive(Serialize)]
struct ModelDto {
    text: String,
    centroid: [f64; 3],
}

fn centers_of(flat: &[f64]) -> Result<Vec<[f64; 3]>, JsError> {
    if flat.len() % 3 != 0 {
        return Err(JsError::new("centers must hold x y z triples"));
    }
    Ok(flat.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect())
}

fn model_options(options: JsValue) -> Result<ObjectModelOptions, JsError> {
    let o: ModelOptions = if options.is_undefined() || options.is_null() {
        Default::default()
    } else {
        serde_wasm_bindgen::from_value(options).map_err(err)?
    };
    let color = match o.color {
        Some(c) => hex_to_rgba(&c).ok_or_else(|| JsError::new(&format!("color must be #rrggbb, got {c}")))?,
        None => DEFAULT_SPHERE_RGBA,
    };
    let d = ObjectModelOptions::default();
    Ok(ObjectModelOptions {
        robot_name: o.name.unwrap_or(d.robot_name.clone()),
        color_rgba: color,
        decimals: o.decimals.unwrap_or(d.decimals),
        total_mass: o.total_mass.unwrap_or(d.total_mass),
        anchored: o.anchored.unwrap_or(false),
        ..d
    })
}

/// `write_object_urdf` or `write_object_mjcf`.
type ModelWriter = fn(&[[f64; 3]], &[f64], &ObjectModelOptions) -> morphit_robot::Result<ObjectModel>;

fn model(centers: &[f64], radii: &[f64], options: JsValue, write: ModelWriter) -> Result<JsValue, JsError> {
    let m = write(&centers_of(centers)?, radii, &model_options(options)?).map_err(err)?;
    to_js(&ModelDto { text: m.text, centroid: m.centroid })
}

/// The spheres as a URDF: one link per sphere on fixed joints, masses split
/// by volume (as the Python `create_object_urdf.py`).
#[wasm_bindgen(js_name = objectUrdf, unchecked_return_type = "ObjectModel")]
pub fn object_urdf(
    centers: &[f64],
    radii: &[f64],
    #[wasm_bindgen(unchecked_param_type = "ObjectModelOptions")] options: JsValue,
) -> Result<JsValue, JsError> {
    model(centers, radii, options, write_object_urdf)
}

/// The spheres as MJCF (MuJoCo): a body with one sphere geom per sphere.
#[wasm_bindgen(js_name = objectMjcf, unchecked_return_type = "ObjectModel")]
pub fn object_mjcf(
    centers: &[f64],
    radii: &[f64],
    #[wasm_bindgen(unchecked_param_type = "ObjectModelOptions")] options: JsValue,
) -> Result<JsValue, JsError> {
    model(centers, radii, options, write_object_mjcf)
}

#[derive(Serialize)]
struct SpheresDto {
    centers: Vec<f64>,
    radii: Vec<f64>,
}

/// Read the spheres back from an object URDF written by `objectUrdf`.
#[wasm_bindgen(js_name = spheresFromObjectUrdf, unchecked_return_type = "{ centers: number[]; radii: number[] }")]
pub fn spheres_from_urdf(text: &str) -> Result<JsValue, JsError> {
    let (c, r) = spheres_from_object_urdf(text).map_err(err)?;
    to_js(&SpheresDto { centers: c.into_iter().flatten().collect(), radii: r })
}

/// Score a result (its JSON) against the mesh: coverage ratios, surface
/// distances, mass and inertia errors (the `debug_quick_eval` metrics).
#[wasm_bindgen(js_name = evaluatePacking, unchecked_return_type = "QualityMetrics")]
pub fn evaluate(
    mesh: &JsMesh,
    result_json: &str,
    #[wasm_bindgen(unchecked_param_type = "Partial<QualityOptions>")] options: JsValue,
) -> Result<JsValue, JsError> {
    let r = PackResult::from_json_str(result_json).map_err(err)?;
    let mut opts = QualityOptions::default();
    if !(options.is_undefined() || options.is_null()) {
        let mut v = serde_json::to_value(opts).map_err(err)?;
        let over: serde_json::Value = serde_wasm_bindgen::from_value(options).map_err(err)?;
        if let (Some(base), Some(over)) = (v.as_object_mut(), over.as_object()) {
            base.extend(over.clone());
        }
        opts = serde_json::from_value(v).map_err(err)?;
    }
    let centers: Vec<DVec3> = r.centers.iter().map(|c| DVec3::from_array(*c)).collect();
    let masses = r.per_sphere_mass.then_some(r.masses.as_slice());
    to_js(&evaluate_packing(&mesh.inner, &centers, &r.radii, masses, &opts))
}

/// The web API's per-link metrics: surface distances and coverage of the
/// spheres against `mesh` (pass the prepared mesh).
#[wasm_bindgen(js_name = linkQuality, unchecked_return_type = "LinkQuality")]
pub fn link_quality(
    mesh: &JsMesh,
    link_name: &str,
    collision_index: usize,
    centers: &[f64],
    radii: &[f64],
) -> Result<JsValue, JsError> {
    to_js(&quality_metrics(&mesh.inner, link_name, collision_index, &centers_of(centers)?, radii))
}

/// Combine per-link metrics (area- and volume-weighted).
#[wasm_bindgen(js_name = aggregateOverall, unchecked_return_type = "Overall")]
pub fn aggregate(
    #[wasm_bindgen(unchecked_param_type = "LinkQuality[]")] links: JsValue,
) -> Result<JsValue, JsError> {
    let links: Vec<LinkQuality> = serde_wasm_bindgen::from_value(links).map_err(err)?;
    to_js(&aggregate_overall(&links))
}
