//! Object mode: pack an uploaded mesh into a sphere URDF, and score a packing
//! against its mesh.

use std::sync::Arc;

use axum::extract::multipart::MultipartRejection;
use axum::extract::{Multipart, State};
use axum::http::HeaderName;
use axum::response::Response;
use morphit::{Mesh, MeshPrepOptions};
use morphit_robot::config::{PackParams, parse_advanced};
use morphit_robot::history::{MAX_POINTS, history_json, subsample_history};
use morphit_robot::object_model::{ObjectModelOptions, spheres_from_object_urdf, write_object_urdf};
use morphit_robot::pack::run_session;
use morphit_robot::quality::{LinkQuality, aggregate_overall, quality_metrics};
use serde::Serialize;

use crate::error::{ApiError, ApiResult};
use crate::forms::{Form, count, seed};
use crate::{
    MAX_OBJECT_MESH_BYTES, SharedState, ascii_json, blocking, bytes_response, file_stem, file_suffix,
    json_response,
};

/// Mesh formats object mode accepts (as the Python API: trimesh-native ones).
pub const ALLOWED_EXTENSIONS: [&str; 3] = [".obj", ".stl", ".ply"];

fn extension_error() -> ApiError {
    let exts: Vec<String> = ALLOWED_EXTENSIONS.iter().map(|e| morphit_robot::py_repr(e)).collect();
    ApiError::bad_request(format!("mesh extension must be one of ({})", exts.join(", ")))
}

/// XML-escape a value placed inside an attribute.
fn attr_escape(s: &str) -> String {
    s.replace('&', "&amp;").replace('<', "&lt;").replace('"', "&quot;")
}

/// `POST /api/morph`: form fields `mesh` (file), `variant`, `num_spheres`,
/// `iterations`, `seed`, `advanced`, `base_color`, and the mesh preparation
/// switches `union_overlapping_bodies` (default true) and `convex_hull`
/// (default false). Returns the URDF with the
/// `X-Morphit-Centroid`, `X-Morphit-Loss` and `X-Morphit-Mesh-Prep` headers.
pub async fn morph(
    State(state): State<SharedState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> ApiResult<Response> {
    let form = Form::read(multipart).await?;
    let mesh = form.file("mesh")?.clone();
    let params_raw = (form.int_or("num_spheres", 20)?, form.int_or("iterations", 200)?, form.int("seed")?);
    let mut params = PackParams {
        variant: form.text_or("variant", "MorphIt-B").to_string(),
        num_spheres: count(params_raw.0),
        iterations: count(params_raw.1),
        seed: None,
        advanced: Vec::new(),
        union_overlapping_bodies: form.bool_or("union_overlapping_bodies", true)?,
        convex_hull: form.bool_or("convex_hull", false)?,
    };
    params.validate()?;
    let suffix = file_suffix(mesh.filename.as_deref());
    if !ALLOWED_EXTENSIONS.contains(&suffix.as_str()) {
        return Err(extension_error());
    }
    let robot_name = file_stem(Some(mesh.filename.as_deref().unwrap_or("object")), "object");
    params.advanced = parse_advanced(form.text_or("advanced", "{}"))?;
    params.seed = seed(params_raw.2)?;
    if mesh.data.len() > MAX_OBJECT_MESH_BYTES {
        return Err(ApiError::too_large(mesh.data.len(), MAX_OBJECT_MESH_BYTES, "Mesh file"));
    }
    let color = morphit_robot::color::safe_color_rgba(Some(form.text_or("base_color", "#3399ff")));
    let device = state.config.device.clone();

    let (urdf, history, prep) = blocking(move || {
        let config = params.config(&device, "", "morphit_result.json")?;
        let m = Mesh::load_from_bytes(&mesh.data, &suffix, Some(format!("input{suffix}")))?;
        let (session, history) = run_session(config, Arc::new(m), |_, _| {})?;
        let result = session.result();
        let opts = ObjectModelOptions {
            robot_name: attr_escape(&robot_name),
            color_rgba: color,
            ..Default::default()
        };
        let urdf = write_object_urdf(&result.centers, &result.radii, &opts)?;
        Ok((urdf, history, session.mesh_prep().clone()))
    })
    .await?;

    let c = urdf.centroid;
    let centroid = format!("{:.6},{:.6},{:.6}", c[0], c[1], c[2]);
    let loss = history_json(&subsample_history(&history, MAX_POINTS));
    let prep = ascii_json(&prep);
    Ok(bytes_response(
        urdf.text,
        "application/xml",
        &[
            (HeaderName::from_static("x-morphit-centroid"), &centroid),
            (HeaderName::from_static("x-morphit-loss"), &loss),
            (HeaderName::from_static("x-morphit-mesh-prep"), &prep),
        ],
    ))
}

/// The `{links, overall}` body shared by both analyze endpoints.
#[derive(Serialize)]
pub struct AnalyzeResponse {
    pub links: Vec<LinkQuality>,
    pub overall: morphit_robot::quality::Overall,
}

/// Refuse non-finite metrics like Python's JSON encoder does (500).
pub fn analyze_response(links: Vec<LinkQuality>) -> ApiResult<Response> {
    if let Some(bad) = links.iter().find(|l| !l.is_finite()) {
        return Err(ApiError::internal(format!(
            "non-finite quality metrics for {}[{}]",
            bad.link_name, bad.collision_index
        )));
    }
    let overall = aggregate_overall(&links);
    Ok(json_response(&AnalyzeResponse { links, overall }))
}

/// `POST /api/morph/analyze`: form fields `mesh` (file), `urdf` (the
/// object URDF text), `union_overlapping_bodies` (default true) and
/// `convex_hull` (default false), which score against the mesh prepared as it
/// was packed. Returns `{links: [row], overall}`.
pub async fn analyze(
    State(_state): State<SharedState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> ApiResult<Response> {
    let form = Form::read(multipart).await?;
    let mesh = form.file("mesh")?.clone();
    let urdf = form.required_text("urdf")?.to_string();
    let prep = MeshPrepOptions {
        union_overlapping_bodies: form.bool_or("union_overlapping_bodies", true)?,
        convex_hull: form.bool_or("convex_hull", false)?,
    };
    let suffix = file_suffix(mesh.filename.as_deref());
    if !ALLOWED_EXTENSIONS.contains(&suffix.as_str()) {
        return Err(extension_error());
    }
    if mesh.data.len() > MAX_OBJECT_MESH_BYTES {
        return Err(ApiError::too_large(mesh.data.len(), MAX_OBJECT_MESH_BYTES, "Mesh file"));
    }
    let (centers, radii) = spheres_from_object_urdf(&urdf)?;
    let name = file_stem(Some(mesh.filename.as_deref().unwrap_or("object")), "object");
    let row = blocking(move || {
        let m = Arc::new(Mesh::load_from_bytes(&mesh.data, &suffix, None)?);
        let (mesh, _) = m.prepared_with(prep);
        Ok(quality_metrics(&mesh, &name, 0, &centers, &radii))
    })
    .await?;
    analyze_response(vec![row])
}
