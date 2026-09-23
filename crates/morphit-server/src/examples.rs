//! The bundled example library: object meshes (with thumbnails and pre-baked
//! URDFs) and robot packages (with pre-baked spherical URDFs), registered in
//! the same order and with the same labels as the Python API.

use axum::extract::{Path, State};
use axum::http::header;
use axum::response::Response;
use serde::Serialize;

use crate::error::{ApiError, ApiResult};
use crate::{SharedState, bytes_response, file_response, json_response};
use morphit_robot::object_model::extract_centroid;
use morphit_robot::py_repr;

pub use morphit_robot::examples::{EXAMPLE_OBJECTS, EXAMPLE_ROBOTS, ExampleObject, ExampleRobot};

pub fn find_object(name: &str) -> ApiResult<&'static ExampleObject> {
    EXAMPLE_OBJECTS
        .iter()
        .find(|o| o.name == name)
        .ok_or_else(|| ApiError::not_found(format!("example {} not found", py_repr(name))))
}

pub fn find_robot(name: &str) -> ApiResult<&'static ExampleRobot> {
    EXAMPLE_ROBOTS
        .iter()
        .find(|r| r.name == name)
        .ok_or_else(|| ApiError::not_found(format!("robot {} not in library", py_repr(name))))
}

#[derive(Serialize)]
struct ObjectEntry {
    name: &'static str,
    label: &'static str,
    filename: &'static str,
    default: bool,
    has_packed: bool,
    has_thumbnail: bool,
}

pub async fn list_objects(State(state): State<SharedState>) -> Response {
    let dir = state.examples_dir();
    let out: Vec<ObjectEntry> = EXAMPLE_OBJECTS
        .iter()
        .filter(|o| dir.join(o.filename).exists())
        .map(|o| ObjectEntry {
            name: o.name,
            label: o.label,
            filename: o.filename,
            default: o.default,
            has_packed: dir.join(format!("{}.urdf", o.name)).exists(),
            has_thumbnail: dir.join(o.filename).with_extension("png").exists(),
        })
        .collect();
    json_response(&out)
}

pub async fn get_object(State(state): State<SharedState>, Path(name): Path<String>) -> ApiResult<Response> {
    let o = find_object(&name)?;
    let path = state.examples_dir().join(o.filename);
    if !path.exists() {
        return Err(ApiError::internal(format!("example file missing on server: {}", o.filename)));
    }
    let media = match crate::file_suffix(Some(o.filename)).as_str() {
        ".obj" => "model/obj",
        ".stl" => "model/stl",
        _ => "application/octet-stream",
    };
    file_response(&path, Some(media), Some(o.filename), &[]).await
}

pub async fn get_packed(State(state): State<SharedState>, Path(name): Path<String>) -> ApiResult<Response> {
    let o = find_object(&name)?;
    let path = state.examples_dir().join(format!("{}.urdf", o.name));
    if !path.exists() {
        return Err(ApiError::not_found(format!("example {} has no pre-baked packing", py_repr(&name))));
    }
    let text = tokio::fs::read_to_string(&path)
        .await
        .map_err(|e| ApiError::internal(format!("cannot read {}: {e}", path.display())))?;
    let Some(c) = extract_centroid(&text) else {
        return Err(ApiError::internal(format!(
            "pre-baked URDF for {} is missing its morphit:centroid comment; re-run the bundling script or \
             re-paste a URDF from the UI (which now embeds the centroid in the file).",
            py_repr(&name)
        )));
    };
    let centroid = format!("{:.6},{:.6},{:.6}", c[0], c[1], c[2]);
    let centroid_header = header::HeaderName::from_static("x-morphit-centroid");
    Ok(bytes_response(text, "application/xml", &[(centroid_header, &centroid)]))
}

pub async fn get_thumbnail(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> ApiResult<Response> {
    let o = find_object(&name)?;
    let thumb = state.examples_dir().join(o.filename).with_extension("png");
    if !thumb.exists() {
        return Err(ApiError::not_found(format!("no thumbnail for example {}", py_repr(&name))));
    }
    file_response(&thumb, Some("image/png"), None, &[(header::CACHE_CONTROL, "public, max-age=86400")]).await
}

#[derive(Serialize)]
struct RobotEntry {
    name: &'static str,
    label: &'static str,
    urdf: &'static str,
    default: bool,
    has_spherical: bool,
}

pub async fn list_robots(State(state): State<SharedState>) -> Response {
    let dir = state.examples_dir();
    let out: Vec<RobotEntry> = EXAMPLE_ROBOTS
        .iter()
        .filter(|r| dir.join(r.folder).exists())
        .map(|r| RobotEntry {
            name: r.name,
            label: r.label,
            urdf: r.urdf,
            default: r.default,
            has_spherical: dir.join(r.spherical_urdf).exists(),
        })
        .collect();
    json_response(&out)
}

pub async fn robot_spherical(
    State(state): State<SharedState>,
    Path(name): Path<String>,
) -> ApiResult<Response> {
    let r = find_robot(&name)?;
    let path = state.examples_dir().join(r.spherical_urdf);
    if !path.exists() {
        return Err(ApiError::not_found(format!("robot {} has no pre-baked spherical URDF", py_repr(&name))));
    }
    file_response(&path, Some("application/xml"), None, &[]).await
}
