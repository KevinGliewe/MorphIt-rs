//! Download the prepared mesh: the mesh MorphIt packs after merging
//! overlapping bodies and, optionally, replacing bodies with convex hulls.

use std::collections::HashMap;
use std::sync::Arc;

use axum::extract::multipart::MultipartRejection;
use axum::extract::{Multipart, Query, State};
use axum::http::{HeaderName, header};
use axum::response::Response;
use morphit::{Mesh, MeshFormat, MeshPrepOptions, MeshPrepReport};
use morphit_robot::inspect::Action;
use serde_json::Value;

use crate::error::{ApiError, ApiResult};
use crate::forms::{Form, parse_bool};
use crate::morph::{ALLOWED_EXTENSIONS, extension_error};
use crate::robot::NO_REPORT_INSPECT;
use crate::{
    MAX_OBJECT_MESH_BYTES, SharedState, ascii_json, blocking, bytes_response, file_stem, file_suffix,
};

/// `obj` or `stl`, else a 422 at `loc`.
fn parse_format(loc: &str, v: &str) -> ApiResult<MeshFormat> {
    v.parse().map_err(|_| {
        ApiError::validation(loc, "format", "enum", "Input should be 'obj' or 'stl'", Value::String(v.into()))
    })
}

/// Optional boolean query parameter (FastAPI parsing, 422 otherwise).
fn query_bool(q: &HashMap<String, String>, name: &str, default: bool) -> ApiResult<bool> {
    match q.get(name).map(String::as_str) {
        None | Some("") => Ok(default),
        Some(v) => parse_bool("query", name, v),
    }
}

/// A file name safe for `Content-Disposition`.
fn safe_filename(stem: &str, ext: &str) -> String {
    let stem: String =
        stem.chars().map(|c| if c.is_ascii_alphanumeric() || "-_.".contains(c) { c } else { '_' }).collect();
    format!("{}_prepared.{ext}", if stem.is_empty() { "mesh" } else { &stem })
}

/// The prepared mesh as a download, with the report in `X-Morphit-Mesh-Prep`.
fn mesh_file(mesh: &Mesh, format: MeshFormat, stem: &str, report: &MeshPrepReport) -> Response {
    let disposition = format!("attachment; filename=\"{}\"", safe_filename(stem, format.extension()));
    bytes_response(
        mesh.to_bytes(format),
        format.mime(),
        &[
            (header::CONTENT_DISPOSITION, &disposition),
            (HeaderName::from_static("x-morphit-mesh-prep"), &ascii_json(report)),
        ],
    )
}

/// `POST /api/mesh/prepare`: form fields `mesh` (file), `union_overlapping_bodies`
/// (default true), `convex_hull` (default false) and `format` (`obj`, default,
/// or `stl`). Returns the prepared mesh with the report in `X-Morphit-Mesh-Prep`.
pub async fn prepare(
    State(_state): State<SharedState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> ApiResult<Response> {
    let form = Form::read(multipart).await?;
    let mesh = form.file("mesh")?.clone();
    let options = MeshPrepOptions {
        union_overlapping_bodies: form.bool_or("union_overlapping_bodies", true)?,
        convex_hull: form.bool_or("convex_hull", false)?,
    };
    let format = parse_format("body", form.text_or("format", "obj"))?;
    let suffix = file_suffix(mesh.filename.as_deref());
    if !ALLOWED_EXTENSIONS.contains(&suffix.as_str()) {
        return Err(extension_error());
    }
    if mesh.data.len() > MAX_OBJECT_MESH_BYTES {
        return Err(ApiError::too_large(mesh.data.len(), MAX_OBJECT_MESH_BYTES, "Mesh file"));
    }
    let stem = file_stem(mesh.filename.as_deref(), "mesh");
    blocking(move || {
        let m = Arc::new(Mesh::load_from_bytes(&mesh.data, &suffix, Some(format!("input{suffix}")))?);
        let (prepared, report) = m.prepared_with(options);
        Ok(mesh_file(&prepared, format, &stem, &report))
    })
    .await
}

/// `GET /api/robot/prepared-mesh?session_id=&link_name=&collision_index=`
/// `&union_overlapping_bodies=&convex_hull=&format=`: the prepared mesh of one
/// `pack` collision of a robot session (defaults as `/api/mesh/prepare`).
pub async fn robot_prepared_mesh(
    State(state): State<SharedState>,
    Query(q): Query<HashMap<String, String>>,
) -> ApiResult<Response> {
    let required =
        |name: &str| q.get(name).map(String::as_str).ok_or_else(|| ApiError::missing("query", name));
    let id = required("session_id")?;
    let link_name = required("link_name")?.to_string();
    let index_raw = required("collision_index")?;
    let collision_index: usize = index_raw.trim().parse().map_err(|_| {
        ApiError::validation(
            "query",
            "collision_index",
            "int_parsing",
            "Input should be a valid integer, unable to parse string as an integer",
            Value::String(index_raw.into()),
        )
    })?;
    let options = MeshPrepOptions {
        union_overlapping_bodies: query_bool(&q, "union_overlapping_bodies", true)?,
        convex_hull: query_bool(&q, "convex_hull", false)?,
    };
    let format = parse_format("query", q.get("format").map_or("obj", String::as_str))?;
    state.sessions.gc();
    let session = state.sessions.get(id)?;
    let report = session.require_report(NO_REPORT_INSPECT)?;
    let item = report
        .collisions
        .iter()
        .find(|c| c.link_name == link_name && c.collision_index == collision_index)
        .ok_or_else(|| ApiError::not_found(format!("no collision item: {link_name}[{collision_index}]")))?;
    if item.action != Action::Pack {
        return Err(ApiError::bad_request(format!(
            "item {link_name}[{collision_index}] is not a mesh collision that MorphIt packs"
        )));
    }
    let path = item.mesh_path.clone().ok_or_else(|| {
        ApiError::bad_request(format!("item {link_name}[{collision_index}] has no mesh file"))
    })?;
    let stem = format!("{link_name}_{collision_index}");
    blocking(move || {
        let m = Arc::new(Mesh::load(&path)?);
        let (prepared, report) = m.prepared_with(options);
        Ok(mesh_file(&prepared, format, &stem, &report))
    })
    .await
}
