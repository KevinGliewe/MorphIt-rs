//! HTTP API for MorphIt, a port of the Python reference's FastAPI service
//! (`web/api/main.py`): same routes, form fields, JSON keys, headers and
//! status codes, so the bundled single-page UI (`web/index.html`) works
//! against it unchanged.
//!
//! - Object mode: `POST /api/morph` packs an uploaded mesh and returns a
//!   URDF; `POST /api/morph/analyze` scores a packing.
//! - Robot mode: a session holds an uploaded URDF package; its collision
//!   meshes are packed one link at a time (`/api/robot/pack-link`, with live
//!   frames on the `/api/robot/pack-live` websocket), then assembled into a
//!   spherical URDF.
//! - The example library under `web/examples` and the UI at `/`.

pub mod error;
pub mod examples;
pub mod forms;
pub mod mesh;
pub mod morph;
pub mod robot;
pub mod sessions;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Duration;

use axum::Router;
use axum::extract::{DefaultBodyLimit, State};
use axum::http::{HeaderMap, HeaderName, HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use serde::Serialize;

use crate::error::{ApiError, ApiResult, MB};
use crate::sessions::SessionStore;

/// Largest single mesh accepted by object mode.
pub const MAX_OBJECT_MESH_BYTES: usize = 100 * MB;
/// Largest robot package (all files together).
pub const MAX_ROBOT_FOLDER_BYTES: usize = 200 * MB;
/// Largest single file in a robot package.
pub const MAX_ROBOT_PER_FILE_BYTES: usize = 50 * MB;

/// Server settings.
#[derive(Clone, Debug)]
pub struct ServerConfig {
    /// Directory holding `index.html` and `examples/`.
    pub web_dir: PathBuf,
    /// Optimizer device for every pack (`auto`, `cpu`, `gpu`, `gpu:N`).
    pub device: String,
    /// Where robot sessions keep their files.
    pub session_root: PathBuf,
    /// Idle time after which a robot session is deleted.
    pub session_ttl: Duration,
}

impl ServerConfig {
    /// Defaults: device `auto`, sessions under the temp directory, 1 h TTL.
    pub fn new(web_dir: impl Into<PathBuf>) -> Self {
        ServerConfig {
            web_dir: web_dir.into(),
            device: "auto".into(),
            session_root: std::env::temp_dir().join("morphit-robot-sessions"),
            session_ttl: Duration::from_secs(3600),
        }
    }
}

/// Shared state of all handlers.
pub struct AppState {
    pub config: ServerConfig,
    pub sessions: SessionStore,
}

impl AppState {
    pub fn new(config: ServerConfig) -> Arc<Self> {
        let sessions = SessionStore::new(config.session_root.clone(), config.session_ttl);
        Arc::new(AppState { config, sessions })
    }

    pub fn examples_dir(&self) -> PathBuf {
        self.config.web_dir.join("examples")
    }
}

pub type SharedState = Arc<AppState>;

/// All routes of the Python API.
pub fn router(state: SharedState) -> Router {
    Router::new()
        .route("/", get(index))
        .route("/healthz", get(healthz))
        .route("/api/examples", get(examples::list_objects))
        .route("/api/example/{name}", get(examples::get_object))
        .route("/api/example/{name}/packed", get(examples::get_packed))
        .route("/api/example/{name}/thumbnail", get(examples::get_thumbnail))
        .route("/api/morph", post(morph::morph))
        .route("/api/morph/analyze", post(morph::analyze))
        .route("/api/mesh/prepare", post(mesh::prepare))
        .route("/api/robot/examples", get(examples::list_robots))
        .route("/api/robot/example/{name}", post(robot::example_load))
        .route("/api/robot/example/{name}/spherical", get(examples::robot_spherical))
        .route("/api/robot/file", get(robot::file))
        .route("/api/robot/inspect", post(robot::inspect))
        .route("/api/robot/pack-link", post(robot::pack_link))
        .route("/api/robot/assemble", post(robot::assemble))
        .route("/api/robot/pack-live", get(robot::pack_live))
        .route("/api/robot/mesh-stats", get(robot::mesh_stats))
        .route("/api/robot/analyze", post(robot::analyze))
        .route("/api/robot/prepared-mesh", get(mesh::robot_prepared_mesh))
        // Uploads are read into memory like the Python service does; the
        // per-route caps (413 with advice) sit below this hard limit.
        .layer(DefaultBodyLimit::max(MAX_ROBOT_FOLDER_BYTES + 16 * MB))
        .with_state(state)
}

async fn healthz() -> Response {
    json_response(&serde_json::json!({ "ok": true }))
}

async fn index(State(state): State<SharedState>) -> ApiResult<Response> {
    let path = state.config.web_dir.join("index.html");
    let body = tokio::fs::read(&path)
        .await
        .map_err(|e| ApiError::internal(format!("cannot read {}: {e}", path.display())))?;
    Ok(bytes_response(body, "text/html; charset=utf-8", &[(header::CACHE_CONTROL, "no-store")]))
}

/// Compact JSON (`application/json`), as Starlette's `JSONResponse` writes it.
pub fn json_response<T: Serialize>(value: &T) -> Response {
    let body = serde_json::to_vec(value).expect("response serializes");
    bytes_response(body, "application/json", &[])
}

/// A body with a content type and extra static headers.
pub fn bytes_response(
    body: impl Into<axum::body::Body>,
    content_type: &str,
    extra: &[(HeaderName, &str)],
) -> Response {
    let mut headers = HeaderMap::new();
    headers.insert(header::CONTENT_TYPE, HeaderValue::from_str(content_type).expect("valid content type"));
    for (k, v) in extra {
        headers.insert(k.clone(), HeaderValue::from_str(v).expect("valid header value"));
    }
    (StatusCode::OK, headers, body.into()).into_response()
}

/// Serve a file like Starlette's `FileResponse`: the given media type or
/// one guessed from the extension (`text/plain` when unknown; text types get
/// `charset=utf-8`), plus optional `content-disposition` and headers.
pub async fn file_response(
    path: &Path,
    media_type: Option<&str>,
    download_name: Option<&str>,
    extra: &[(HeaderName, &str)],
) -> ApiResult<Response> {
    let body = tokio::fs::read(path)
        .await
        .map_err(|e| ApiError::internal(format!("cannot read {}: {e}", path.display())))?;
    let mut ct = match media_type {
        Some(m) => m.to_string(),
        None => match file_suffix(path.to_str()).as_str() {
            // mime_guess maps .stl to a certificate type; pin the robot formats.
            ".stl" => "model/stl".into(),
            ".obj" => "model/obj".into(),
            ".dae" => "model/vnd.collada+xml".into(),
            ".ply" => "application/octet-stream".into(),
            ".urdf" | ".xacro" => "application/xml".into(),
            _ => mime_guess::from_path(path).first_raw().unwrap_or("text/plain").to_string(),
        },
    };
    if ct.starts_with("text/") && !ct.contains("charset") {
        ct += "; charset=utf-8";
    }
    let mut r = bytes_response(body, &ct, extra);
    if let Some(name) = download_name {
        let v = format!("attachment; filename=\"{}\"", name.replace('"', "%22"));
        if let Ok(v) = HeaderValue::from_str(&v) {
            r.headers_mut().insert(header::CONTENT_DISPOSITION, v);
        }
    }
    Ok(r)
}

/// JSON for an HTTP header value, escaping every non-ASCII character as
/// `\uXXXX` (Python's `ensure_ascii=True`).
pub fn ascii_json<T: Serialize>(value: &T) -> String {
    let s = serde_json::to_string(value).expect("header value serializes");
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        if c.is_ascii() {
            out.push(c);
        } else {
            let mut buf = [0u16; 2];
            for u in c.encode_utf16(&mut buf) {
                out += &format!("\\u{u:04x}");
            }
        }
    }
    out
}

/// `Path(name).stem`, or `fallback` when empty.
pub fn file_stem(filename: Option<&str>, fallback: &str) -> String {
    filename
        .and_then(|f| Path::new(f).file_stem())
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| fallback.to_string())
}

/// `Path(name).suffix.lower()` (with the dot), or empty.
pub fn file_suffix(filename: Option<&str>) -> String {
    filename
        .and_then(|f| Path::new(f).extension())
        .map(|e| format!(".{}", e.to_string_lossy().to_lowercase()))
        .unwrap_or_default()
}

/// Run blocking work (optimization, mesh loading, file trees) off the
/// async workers.
pub async fn blocking<T: Send + 'static>(f: impl FnOnce() -> ApiResult<T> + Send + 'static) -> ApiResult<T> {
    tokio::task::spawn_blocking(f).await?
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn helpers() {
        let bs = '\\';
        assert_eq!(ascii_json(&"\u{e9} \u{1f600} a"), format!("\"{bs}u00e9 {bs}ud83d{bs}ude00 a\""));
        assert_eq!(file_stem(Some("dir/bunny.obj"), "object"), "bunny");
        assert_eq!(file_stem(Some(""), "object"), "object");
        assert_eq!(file_stem(None, "object"), "object");
        assert_eq!(file_suffix(Some("A.OBJ")), ".obj");
        assert_eq!(file_suffix(Some(".obj")), "");
        assert_eq!(file_suffix(Some("x.tar.gz")), ".gz");
    }
}
