//! Error responses shaped like FastAPI's: `{"detail": ...}` with the status
//! code, compact JSON.

use axum::http::{HeaderValue, StatusCode, header};
use axum::response::{IntoResponse, Response};
use serde_json::Value;

/// One MiB, as the Python API counts it.
pub const MB: usize = 1024 * 1024;

/// FastAPI's validation error entry (key order `type, loc, msg, input`).
#[derive(serde::Serialize)]
struct ValidationItem<'a> {
    #[serde(rename = "type")]
    kind: &'a str,
    loc: [&'a str; 2],
    msg: &'a str,
    input: Value,
}

/// An error that renders as FastAPI's `HTTPException`.
#[derive(Debug)]
pub struct ApiError {
    pub status: StatusCode,
    /// The `detail` value, already serialized (keeps key order).
    pub detail: String,
}

impl ApiError {
    pub fn new(status: StatusCode, detail: impl Into<String>) -> Self {
        let detail = serde_json::to_string(&detail.into()).expect("string serializes");
        ApiError { status, detail }
    }

    pub fn bad_request(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::BAD_REQUEST, detail)
    }

    pub fn not_found(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::NOT_FOUND, detail)
    }

    pub fn forbidden(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::FORBIDDEN, detail)
    }

    pub fn internal(detail: impl Into<String>) -> Self {
        Self::new(StatusCode::INTERNAL_SERVER_ERROR, detail)
    }

    /// FastAPI's request-validation error (422) for one field.
    pub fn validation(loc: &str, field: &str, kind: &str, msg: &str, input: Value) -> Self {
        ApiError {
            status: StatusCode::UNPROCESSABLE_ENTITY,
            detail: serde_json::to_string(&[ValidationItem { kind, loc: [loc, field], msg, input }])
                .expect("serializes"),
        }
    }

    /// A required field is missing.
    pub fn missing(loc: &str, field: &str) -> Self {
        Self::validation(loc, field, "missing", "Field required", Value::Null)
    }

    /// 413 with the Python API's advice on shrinking the mesh.
    pub fn too_large(actual: usize, limit: usize, kind: &str) -> Self {
        let mb = MB as f64;
        Self::new(
            StatusCode::PAYLOAD_TOO_LARGE,
            format!(
                "{kind} too large: {:.1} MB exceeds the {:.0} MB limit. Reduce the file size of the heaviest \
                 mesh by decimating it (e.g. trimesh's simplify_quadric_decimation, MeshLab's Quadric Edge \
                 Collapse, or Blender's Decimate modifier) or by replacing it with a convex decomposition \
                 (V-HACD, CoACD).",
                actual as f64 / mb,
                limit as f64 / mb
            ),
        )
    }
}

impl From<morphit_robot::Error> for ApiError {
    fn from(e: morphit_robot::Error) -> Self {
        match e {
            morphit_robot::Error::Invalid(m) => ApiError::bad_request(m),
            morphit_robot::Error::Morphit(morphit::Error::Mesh(m)) => ApiError::bad_request(m),
            other => ApiError::internal(other.to_string()),
        }
    }
}

impl From<morphit::Error> for ApiError {
    fn from(e: morphit::Error) -> Self {
        morphit_robot::Error::from(e).into()
    }
}

impl From<tokio::task::JoinError> for ApiError {
    fn from(e: tokio::task::JoinError) -> Self {
        tracing::error!("worker task failed: {e}");
        ApiError::internal("Internal Server Error")
    }
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let body = format!("{{\"detail\":{}}}", self.detail);
        let mut r = (self.status, body).into_response();
        r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("application/json"));
        r
    }
}

/// Handler result.
pub type ApiResult<T> = Result<T, ApiError>;
