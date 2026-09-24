//! Multipart form handling with FastAPI's semantics: empty values count as
//! absent, missing required fields and unparsable numbers are 422s.

use axum::body::Bytes;
use axum::extract::Multipart;
use axum::extract::multipart::MultipartRejection;
use axum::http::StatusCode;
use serde_json::Value;

use crate::error::{ApiError, ApiResult};

/// An uploaded file part.
#[derive(Debug, Clone)]
pub struct Upload {
    pub filename: Option<String>,
    pub data: Bytes,
}

/// All parts of a `multipart/form-data` body.
#[derive(Debug, Default)]
pub struct Form {
    text: Vec<(String, String)>,
    files: Vec<(String, Upload)>,
}

impl Form {
    /// Read every part. A body that is not multipart at all reads as an
    /// empty form, so the handler reports the missing fields (422) like
    /// FastAPI does.
    pub async fn read(multipart: Result<Multipart, MultipartRejection>) -> ApiResult<Form> {
        let mut form = Form::default();
        let Ok(mut mp) = multipart else { return Ok(form) };
        loop {
            let field = mp.next_field().await.map_err(|e| ApiError::new(e.status(), e.body_text()))?;
            let Some(field) = field else { break };
            let name = field.name().unwrap_or_default().to_string();
            match field.file_name().map(str::to_string) {
                Some(filename) => {
                    let data = field.bytes().await.map_err(|e| ApiError::new(e.status(), e.body_text()))?;
                    form.files.push((name, Upload { filename: Some(filename), data }));
                }
                None => {
                    let text = field.text().await.map_err(|e| ApiError::new(e.status(), e.body_text()))?;
                    form.text.push((name, text));
                }
            }
        }
        Ok(form)
    }

    /// First non-empty text value of `name`.
    pub fn text(&self, name: &str) -> Option<&str> {
        self.text.iter().find(|(n, v)| n == name && !v.is_empty()).map(|(_, v)| v.as_str())
    }

    pub fn text_or<'a>(&'a self, name: &str, default: &'a str) -> &'a str {
        self.text(name).unwrap_or(default)
    }

    pub fn required_text(&self, name: &str) -> ApiResult<&str> {
        self.text(name).ok_or_else(|| ApiError::missing("body", name))
    }

    /// Optional integer field.
    pub fn int(&self, name: &str) -> ApiResult<Option<i64>> {
        match self.text(name) {
            None => Ok(None),
            Some(v) => v.trim().parse::<i64>().map(Some).map_err(|_| {
                ApiError::validation(
                    "body",
                    name,
                    "int_parsing",
                    "Input should be a valid integer, unable to parse string as an integer",
                    Value::String(v.into()),
                )
            }),
        }
    }

    pub fn int_or(&self, name: &str, default: i64) -> ApiResult<i64> {
        Ok(self.int(name)?.unwrap_or(default))
    }

    pub fn required_int(&self, name: &str) -> ApiResult<i64> {
        self.int(name)?.ok_or_else(|| ApiError::missing("body", name))
    }

    pub fn float_or(&self, name: &str, default: f64) -> ApiResult<f64> {
        match self.text(name) {
            None => Ok(default),
            Some(v) => v.trim().parse::<f64>().map_err(|_| {
                ApiError::validation(
                    "body",
                    name,
                    "float_parsing",
                    "Input should be a valid number, unable to parse string as a number",
                    Value::String(v.into()),
                )
            }),
        }
    }

    /// Optional boolean field, parsed as FastAPI does (`true/false`, `1/0`,
    /// `yes/no`, `on/off`, ...; 422 otherwise).
    pub fn bool_or(&self, name: &str, default: bool) -> ApiResult<bool> {
        self.text(name).map_or(Ok(default), |v| parse_bool("body", name, v))
    }

    /// The first file part named `name` (422 when absent).
    pub fn file(&self, name: &str) -> ApiResult<&Upload> {
        self.files
            .iter()
            .find(|(n, _)| n == name)
            .map(|(_, f)| f)
            .ok_or_else(|| ApiError::missing("body", name))
    }

    /// Every file part named `name` (422 when there is none).
    pub fn files(&self, name: &str) -> ApiResult<Vec<&Upload>> {
        let v: Vec<&Upload> = self.files.iter().filter(|(n, _)| n == name).map(|(_, f)| f).collect();
        if v.is_empty() { Err(ApiError::missing("body", name)) } else { Ok(v) }
    }
}

/// Validate an integer that must be a non-negative count.
pub fn count(v: i64) -> usize {
    usize::try_from(v).unwrap_or(0)
}

/// `Optional[int]` seed: negative values have no `u64` equivalent.
pub fn seed(v: Option<i64>) -> ApiResult<Option<u64>> {
    match v {
        None => Ok(None),
        Some(s) => u64::try_from(s).map(Some).map_err(|_| {
            ApiError::new(StatusCode::BAD_REQUEST, format!("seed must be a non-negative integer, got {s}"))
        }),
    }
}

/// A boolean as FastAPI parses it (`true/false`, `1/0`, `yes/no`, `on/off`,
/// ...); anything else is a 422 at `loc` (`body` or `query`).
pub fn parse_bool(loc: &str, name: &str, v: &str) -> ApiResult<bool> {
    match v.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "t" | "yes" | "y" | "on" => Ok(true),
        "0" | "false" | "f" | "no" | "n" | "off" => Ok(false),
        _ => Err(ApiError::validation(
            loc,
            name,
            "bool_parsing",
            "Input should be a valid boolean, unable to interpret input",
            Value::String(v.into()),
        )),
    }
}
