//! Robot mode: a session holds an uploaded (or bundled) URDF package; its
//! collision meshes are packed one link at a time, followed live over a
//! websocket, then assembled into a spherical URDF and scored.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use axum::extract::multipart::MultipartRejection;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Multipart, Path as UrlPath, Query, State};
use axum::response::{IntoResponse, Response};
use morphit::Mesh;
use morphit_robot::assemble::{load_spheres_str, rewrite_urdf};
use morphit_robot::color::safe_color_rgba;
use morphit_robot::config::{PackParams, parse_advanced};
use morphit_robot::history::{MAX_POINTS, subsample_history};
use morphit_robot::inspect::{
    Action, InspectionReport, find_urdfs, inspect_urdf, resolve_mesh_path, select_urdf,
};
use morphit_robot::pack::{
    PackLinkOutcome, json_filename, pack_one_link, result_uses_mesh_prep, spheres_dir,
};
use morphit_robot::paths::{copy_tree, is_within, safe_join};
use morphit_robot::py_repr;
use morphit_robot::quality::{mc_interior_samples, quality_metrics};
use serde::Serialize;

use crate::error::{ApiError, ApiResult};
use crate::examples::find_robot;
use crate::forms::{Form, count, seed};
use crate::morph::analyze_response;
use crate::sessions::RobotSession;
use crate::{
    MAX_ROBOT_FOLDER_BYTES, MAX_ROBOT_PER_FILE_BYTES, SharedState, blocking, bytes_response, file_response,
    json_response,
};

const NO_REPORT: &str = "session has no inspection report";
const NO_REPORT_INSPECT: &str = "session has no inspection report; call /inspect first";
/// Minimum time between two live frames of one pack.
const LIVE_INTERVAL: Duration = Duration::from_millis(80);

/// `{"session_id": ..., <report fields>}`.
#[derive(Serialize)]
struct SessionReport<'a> {
    session_id: &'a str,
    #[serde(flatten)]
    report: &'a InspectionReport,
}

fn session_report(session: &RobotSession, report: &InspectionReport) -> Response {
    json_response(&SessionReport { session_id: &session.id, report })
}

fn query<'a>(q: &'a HashMap<String, String>, name: &str) -> ApiResult<&'a str> {
    q.get(name).map(String::as_str).ok_or_else(|| ApiError::missing("query", name))
}

/// `POST /api/robot/example/{name}`: start a session from a bundled robot.
pub async fn example_load(
    State(state): State<SharedState>,
    UrlPath(name): UrlPath<String>,
) -> ApiResult<Response> {
    let robot = find_robot(&name)?;
    let folder = state.examples_dir().join(robot.folder);
    if !folder.exists() {
        return Err(ApiError::internal(format!(
            "robot {} not bundled; expected {}",
            py_repr(&name),
            folder.display()
        )));
    }
    state.sessions.gc();
    let session = state.sessions.create()?;
    let s = session.clone();
    let report = blocking(move || {
        copy_tree(&folder, &s.work_dir.join(robot.folder))
            .map_err(|e| ApiError::internal(format!("cannot copy {}: {e}", folder.display())))?;
        let urdf = select_urdf(&find_urdfs(&s.work_dir), Some(robot.urdf))
            .map_err(|e| ApiError::internal(format!("robot {} malformed: {e}", py_repr(robot.name))))?;
        Ok(inspect_urdf(&s.work_dir, &urdf)?)
    })
    .await?;
    let resp = session_report(&session, &report);
    session.set_report(report);
    Ok(resp)
}

/// `GET /api/robot/file?session_id=&path=`: a file of the session's package,
/// by `package://` URI, session-relative path, or absolute path inside the
/// session.
pub async fn file(
    State(state): State<SharedState>,
    Query(q): Query<HashMap<String, String>>,
) -> ApiResult<Response> {
    let id = query(&q, "session_id")?;
    let path = query(&q, "path")?;
    state.sessions.gc();
    let session = state.sessions.get(id)?;
    let resolved: PathBuf = if path.starts_with("package://") {
        let urdf = session.report().map(|r| PathBuf::from(&r.urdf_path)).unwrap_or(session.work_dir.clone());
        resolve_mesh_path(path, &session.work_dir, &urdf)
            .ok_or_else(|| ApiError::not_found(format!("could not resolve {}", py_repr(path))))?
    } else if Path::new(path).is_absolute() {
        PathBuf::from(path)
    } else {
        safe_join(&session.work_dir, path)?
    };
    if !resolved.exists() {
        return Err(ApiError::not_found(format!("file not found: {}", py_repr(path))));
    }
    if !is_within(&resolved, &session.work_dir) {
        return Err(ApiError::forbidden("path escapes session sandbox"));
    }
    file_response(&resolved, None, None, &[]).await
}

/// `POST /api/robot/inspect`: repeated `files` parts whose filenames are the
/// paths inside the package, plus an optional `urdf` basename.
pub async fn inspect(
    State(state): State<SharedState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> ApiResult<Response> {
    let form = Form::read(multipart).await?;
    let files = form.files("files")?;
    let requested = form.text("urdf").map(str::to_string);
    state.sessions.gc();
    let session = state.sessions.create()?;

    let mut writes = Vec::with_capacity(files.len());
    let mut total = 0usize;
    for f in files {
        let rel = f.filename.as_deref().unwrap_or("");
        let target = safe_join(&session.work_dir, rel)?;
        let size = f.data.len();
        if size > MAX_ROBOT_PER_FILE_BYTES {
            state.sessions.remove(&session.id);
            return Err(ApiError::too_large(
                size,
                MAX_ROBOT_PER_FILE_BYTES,
                &format!("File {}", py_repr(rel)),
            ));
        }
        total += size;
        if total > MAX_ROBOT_FOLDER_BYTES {
            state.sessions.remove(&session.id);
            return Err(ApiError::too_large(total, MAX_ROBOT_FOLDER_BYTES, "Folder"));
        }
        writes.push((target, f.data.clone()));
    }
    let s = session.clone();
    let report = blocking(move || {
        for (target, data) in writes {
            let write = || -> std::io::Result<()> {
                if let Some(parent) = target.parent() {
                    std::fs::create_dir_all(parent)?;
                }
                std::fs::write(&target, &data)
            };
            write().map_err(|e| ApiError::internal(format!("cannot write {}: {e}", target.display())))?;
        }
        let urdf = select_urdf(&find_urdfs(&s.work_dir), requested.as_deref())
            .map_err(|e| ApiError::bad_request(e.to_string()))?;
        Ok(inspect_urdf(&s.work_dir, &urdf)?)
    })
    .await?;
    let resp = session_report(&session, &report);
    session.set_report(report);
    Ok(resp)
}

/// One live frame on the `pack-live` websocket.
#[derive(Serialize)]
struct LiveFrameJson<'a> {
    link_name: &'a str,
    collision_index: usize,
    iteration: usize,
    total_iterations: usize,
    centers: Vec<[f64; 3]>,
    radii: Vec<f64>,
    loss: f64,
}

#[derive(Serialize)]
struct PackLinkResponse {
    #[serde(flatten)]
    outcome: PackLinkOutcome,
    loss_history: Vec<(usize, f64)>,
}

/// `POST /api/robot/pack-link`: pack one collision of the session's robot.
pub async fn pack_link(
    State(state): State<SharedState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> ApiResult<Response> {
    let form = Form::read(multipart).await?;
    let id = form.required_text("session_id")?.to_string();
    let link_name = form.required_text("link_name")?.to_string();
    let collision_index = form.required_int("collision_index")?;
    let (n, iters, seed_raw) =
        (form.int_or("num_spheres", 20)?, form.int_or("iterations", 200)?, form.int("seed")?);
    state.sessions.gc();
    let session = state.sessions.get(&id)?;
    let report = session.require_report(NO_REPORT_INSPECT)?;
    let mut params = PackParams {
        variant: form.text_or("variant", "MorphIt-B").to_string(),
        num_spheres: count(n),
        iterations: count(iters),
        seed: None,
        advanced: Vec::new(),
        union_overlapping_bodies: form.bool_or("union_overlapping_bodies", true)?,
    };
    params.validate()?;
    let item = report
        .collisions
        .iter()
        .find(|c| c.link_name == link_name && c.collision_index as i64 == collision_index)
        .ok_or_else(|| ApiError::not_found(format!("no collision item: {link_name}[{collision_index}]")))?
        .clone();
    if item.action != Action::Pack {
        let action = serde_json::to_value(item.action).expect("action serializes");
        return Err(ApiError::bad_request(format!(
            "item {link_name}[{collision_index}] has action {}, not 'pack'",
            py_repr(action.as_str().unwrap_or_default())
        )));
    }
    params.advanced = parse_advanced(form.text_or("advanced", "{}"))?;
    params.seed = seed(seed_raw)?;
    let device = state.config.device.clone();

    let s = session.clone();
    let (outcome, history) = blocking(move || {
        let mut last: Option<Instant> = None;
        let total_iterations = params.iterations;
        let (outcome, _, history) = pack_one_link(&item, &params, &device, &s.output_dir, |sess, info| {
            if last.is_some_and(|t| t.elapsed() < LIVE_INTERVAL) {
                return;
            }
            last = Some(Instant::now());
            let frame = LiveFrameJson {
                link_name: &item.link_name,
                collision_index: item.collision_index,
                iteration: info.iteration,
                total_iterations,
                centers: sess.spheres().centers.iter().map(|c| c.to_array()).collect(),
                radii: sess.spheres().radii(),
                loss: info.total_loss,
            };
            let text = serde_json::to_string(&frame).expect("frame serializes");
            s.live.send_replace(Some(Arc::from(text)));
        })?;
        Ok((outcome, history))
    })
    .await?;
    Ok(json_response(&PackLinkResponse { outcome, loss_history: subsample_history(&history, MAX_POINTS) }))
}

/// `POST /api/robot/assemble`: the spherical URDF of everything packed.
pub async fn assemble(
    State(state): State<SharedState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> ApiResult<Response> {
    let form = Form::read(multipart).await?;
    let id = form.required_text("session_id")?.to_string();
    let base_color = safe_color_rgba(Some(form.text_or("base_color", "#3399ff")));
    let variation = form.float_or("color_variation", 0.3)?;
    state.sessions.gc();
    let session = state.sessions.get(&id)?;
    let report = session.require_report(NO_REPORT)?;
    if !(0.0..=1.0).contains(&variation) {
        return Err(ApiError::bad_request("color_variation must be in [0, 1]"));
    }
    let stem = Path::new(&report.urdf_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "robot".into());
    let output = session.output_dir.join(format!("{stem}_spherical.urdf"));
    let spheres = spheres_dir(&session.output_dir);
    let (text, stats) =
        blocking(move || Ok(rewrite_urdf(&report, &spheres, &output, base_color, variation)?)).await?;
    if !stats.skipped_pack_items.is_empty() {
        return Err(ApiError::bad_request(format!(
            "Some collisions weren't packed: {}",
            stats.skipped_summary()
        )));
    }
    Ok(bytes_response(text, "application/xml", &[]))
}

/// `GET /api/robot/pack-live?session_id=`: forwards each new live frame of
/// the session as a text message. Client messages are ignored; the socket
/// closes when the client leaves or the session is gone.
pub async fn pack_live(
    ws: WebSocketUpgrade,
    State(state): State<SharedState>,
    Query(q): Query<HashMap<String, String>>,
) -> ApiResult<Response> {
    let id = query(&q, "session_id")?.to_string();
    Ok(ws.on_upgrade(move |socket| follow_live(socket, state, id)).into_response())
}

async fn follow_live(mut socket: WebSocket, state: SharedState, id: String) {
    let Some(session) = state.sessions.peek(&id) else {
        let _ = socket.send(Message::Close(None)).await;
        return;
    };
    let mut rx = session.live.subscribe();
    // The frame present at connect time counts as seen, so a reconnect does
    // not replay the previous run's last frame.
    rx.borrow_and_update();
    drop(session);
    let mut check = tokio::time::interval(Duration::from_secs(1));
    loop {
        tokio::select! {
            changed = rx.changed() => {
                if changed.is_err() {
                    break;
                }
                let frame = rx.borrow_and_update().clone();
                if let Some(text) = frame
                    && socket.send(Message::Text(text.as_ref().into())).await.is_err()
                {
                    break;
                }
            }
            msg = socket.recv() => match msg {
                None | Some(Err(_)) | Some(Ok(Message::Close(_))) => break,
                Some(Ok(_)) => {}
            },
            _ = check.tick() => {
                if state.sessions.peek(&id).is_none() {
                    break;
                }
            }
        }
    }
    let _ = socket.send(Message::Close(None)).await;
}

/// One `/api/robot/mesh-stats` item.
#[derive(Serialize)]
struct MeshStat {
    link_name: String,
    collision_index: usize,
    #[serde(flatten)]
    value: StatValue,
}

#[derive(Serialize)]
#[serde(untagged)]
enum StatValue {
    Ok { area: f64, volume: f64 },
    Err { error: String },
}

/// `GET /api/robot/mesh-stats?session_id=`: area and Monte-Carlo volume of
/// every pack item's (prepared) mesh, cached per session.
pub async fn mesh_stats(
    State(state): State<SharedState>,
    Query(q): Query<HashMap<String, String>>,
) -> ApiResult<Response> {
    let id = query(&q, "session_id")?;
    state.sessions.gc();
    let session = state.sessions.get(id)?;
    let report = session.require_report(NO_REPORT_INSPECT)?;
    if let Some(body) = session.cached_mesh_stats() {
        return Ok(bytes_response(body.to_string(), "application/json", &[]));
    }
    let items = blocking(move || {
        Ok(report
            .to_pack()
            .map(|item| {
                let value = item
                    .mesh_path
                    .as_deref()
                    .ok_or_else(|| "no resolved mesh path".to_string())
                    .and_then(|p| Mesh::load(p).map_err(|e| e.to_string()))
                    .map(|m| {
                        let (m, _) = Arc::new(m).prepared();
                        let (_, volume) = mc_interior_samples(&m, 0);
                        StatValue::Ok { area: m.area(), volume }
                    })
                    .unwrap_or_else(|error| StatValue::Err { error });
                MeshStat { link_name: item.link_name.clone(), collision_index: item.collision_index, value }
            })
            .collect::<Vec<_>>())
    })
    .await?;
    #[derive(Serialize)]
    struct Items {
        items: Vec<MeshStat>,
    }
    let body: Arc<str> = serde_json::to_string(&Items { items }).expect("serializes").into();
    session.cache_mesh_stats(body.clone());
    Ok(bytes_response(body.to_string(), "application/json", &[]))
}

/// `POST /api/robot/analyze`: quality metrics of every packed link.
pub async fn analyze(
    State(state): State<SharedState>,
    multipart: Result<Multipart, MultipartRejection>,
) -> ApiResult<Response> {
    let form = Form::read(multipart).await?;
    let id = form.required_text("session_id")?.to_string();
    state.sessions.gc();
    let session = state.sessions.get(&id)?;
    let report = session.require_report(NO_REPORT)?;
    let dir = spheres_dir(&session.output_dir);
    let pack: Vec<_> = report.to_pack().cloned().collect();
    if pack.is_empty() {
        return Err(ApiError::bad_request("nothing was packed in this session"));
    }
    let todo: Vec<_> = pack
        .into_iter()
        .map(|item| {
            let p = dir.join(json_filename(&item.link_name, item.collision_index));
            (item, p)
        })
        .filter(|(_, p)| p.exists())
        .collect();
    if todo.is_empty() {
        return Err(ApiError::bad_request(
            "No packed links found in this session. Run MorphIt first, then analyze.",
        ));
    }
    let links = blocking(move || {
        todo.iter()
            .map(|(item, json)| {
                let text = std::fs::read_to_string(json)
                    .map_err(|e| ApiError::internal(format!("cannot read {}: {e}", json.display())))?;
                let (centers, radii) = load_spheres_str(&text, &json.display().to_string())?;
                let path = item.mesh_path.as_deref().unwrap_or_default();
                // Score against the mesh the link was packed on.
                let mesh = Arc::new(Mesh::load(path)?);
                let mesh = if result_uses_mesh_prep(&text) { mesh.prepared().0 } else { mesh };
                Ok(quality_metrics(&mesh, &item.link_name, item.collision_index, &centers, &radii))
            })
            .collect::<ApiResult<Vec<_>>>()
    })
    .await?;
    analyze_response(links)
}
