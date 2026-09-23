//! In-memory robot-mode sessions, as in the Python API: a random hex id,
//! a work directory holding the uploaded package (`<root>/<id>/robot`) and
//! an output directory (`<root>/<id>/out`), an inactivity TTL enforced by a
//! sweep at the start of every robot endpoint, and the latest live frame of
//! the link being packed. One process owns its sessions, so run a single
//! replica.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, Instant};

use morphit_robot::inspect::InspectionReport;
use tokio::sync::watch;

use crate::error::{ApiError, ApiResult};

/// A live frame, already serialized as the JSON text sent on the websocket.
pub type LiveFrame = Arc<str>;

/// One robot-mode session.
pub struct RobotSession {
    pub id: String,
    pub work_dir: PathBuf,
    pub output_dir: PathBuf,
    report: Mutex<Option<Arc<InspectionReport>>>,
    last_access: Mutex<Instant>,
    /// Latest sphere state of the link being packed; the websocket follows it.
    pub live: watch::Sender<Option<LiveFrame>>,
    /// Cached `/api/robot/mesh-stats` response body.
    mesh_stats: Mutex<Option<Arc<str>>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|p| p.into_inner())
}

impl RobotSession {
    pub fn cached_mesh_stats(&self) -> Option<Arc<str>> {
        lock(&self.mesh_stats).clone()
    }

    pub fn cache_mesh_stats(&self, body: Arc<str>) {
        *lock(&self.mesh_stats) = Some(body);
    }

    pub fn report(&self) -> Option<Arc<InspectionReport>> {
        lock(&self.report).clone()
    }

    pub fn set_report(&self, report: InspectionReport) {
        *lock(&self.report) = Some(Arc::new(report));
    }

    /// The report, or the Python API's 400 when inspection has not run.
    pub fn require_report(&self, detail: &str) -> ApiResult<Arc<InspectionReport>> {
        self.report().ok_or_else(|| ApiError::bad_request(detail))
    }

    fn base(&self) -> &Path {
        self.work_dir.parent().expect("work dir has a parent")
    }
}

/// All sessions of this process.
pub struct SessionStore {
    root: PathBuf,
    ttl: Duration,
    sessions: Mutex<HashMap<String, Arc<RobotSession>>>,
}

impl SessionStore {
    pub fn new(root: PathBuf, ttl: Duration) -> Self {
        SessionStore { root, ttl, sessions: Mutex::new(HashMap::new()) }
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Drop sessions idle for longer than the TTL, with their directories.
    pub fn gc(&self) {
        let expired: Vec<Arc<RobotSession>> = {
            let mut map = lock(&self.sessions);
            let now = Instant::now();
            let ids: Vec<String> = map
                .iter()
                .filter(|(_, s)| now.duration_since(*lock(&s.last_access)) > self.ttl)
                .map(|(id, _)| id.clone())
                .collect();
            ids.iter().filter_map(|id| map.remove(id)).collect()
        };
        for s in expired {
            tracing::info!("robot session {} expired", s.id);
            let _ = std::fs::remove_dir_all(s.base());
        }
    }

    /// A fresh session with empty work and output directories.
    pub fn create(&self) -> ApiResult<Arc<RobotSession>> {
        let id = uuid::Uuid::new_v4().simple().to_string();
        let base = self.root.join(&id);
        let (work_dir, output_dir) = (base.join("robot"), base.join("out"));
        for d in [&work_dir, &output_dir] {
            std::fs::create_dir_all(d)
                .map_err(|e| ApiError::internal(format!("cannot create {}: {e}", d.display())))?;
        }
        let s = Arc::new(RobotSession {
            id: id.clone(),
            work_dir,
            output_dir,
            report: Mutex::new(None),
            last_access: Mutex::new(Instant::now()),
            live: watch::Sender::new(None),
            mesh_stats: Mutex::new(None),
        });
        lock(&self.sessions).insert(id, s.clone());
        Ok(s)
    }

    /// Look a session up and refresh its TTL; 404 when unknown or expired.
    pub fn get(&self, id: &str) -> ApiResult<Arc<RobotSession>> {
        let s = self.peek(id).ok_or_else(|| {
            ApiError::not_found(format!("session {} not found or expired", morphit_robot::py_repr(id)))
        })?;
        *lock(&s.last_access) = Instant::now();
        Ok(s)
    }

    /// Look a session up without touching its TTL.
    pub fn peek(&self, id: &str) -> Option<Arc<RobotSession>> {
        lock(&self.sessions).get(id).cloned()
    }

    /// Forget a session and delete its directories.
    pub fn remove(&self, id: &str) {
        if let Some(s) = lock(&self.sessions).remove(id) {
            let _ = std::fs::remove_dir_all(s.base());
        }
    }

    pub fn len(&self) -> usize {
        lock(&self.sessions).len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Drop for SessionStore {
    fn drop(&mut self) {
        let map = std::mem::take(&mut *lock(&self.sessions));
        for s in map.values() {
            let _ = std::fs::remove_dir_all(s.base());
        }
    }
}
