//! Opaque handle types and the session's concurrency machinery.
//!
//! Locking scheme for a session:
//! - `inner: Mutex<Session>` owns all optimizer state. It is held for one
//!   iteration at a time (or one finalize), never across a whole run and never
//!   while a user callback executes.
//! - `snapshot: RwLock<Arc<Snapshot>>` is republished after every iteration.
//!   All read-only API calls clone the `Arc` under a momentary read lock and
//!   work on that copy, so they never wait for an iteration to finish.
//! - `running` marks an active `morphit_run`; conflicting calls return BUSY.
//! - `cancel` is a request flag consumed by the run it stops.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

use morphit::{Config, LossId, Mesh, MeshPrepReport, PackResult, Session, SessionState, StepInfo};

use crate::ffi::{FfiError, FfiResult, morphit_status};

/// Loss term indices into `morphit_step_info::weighted_losses` / `raw_losses`.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum morphit_loss_id {
    MORPHIT_LOSS_COVERAGE = 0,
    MORPHIT_LOSS_OVERLAP = 1,
    MORPHIT_LOSS_BOUNDARY = 2,
    MORPHIT_LOSS_SURFACE = 3,
    MORPHIT_LOSS_CONTAINMENT = 4,
    MORPHIT_LOSS_SQEM = 5,
    MORPHIT_LOSS_HAUSDORFF = 6,
    MORPHIT_LOSS_MESH_CONTAINMENT = 7,
    MORPHIT_LOSS_MASS = 8,
    MORPHIT_LOSS_COM = 9,
    MORPHIT_LOSS_INERTIA = 10,
    /// Not implemented; always 0.
    MORPHIT_LOSS_FLATNESS = 11,
}

/// Number of entries in the per-loss arrays of `morphit_step_info`.
pub const MORPHIT_LOSS_COUNT: usize = 12;

const _: () = assert!(MORPHIT_LOSS_COUNT == LossId::COUNT);

/// Lifecycle of a session.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum morphit_session_state {
    /// Iterations remain.
    MORPHIT_STATE_RUNNING = 0,
    /// Stopped early by the convergence test (`training.early_stopping`).
    MORPHIT_STATE_CONVERGED = 1,
    /// All configured iterations ran.
    MORPHIT_STATE_COMPLETED = 2,
    /// The final escaped-sphere prune ran; the result is final.
    MORPHIT_STATE_FINALIZED = 3,
}

impl From<SessionState> for morphit_session_state {
    fn from(s: SessionState) -> Self {
        match s {
            SessionState::Running => morphit_session_state::MORPHIT_STATE_RUNNING,
            SessionState::Converged => morphit_session_state::MORPHIT_STATE_CONVERGED,
            SessionState::Completed => morphit_session_state::MORPHIT_STATE_COMPLETED,
            SessionState::Finalized => morphit_session_state::MORPHIT_STATE_FINALIZED,
        }
    }
}

/// Mesh properties.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct morphit_mesh_info {
    pub num_vertices: usize,
    pub num_faces: usize,
    pub volume: f64,
    pub area: f64,
    /// Length of the bounding-box diagonal.
    pub scale: f64,
    pub bounds_min: [f64; 3],
    pub bounds_max: [f64; 3],
    pub center_mass: [f64; 3],
    /// Inertia tensor about the center of mass at density 1, row-major.
    pub inertia: [f64; 9],
    /// Nonzero if the input was wound inward and all faces were flipped.
    pub winding_flipped: i32,
}

/// What one iteration did.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct morphit_step_info {
    /// Zero-based index of the iteration that ran.
    pub iteration: u64,
    pub total_loss: f64,
    /// `weight * value` per loss, indexed by `morphit_loss_id`.
    pub weighted_losses: [f64; MORPHIT_LOSS_COUNT],
    /// Unweighted value per loss, indexed by `morphit_loss_id`.
    pub raw_losses: [f64; MORPHIT_LOSS_COUNT],
    /// Mean per-sphere center-gradient norm before clipping.
    pub position_grad_mag: f64,
    /// Raw-radius gradient norm before clipping.
    pub radius_grad_mag: f64,
    pub num_spheres: usize,
    /// Centers projected back inside the mesh during this iteration.
    pub projected: usize,
    /// Nonzero if density control ran after this iteration.
    pub density_control_fired: i32,
    /// Spheres culled and reseeded by density control (0 if it did not run).
    pub density_control_replaced: usize,
    /// Nonzero when the session has no iterations left.
    pub done: i32,
    /// Nonzero when the convergence test stopped the run here.
    pub converged: i32,
    pub seconds: f64,
}

impl From<&StepInfo> for morphit_step_info {
    fn from(s: &StepInfo) -> Self {
        morphit_step_info {
            iteration: s.iteration as u64,
            total_loss: s.total_loss,
            weighted_losses: s.weighted_losses,
            raw_losses: s.raw_losses,
            position_grad_mag: s.position_grad_mag,
            radius_grad_mag: s.radius_grad_mag,
            num_spheres: s.num_spheres,
            projected: s.projected,
            density_control_fired: s.density_control.is_some() as i32,
            density_control_replaced: s.density_control.map_or(0, |d| d.removed),
            done: s.done as i32,
            converged: s.converged as i32,
            seconds: s.seconds,
        }
    }
}

/// Session progress, readable at any time from any thread.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct morphit_state_info {
    pub state: morphit_session_state,
    /// Iterations run so far.
    pub iteration: u64,
    pub total_iterations: u64,
    pub num_spheres: usize,
    /// Loss of the last iteration (NaN before the first).
    pub total_loss: f64,
    /// Nonzero while `morphit_run` is active.
    pub running: i32,
    pub density_control_passes: u64,
    /// Spheres removed by the final prune.
    pub pruned: usize,
    /// Seed that drove all sampling (drawn at random when the config had none).
    pub seed: u64,
}

/// Immutable mesh, shareable by any number of sessions and threads.
pub struct morphit_mesh {
    pub(crate) mesh: Arc<Mesh>,
}

/// Mutable configuration. Setters lock internally, so a config may be edited
/// and read from several threads; sessions copy it on creation.
pub struct morphit_config {
    pub(crate) config: Mutex<Config>,
}

impl morphit_config {
    pub(crate) fn lock(&self) -> MutexGuard<'_, Config> {
        // A config is only ever mutated through `Config::set`, which is atomic,
        // so a poisoned lock still guards a consistent value.
        self.config.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// Everything readers need, copied out of the session after each iteration.
#[derive(Debug)]
pub(crate) struct Snapshot {
    pub centers: Vec<f64>,
    pub radii: Vec<f64>,
    pub masses: Vec<f64>,
    pub state: SessionState,
    pub iteration: usize,
    pub total_iterations: usize,
    pub total_loss: f64,
    pub last_step: Option<morphit_step_info>,
    pub density_control_passes: usize,
    pub pruned: usize,
    pub seed: u64,
    pub per_sphere_mass: bool,
    pub config: Arc<Config>,
    pub mesh_prep: Arc<MeshPrepReport>,
}

impl Snapshot {
    pub fn of(s: &Session, config: &Arc<Config>, mesh_prep: &Arc<MeshPrepReport>) -> Snapshot {
        let sp = s.spheres();
        Snapshot {
            centers: sp.centers.iter().flat_map(|c| c.to_array()).collect(),
            radii: sp.radii(),
            masses: sp.masses(s.config().model.density),
            state: s.state(),
            iteration: s.iteration(),
            total_iterations: s.total_iterations(),
            total_loss: s.last_step().map_or(f64::NAN, |i| i.total_loss),
            last_step: s.last_step().map(morphit_step_info::from),
            density_control_passes: s.density_passes(),
            pruned: s.pruned(),
            seed: s.init_info().seed,
            per_sphere_mass: sp.per_sphere_mass(),
            config: Arc::clone(config),
            mesh_prep: Arc::clone(mesh_prep),
        }
    }

    pub fn result(&self) -> PackResult {
        PackResult {
            centers: self.centers.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            radii: self.radii.clone(),
            masses: self.masses.clone(),
            mesh_path: self.config.model.mesh_path.clone(),
            num_spheres: self.radii.len(),
            per_sphere_mass: self.per_sphere_mass,
            mesh_prep: Some((*self.mesh_prep).clone()),
            config: (*self.config).clone(),
        }
    }
}

/// A packing session. See the module docs for the locking scheme.
pub struct morphit_session {
    inner: Mutex<Session>,
    snapshot: RwLock<Arc<Snapshot>>,
    pub(crate) cancel: AtomicBool,
    pub(crate) running: AtomicBool,
    config: Arc<Config>,
    /// Resolved compute device (fixed for the session's lifetime).
    pub(crate) device: String,
    /// What mesh preparation did (fixed at creation).
    pub(crate) mesh_prep: Arc<MeshPrepReport>,
}

impl morphit_session {
    pub(crate) fn new(session: Session) -> Self {
        let config = Arc::new(session.config().clone());
        let mesh_prep = Arc::new(session.mesh_prep().clone());
        let snap = Snapshot::of(&session, &config, &mesh_prep);
        let device = session.device().to_string();
        morphit_session {
            inner: Mutex::new(session),
            snapshot: RwLock::new(Arc::new(snap)),
            cancel: AtomicBool::new(false),
            running: AtomicBool::new(false),
            config,
            device,
            mesh_prep,
        }
    }

    /// Lock the optimizer state. A panic during an earlier iteration poisons it.
    pub(crate) fn lock(&self) -> FfiResult<MutexGuard<'_, Session>> {
        self.inner.lock().map_err(|_| {
            FfiError::new(
                morphit_status::MORPHIT_ERR_STATE,
                "session is unusable after an internal error in an earlier call",
            )
        })
    }

    /// Current snapshot (never blocks on a running iteration).
    pub(crate) fn snapshot(&self) -> Arc<Snapshot> {
        Arc::clone(&self.snapshot.read().unwrap_or_else(|p| p.into_inner()))
    }

    /// Publish a new snapshot built while holding the session lock.
    pub(crate) fn publish(&self, session: &Session) {
        let snap = Arc::new(Snapshot::of(session, &self.config, &self.mesh_prep));
        *self.snapshot.write().unwrap_or_else(|p| p.into_inner()) = snap;
    }

    pub(crate) fn is_running(&self) -> bool {
        self.running.load(Ordering::Acquire)
    }
}

/// Clears `running` when a run ends, including by panic.
pub(crate) struct RunningGuard<'a>(pub &'a AtomicBool);

impl Drop for RunningGuard<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<morphit_mesh>();
    assert_send_sync::<morphit_config>();
    assert_send_sync::<morphit_session>();
    assert_send_sync::<Snapshot>();
};
