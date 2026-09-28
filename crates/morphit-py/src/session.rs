//! `Session`: step-by-step packing with the GIL released, a snapshot that is
//! always readable, and thread-safe cancellation. The locking scheme is the C
//! API's: the optimizer lock is held for one step at a time, readers use the
//! snapshot published after every step, and a second run is refused (BusyError)
//! instead of blocking.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, RwLock};

use morphit::{Config, LossId, Mesh, MeshPrepReport, PackResult, Session, SessionState, StepInfo};
use numpy::{PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::config::PyConfig;
use crate::convert::{PointSets, dvec3s, flat3, to_py, vec1};
use crate::errors::{self, BusyError, StateError};
use crate::mesh::PyMesh;
use crate::result::PyPackResult;

/// Everything readers need, copied out of the session after each step.
struct Snapshot {
    centers: Vec<f64>,
    radii: Vec<f64>,
    masses: Vec<f64>,
    state: SessionState,
    iteration: usize,
    total_iterations: usize,
    per_sphere_mass: bool,
    last_step: Option<StepInfo>,
    density_passes: usize,
    pruned: usize,
    gpu_error: Option<String>,
}

impl Snapshot {
    fn of(s: &Session) -> Snapshot {
        let sp = s.spheres();
        Snapshot {
            centers: sp.centers.iter().flat_map(|c| c.to_array()).collect(),
            radii: sp.radii(),
            masses: sp.masses(s.config().model.density),
            state: s.state(),
            iteration: s.iteration(),
            total_iterations: s.total_iterations(),
            per_sphere_mass: sp.per_sphere_mass(),
            last_step: s.last_step().cloned(),
            density_passes: s.density_passes(),
            pruned: s.pruned(),
            gpu_error: s.gpu_error(),
        }
    }
}

pub(crate) struct Core {
    session: Mutex<Session>,
    snapshot: RwLock<Arc<Snapshot>>,
    cancel: AtomicBool,
    running: AtomicBool,
    config: Arc<Config>,
    mesh_prep: Arc<MeshPrepReport>,
    mesh: Arc<Mesh>,
    device: String,
    init: morphit::InitInfo,
}

impl Core {
    fn new(s: Session) -> Core {
        Core {
            config: Arc::new(s.config().clone()),
            mesh_prep: Arc::new(s.mesh_prep().clone()),
            mesh: Arc::clone(s.mesh()),
            device: s.device().to_string(),
            init: s.init_info(),
            snapshot: RwLock::new(Arc::new(Snapshot::of(&s))),
            session: Mutex::new(s),
            cancel: AtomicBool::new(false),
            running: AtomicBool::new(false),
        }
    }

    fn lock(&self) -> PyResult<MutexGuard<'_, Session>> {
        self.session.lock().map_err(|_| {
            StateError::new_err("session is unusable after an internal error in an earlier call")
        })
    }

    fn snapshot(&self) -> Arc<Snapshot> {
        Arc::clone(&self.snapshot.read().unwrap_or_else(|p| p.into_inner()))
    }

    fn publish(&self, s: &Session) {
        *self.snapshot.write().unwrap_or_else(|p| p.into_inner()) = Arc::new(Snapshot::of(s));
    }

    /// Mark the session running; a guard clears the flag again (also on errors).
    fn claim(&self) -> PyResult<Running<'_>> {
        self.running
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .map(|_| Running(&self.running))
            .map_err(|_| BusyError::new_err("the session is running on another thread"))
    }

    fn result(&self) -> PackResult {
        let snap = self.snapshot();
        PackResult {
            centers: snap.centers.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(),
            radii: snap.radii.clone(),
            masses: snap.masses.clone(),
            mesh_path: self.config.model.mesh_path.clone(),
            num_spheres: snap.radii.len(),
            per_sphere_mass: snap.per_sphere_mass,
            mesh_prep: Some((*self.mesh_prep).clone()),
            config: (*self.config).clone(),
        }
    }
}

struct Running<'a>(&'a AtomicBool);

impl Drop for Running<'_> {
    fn drop(&mut self) {
        self.0.store(false, Ordering::Release);
    }
}

/// One step with the GIL released. `None` when the session has no iterations left.
fn step_once(py: Python<'_>, core: &Arc<Core>) -> PyResult<Option<StepInfo>> {
    let core = Arc::clone(core);
    py.detach(move || -> PyResult<Option<StepInfo>> {
        let mut s = core.lock()?;
        if s.is_done() {
            return Ok(None);
        }
        let info = s.step().map_err(errors::core)?;
        core.publish(&s);
        Ok(Some(info))
    })
}

/// What happened during one optimizer iteration.
#[pyclass(name = "StepInfo", module = "morphit_rs", frozen)]
pub struct PyStepInfo {
    inner: StepInfo,
}

impl PyStepInfo {
    pub(crate) fn new(inner: StepInfo) -> Self {
        PyStepInfo { inner }
    }
}

fn losses<'py>(py: Python<'py>, v: &[f64; LossId::COUNT]) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    for id in LossId::ALL {
        d.set_item(id.name(), v[id.index()])?;
    }
    Ok(d)
}

#[pymethods]
impl PyStepInfo {
    /// Zero-based index of the iteration that just ran.
    #[getter]
    fn iteration(&self) -> usize {
        self.inner.iteration
    }
    #[getter]
    fn total_loss(&self) -> f64 {
        self.inner.total_loss
    }
    /// `weight * value` per loss (`coverage_loss`, `overlap_penalty`, ...).
    #[getter]
    fn weighted_losses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        losses(py, &self.inner.weighted_losses)
    }
    /// Unweighted value per loss.
    #[getter]
    fn raw_losses<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        losses(py, &self.inner.raw_losses)
    }
    #[getter]
    fn position_grad_mag(&self) -> f64 {
        self.inner.position_grad_mag
    }
    #[getter]
    fn radius_grad_mag(&self) -> f64 {
        self.inner.radius_grad_mag
    }
    #[getter]
    fn num_spheres(&self) -> usize {
        self.inner.num_spheres
    }
    /// Centers moved back inside the mesh after the update.
    #[getter]
    fn projected(&self) -> usize {
        self.inner.projected
    }
    /// `{"added", "removed", "bad"}` when density control ran after this step.
    #[getter]
    fn density_control<'py>(&self, py: Python<'py>) -> PyResult<Option<Bound<'py, PyDict>>> {
        self.inner
            .density_control
            .map(|d| {
                let out = PyDict::new(py);
                out.set_item("added", d.added)?;
                out.set_item("removed", d.removed)?;
                out.set_item("bad", d.bad)?;
                Ok(out)
            })
            .transpose()
    }
    #[getter]
    fn done(&self) -> bool {
        self.inner.done
    }
    #[getter]
    fn converged(&self) -> bool {
        self.inner.converged
    }
    /// Wall time of the step.
    #[getter]
    fn seconds(&self) -> f64 {
        self.inner.seconds
    }
    fn __repr__(&self) -> String {
        format!(
            "StepInfo(iteration={}, total_loss={:.6}, num_spheres={})",
            self.inner.iteration, self.inner.total_loss, self.inner.num_spheres
        )
    }
}

fn state_name(s: SessionState) -> &'static str {
    match s {
        SessionState::Running => "running",
        SessionState::Converged => "converged",
        SessionState::Completed => "completed",
        SessionState::Finalized => "finalized",
    }
}

/// A packing run you can step, iterate, observe from other threads and cancel.
///
/// ```python
/// s = Session(mesh, config)
/// for step in s:              # Ctrl+C stops between steps; the session stays usable
///     print(step.iteration, step.total_loss)
/// s.finalize()
/// result = s.result()
/// ```
#[pyclass(name = "Session", module = "morphit_rs", frozen)]
pub struct PySession {
    core: Arc<Core>,
}

impl PySession {
    pub(crate) fn from_session(s: Session) -> Self {
        PySession { core: Arc::new(Core::new(s)) }
    }

    /// The run loop behind `run` (and `RobotPackage.pack_all`): steps with the
    /// GIL released, `on_step` with it held (`false` stops), finalize at the end.
    pub(crate) fn run_with(
        &self,
        py: Python<'_>,
        mut on_step: impl FnMut(Python<'_>, StepInfo) -> PyResult<bool>,
    ) -> PyResult<&'static str> {
        let _running = self.core.claim()?;
        loop {
            if self.core.cancel.swap(false, Ordering::AcqRel) {
                return Ok("cancelled");
            }
            py.check_signals()?;
            let Some(info) = step_once(py, &self.core)? else { break };
            if !on_step(py, info)? {
                return Ok("cancelled");
            }
        }
        let core = Arc::clone(&self.core);
        let state = py.detach(move || -> PyResult<SessionState> {
            let mut s = core.lock()?;
            let state = s.state();
            s.finalize();
            core.publish(&s);
            Ok(state)
        })?;
        self.core.cancel.store(false, Ordering::Release);
        Ok(if state == SessionState::Converged { "converged" } else { "completed" })
    }

    pub(crate) fn result_inner(&self) -> PackResult {
        self.core.result()
    }
}

#[pymethods]
impl PySession {
    /// Prepare the mesh, draw the samples and place the initial spheres.
    #[new]
    fn new(py: Python<'_>, mesh: &PyMesh, config: &PyConfig) -> PyResult<Self> {
        let (mesh, config) = (Arc::clone(&mesh.inner), config.inner.clone());
        let s = py.detach(move || Session::new(config, mesh)).map_err(errors::core)?;
        Ok(Self::from_session(s))
    }

    /// Run one iteration (the GIL is released while it runs).
    fn step(&self, py: Python<'_>) -> PyResult<PyStepInfo> {
        let _running = self.core.claim()?;
        match step_once(py, &self.core)? {
            Some(inner) => Ok(PyStepInfo { inner }),
            None => Err(StateError::new_err("the session has no iterations left")),
        }
    }

    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// Iteration yields one `StepInfo` per step until the session is done.
    fn __next__(&self, py: Python<'_>) -> PyResult<Option<PyStepInfo>> {
        py.check_signals()?;
        let _running = self.core.claim()?;
        Ok(step_once(py, &self.core)?.map(|inner| PyStepInfo { inner }))
    }

    /// Step until done, then finalize. `callback(step)` is called every
    /// `every` iterations (and on the last); returning `False` stops the run.
    /// `cancel()` from another thread stops it too. Returns `"completed"`,
    /// `"converged"` or `"cancelled"`; a cancelled session can be run again.
    #[pyo3(signature = (callback = None, *, every = 1))]
    fn run(
        &self,
        py: Python<'_>,
        callback: Option<Bound<'_, PyAny>>,
        every: usize,
    ) -> PyResult<&'static str> {
        let every = every.max(1);
        self.run_with(py, |py, info| match &callback {
            Some(cb) if info.iteration % every == 0 || info.done => {
                // Only an explicit False stops (None, the usual return, continues).
                let ret = cb.call1((Bound::new(py, PyStepInfo::new(info))?,))?;
                Ok(!matches!(ret.extract::<bool>(), Ok(false)))
            }
            _ => Ok(true),
        })
    }

    /// Ask a running `run()` to stop after the current step (thread-safe, never blocks).
    fn cancel(&self) {
        self.core.cancel.store(true, Ordering::Release);
    }

    /// Remove spheres whose centers ended outside the mesh and end the
    /// session; returns how many were removed. Idempotent.
    fn finalize(&self, py: Python<'_>) -> PyResult<usize> {
        let _running = self.core.claim()?;
        let core = Arc::clone(&self.core);
        py.detach(move || -> PyResult<usize> {
            let mut s = core.lock()?;
            let removed = s.finalize();
            core.publish(&s);
            Ok(removed)
        })
    }

    /// The current spheres as a result (after `finalize`, the final one).
    fn result(&self) -> PyPackResult {
        PyPackResult { inner: self.core.result() }
    }

    #[getter]
    fn iteration(&self) -> usize {
        self.core.snapshot().iteration
    }
    #[getter]
    fn total_iterations(&self) -> usize {
        self.core.snapshot().total_iterations
    }
    /// `"running"`, `"converged"`, `"completed"` or `"finalized"`.
    #[getter]
    fn state(&self) -> &'static str {
        state_name(self.core.snapshot().state)
    }
    #[getter]
    fn is_done(&self) -> bool {
        self.core.snapshot().state != SessionState::Running
    }
    /// True while `run`, `step` or `finalize` is active on some thread.
    #[getter]
    fn is_running(&self) -> bool {
        self.core.running.load(Ordering::Acquire)
    }
    #[getter]
    fn centers<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        flat3(py, &self.core.snapshot().centers)
    }
    #[getter]
    fn radii<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.core.snapshot().radii.clone())
    }
    #[getter]
    fn masses<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f64>> {
        vec1(py, self.core.snapshot().masses.clone())
    }
    #[getter]
    fn num_spheres(&self) -> usize {
        self.core.snapshot().radii.len()
    }
    /// The last step's info (None before the first step).
    #[getter]
    fn last_step(&self) -> Option<PyStepInfo> {
        self.core.snapshot().last_step.clone().map(|inner| PyStepInfo { inner })
    }
    #[getter]
    fn density_passes(&self) -> usize {
        self.core.snapshot().density_passes
    }
    /// Spheres removed by `finalize`.
    #[getter]
    fn pruned(&self) -> usize {
        self.core.snapshot().pruned
    }
    /// Compute device, e.g. `"cpu"` or `"gpu:0 (NVIDIA ...)"`.
    #[getter]
    fn device(&self) -> &str {
        &self.core.device
    }
    /// Why the session fell back from the GPU to the CPU, if it did.
    #[getter]
    fn gpu_error(&self) -> Option<String> {
        self.core.snapshot().gpu_error.clone()
    }
    #[getter]
    fn config(&self) -> PyConfig {
        PyConfig { inner: (*self.core.config).clone() }
    }
    /// The mesh as packed (after mesh preparation).
    #[getter]
    fn mesh(&self) -> PyMesh {
        PyMesh::new(Arc::clone(&self.core.mesh))
    }
    /// What mesh preparation did, as a dict.
    #[getter]
    fn mesh_prep<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        to_py(py, &*self.core.mesh_prep)
    }
    /// `{"seed", "voxel_size", "voxel_candidates"}` of the initialization.
    #[getter]
    fn init_info<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let d = PyDict::new(py);
        d.set_item("seed", self.core.init.seed)?;
        d.set_item("voxel_size", self.core.init.voxel_size)?;
        d.set_item("voxel_candidates", self.core.init.voxel_candidates)?;
        Ok(d)
    }

    /// The per-iteration history (losses, gradients, sphere statistics).
    fn history<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyAny>> {
        let core = Arc::clone(&self.core);
        let v = py.detach(move || core.lock().map(|s| s.history().to_json()))?;
        to_py(py, &v)
    }

    /// The sample points: `(inside (k, 3), surface (m, 3))`.
    fn samples<'py>(&self, py: Python<'py>) -> PyResult<PointSets<'py>> {
        let core = Arc::clone(&self.core);
        let (inside, surface) = py.detach(move || {
            core.lock().map(|s| {
                let smp = &s.problem().samples;
                (smp.inside.clone(), smp.surface.clone())
            })
        })?;
        Ok((dvec3s(py, &inside), dvec3s(py, &surface)))
    }

    fn __repr__(&self) -> String {
        let s = self.core.snapshot();
        format!(
            "Session(state={:?}, iteration={}/{}, num_spheres={}, device={:?})",
            state_name(s.state),
            s.iteration,
            s.total_iterations,
            s.radii.len(),
            self.core.device
        )
    }
}
