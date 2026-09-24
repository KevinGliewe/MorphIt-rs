//! A packing session: initialization, the training loop, and finalization
//! (`morphit.py` + `training.py`).
//!
//! [`Session`] is a plain single-owner state machine; it is `Send + Sync` so it
//! can live behind a mutex (the C API does exactly that).

use std::ops::ControlFlow;
use std::sync::Arc;
use web_time::Instant;

use rand::SeedableRng;

use crate::config::{Config, LossId, LossWeights};
use crate::density::{DensityController, RepackOutcome};
use crate::device::{Device, ResolvedDevice};
use crate::error::{Error, Result};
use crate::loss::{PhysicsTargets, Problem, Samples, evaluate_async};
use crate::mesh::Mesh;
use crate::mesh_prep::MeshPrepReport;
use crate::metrics::{DensityEvent, History, IterationRecord};
use crate::optim::{Optimizer, clip_grad_norm, clip_grad_norm_vec3};
use crate::result::PackResult;
use crate::sampling::{MorphRng, lognormal_radii, sample_inside, sample_surface, voxel_grid_centers};
use crate::state::Spheres;

/// Lifecycle of a session.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SessionState {
    /// Iterations remain.
    Running,
    /// Stopped early by the convergence test.
    Converged,
    /// All configured iterations ran.
    Completed,
    /// The final escaped-sphere prune ran; no more steps.
    Finalized,
}

/// What happened during one [`Session::step`].
#[derive(Clone, Debug, PartialEq)]
pub struct StepInfo {
    /// Zero-based index of the iteration that just ran.
    pub iteration: usize,
    pub total_loss: f64,
    /// `weight * value` per [`LossId`].
    pub weighted_losses: [f64; LossId::COUNT],
    /// Unweighted value per [`LossId`].
    pub raw_losses: [f64; LossId::COUNT],
    /// Mean per-sphere center-gradient norm before clipping.
    pub position_grad_mag: f64,
    /// Norm of the raw-radius gradient before clipping.
    pub radius_grad_mag: f64,
    pub num_spheres: usize,
    /// Centers projected back inside the mesh after the update.
    pub projected: usize,
    /// Set when density control ran after this iteration.
    pub density_control: Option<RepackOutcome>,
    /// The session has no iterations left.
    pub done: bool,
    /// The convergence test stopped the run at this iteration.
    pub converged: bool,
    pub seconds: f64,
}

/// How [`Session::run`] ended.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RunOutcome {
    Completed,
    Converged,
    /// The callback asked to stop; iterations may remain.
    Cancelled,
}

/// Summary of the initialization.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct InitInfo {
    /// Seed that drove all sampling (drawn from OS entropy when the config has none).
    pub seed: u64,
    pub voxel_size: f64,
    pub voxel_candidates: usize,
}

/// One packing run over one mesh.
#[derive(Debug)]
pub struct Session {
    config: Config,
    mesh: Arc<Mesh>,
    problem: Problem,
    weights: LossWeights,
    spheres: Spheres,
    optimizer: Optimizer,
    density: DensityController,
    history: History,
    iteration: usize,
    state: SessionState,
    last_step: Option<StepInfo>,
    density_passes: usize,
    pruned: usize,
    init: InitInfo,
    mesh_prep: MeshPrepReport,
}

/// The mesh a session packs: prepared as `model.union_overlapping_bodies`
/// and `model.convex_hull` ask, the mesh as given when both are off.
fn prepare(config: &Config, mesh: Arc<Mesh>) -> (Arc<Mesh>, MeshPrepReport) {
    mesh.for_model(&config.model)
}

impl Session {
    /// Sample the mesh and initialize spheres (`MorphIt.__init__`).
    pub fn new(mut config: Config, mesh: Arc<Mesh>) -> Result<Session> {
        config.validate()?;
        if let Some(p) = mesh.source_path() {
            config.model.mesh_path = p.to_string();
        }
        let (mesh, mesh_prep) = prepare(&config, mesh);
        let seed = config.random_seed.unwrap_or_else(rand::random);
        let mut rng = MorphRng::seed_from_u64(seed);
        let m = &config.model;
        let n = m.num_spheres;

        let voxel = voxel_grid_centers(&mesh, n, &mut rng);
        let radii = lognormal_radii(mesh.volume(), n, m.initial_radius_variation, &mut rng);
        let inside = sample_inside(&mesh, m.num_inside_samples, &mut rng)?;
        let (surface, face_ids) = sample_surface(&mesh, m.num_surface_samples, &mut rng);
        let surface_normals = face_ids.iter().map(|&f| mesh.face_normals()[f as usize]).collect();

        let targets = PhysicsTargets::from_mesh(&mesh, m.density);
        let masses = m.per_sphere_mass.then(|| vec![targets.mass / n as f64; n]);
        let spheres = Spheres::from_real(voxel.centers, &radii, masses.as_deref());
        let device = Device::parse(&m.device)?;
        let samples = Samples { inside, surface, surface_normals };
        let problem = Problem::new(samples, targets, m.density, device, n)?;
        let init = InitInfo { seed, voxel_size: voxel.voxel_size, voxel_candidates: voxel.candidates };
        let device = problem.search.device();
        tracing::debug!(n, seed, voxel_size = voxel.voxel_size, %device, "session initialized");
        Ok(Self::assemble(config, mesh, mesh_prep, problem, spheres, init))
    }

    /// Build a session from explicit samples and spheres, skipping all random
    /// initialization (used for parity tests and for resuming from saved state).
    pub fn from_parts(
        config: Config,
        mesh: Arc<Mesh>,
        samples: Samples,
        spheres: Spheres,
    ) -> Result<Session> {
        config.validate()?;
        if spheres.is_empty() {
            return Err(Error::config("spheres", "at least one sphere is required"));
        }
        if spheres.per_sphere_mass() != config.model.per_sphere_mass {
            return Err(Error::config("model.per_sphere_mass", "does not match the provided spheres"));
        }
        if samples.inside.is_empty()
            || samples.surface.is_empty()
            || samples.surface.len() != samples.surface_normals.len()
        {
            return Err(Error::config("samples", "need interior samples and one normal per surface sample"));
        }
        let (mesh, mesh_prep) = prepare(&config, mesh);
        let targets = PhysicsTargets::from_mesh(&mesh, config.model.density);
        let device = Device::parse(&config.model.device)?;
        let problem = Problem::new(samples, targets, config.model.density, device, spheres.len())?;
        let init =
            InitInfo { seed: config.random_seed.unwrap_or(0), voxel_size: f64::NAN, voxel_candidates: 0 };
        Ok(Self::assemble(config, mesh, mesh_prep, problem, spheres, init))
    }

    fn assemble(
        config: Config,
        mesh: Arc<Mesh>,
        mesh_prep: MeshPrepReport,
        problem: Problem,
        spheres: Spheres,
        init: InitInfo,
    ) -> Session {
        let weights = config.training.loss_weights();
        let optimizer = Optimizer::new(&config.training, &spheres);
        let density = DensityController::new(config.model.num_spheres, &config.training);
        let state =
            if config.training.iterations == 0 { SessionState::Completed } else { SessionState::Running };
        Session {
            config,
            mesh,
            problem,
            weights,
            spheres,
            optimizer,
            density,
            history: History::default(),
            iteration: 0,
            state,
            last_step: None,
            density_passes: 0,
            pruned: 0,
            init,
            mesh_prep,
        }
    }

    pub fn config(&self) -> &Config {
        &self.config
    }

    /// The mesh being packed (the union when mesh preparation merged bodies).
    pub fn mesh(&self) -> &Arc<Mesh> {
        &self.mesh
    }

    /// What mesh preparation did when the session was created.
    pub fn mesh_prep(&self) -> &MeshPrepReport {
        &self.mesh_prep
    }

    pub fn problem(&self) -> &Problem {
        &self.problem
    }

    pub fn spheres(&self) -> &Spheres {
        &self.spheres
    }

    pub fn history(&self) -> &History {
        &self.history
    }

    pub fn state(&self) -> SessionState {
        self.state
    }

    /// No more iterations will run (converged, completed or finalized).
    pub fn is_done(&self) -> bool {
        self.state != SessionState::Running
    }

    /// Number of iterations run so far.
    pub fn iteration(&self) -> usize {
        self.iteration
    }

    pub fn total_iterations(&self) -> usize {
        self.config.training.iterations
    }

    pub fn last_step(&self) -> Option<&StepInfo> {
        self.last_step.as_ref()
    }

    pub fn density_passes(&self) -> usize {
        self.density_passes
    }

    pub fn density_controller(&self) -> &DensityController {
        &self.density
    }

    /// Spheres removed by [`Session::finalize`].
    pub fn pruned(&self) -> usize {
        self.pruned
    }

    /// The device the distance searches run on.
    pub fn device(&self) -> &ResolvedDevice {
        self.problem.search.device()
    }

    /// Why the GPU searches fell back to the CPU, if they did (the results
    /// are the same either way).
    pub fn gpu_error(&self) -> Option<String> {
        self.problem.search.gpu_error()
    }

    pub fn init_info(&self) -> InitInfo {
        self.init
    }

    /// Whether [`Session::step_async`] must be used: the searches run on
    /// WebGPU in the browser, whose readbacks resolve only after yielding to
    /// the event loop. Always false natively.
    pub fn needs_async(&self) -> bool {
        self.problem.search.needs_async()
    }

    /// Run one training iteration (`MorphItTrainer._training_step` plus the
    /// loop body: tracking, convergence check, density control).
    ///
    /// Fails without touching the session when [`Session::needs_async`] is
    /// set; drive such a session with [`Session::step_async`].
    pub fn step(&mut self) -> Result<StepInfo> {
        if self.needs_async() {
            return Err(Error::State("GPU searches in the browser need Session::step_async".into()));
        }
        crate::exec::block_on(self.step_async())
    }

    /// [`Session::step`], awaiting the GPU readbacks. Natively the future
    /// completes on its first poll; with WebGPU in the browser it yields while
    /// the GPU works.
    ///
    /// Not cancel-safe: dropping the future before it completes can leave the
    /// spheres half-updated. Let the step in flight finish, then stop.
    pub async fn step_async(&mut self) -> Result<StepInfo> {
        match self.state {
            SessionState::Running => {}
            SessionState::Finalized => return Err(Error::State("session is finalized".into())),
            _ => return Err(Error::State("no iterations remain".into())),
        }
        let started = Instant::now();
        let t = &self.config.training;

        let eval = evaluate_async(&self.problem, &self.spheres, &self.weights).await;
        let mut grads = eval.grads;
        let n = self.spheres.len();
        let position_grad_mag = grads.centers.iter().map(|g| g.length()).sum::<f64>() / n as f64;
        let radius_grad_mag = grads.raw_radii.iter().map(|g| g * g).sum::<f64>().sqrt();
        clip_grad_norm_vec3(&mut grads.centers, t.grad_clip_norm);
        clip_grad_norm(&mut grads.raw_radii, t.grad_clip_norm * 0.5);
        self.optimizer.step(&mut self.spheres, &grads);
        let projected =
            project_centers_inside_async(&self.problem, &mut self.spheres, self.mesh.scale()).await;

        let it = self.iteration;
        let radii = self.spheres.radii();
        self.history.record(IterationRecord {
            iteration: it,
            total_loss: eval.total,
            weighted: &eval.weighted,
            position_grad_mag,
            radius_grad_mag,
            radii: &radii,
            seconds: started.elapsed().as_secs_f64(),
        });

        let mut converged = false;
        let mut density_control = None;
        if it % t.verbose_frequency == 0 {
            tracing::debug!(iteration = it, total_loss = eval.total, spheres = n, "progress");
            if t.early_stopping
                && it > t.convergence_patience
                && self.history.analyze_convergence(t.convergence_patience, t.convergence_threshold).converged
            {
                converged = true;
                self.state = SessionState::Converged;
                tracing::info!(iteration = it, "training converged");
            }
        }
        if !converged && t.density_control_enabled && self.density.should_trigger(t, &self.history, it) {
            let out = self
                .density
                .repack_async(&self.problem, &self.mesh, &mut self.spheres, t, &self.weights)
                .await;
            self.history.density_control_events.push(DensityEvent {
                iteration: it,
                spheres_added: out.added,
                spheres_removed: out.removed,
            });
            self.density.mark(it);
            self.density_passes += 1;
            if out.added > 0 || out.removed > 0 {
                self.optimizer.reset(&self.spheres);
            }
            density_control = Some(out);
        }

        self.iteration += 1;
        if self.state == SessionState::Running && self.iteration >= t.iterations {
            self.state = SessionState::Completed;
        }
        let info = StepInfo {
            iteration: it,
            total_loss: eval.total,
            weighted_losses: eval.weighted,
            raw_losses: eval.raw,
            position_grad_mag,
            radius_grad_mag,
            num_spheres: self.spheres.len(),
            projected,
            density_control,
            done: self.is_done(),
            converged,
            seconds: started.elapsed().as_secs_f64(),
        };
        self.last_step = Some(info.clone());
        Ok(info)
    }

    /// Step until done or until `on_step` returns `ControlFlow::Break`.
    pub fn run(&mut self, mut on_step: impl FnMut(&StepInfo) -> ControlFlow<()>) -> Result<RunOutcome> {
        while !self.is_done() {
            let info = self.step()?;
            if on_step(&info).is_break() {
                return Ok(RunOutcome::Cancelled);
            }
        }
        Ok(match self.state {
            SessionState::Converged => RunOutcome::Converged,
            _ => RunOutcome::Completed,
        })
    }

    /// Remove spheres whose centers ended outside the mesh
    /// (`_finalize_training`) and end the session. Idempotent; returns the
    /// number of spheres removed. At least one sphere is always kept.
    pub fn finalize(&mut self) -> usize {
        if self.state == SessionState::Finalized {
            return self.pruned;
        }
        let inside = self.mesh.contains_many(&self.spheres.centers);
        let mut keep: Vec<usize> = inside.iter().enumerate().filter_map(|(i, &ok)| ok.then_some(i)).collect();
        if keep.is_empty() {
            tracing::warn!("every sphere center is outside the mesh; keeping the first sphere");
            keep.push(0);
        }
        let removed = self.spheres.len() - keep.len();
        if removed > 0 {
            self.spheres = self.spheres.select(&keep);
            tracing::info!(removed, "final prune removed spheres whose centers escaped the mesh");
        }
        self.pruned = removed;
        self.state = SessionState::Finalized;
        removed
    }

    /// Current spheres as a result in the Python JSON schema.
    pub fn result(&self) -> PackResult {
        let radii = self.spheres.radii();
        let masses = self.spheres.masses(self.problem.density);
        PackResult {
            centers: self.spheres.centers.iter().map(|c| c.to_array()).collect(),
            radii,
            masses,
            mesh_path: self.config.model.mesh_path.clone(),
            num_spheres: self.spheres.len(),
            per_sphere_mass: self.spheres.per_sphere_mass(),
            mesh_prep: Some(self.mesh_prep.clone()),
            config: self.config.clone(),
        }
    }
}

/// Snap centers that drifted outside back just inside the surface, using the
/// nearest surface sample and its normal (`_project_centers_inside_mesh`).
/// Returns how many centers moved.
pub fn project_centers_inside(problem: &Problem, spheres: &mut Spheres, mesh_scale: f64) -> usize {
    crate::exec::block_on(project_centers_inside_async(problem, spheres, mesh_scale))
}

/// [`project_centers_inside`], awaiting the GPU readback.
pub async fn project_centers_inside_async(
    problem: &Problem,
    spheres: &mut Spheres,
    mesh_scale: f64,
) -> usize {
    let surface = &problem.samples.surface;
    let normals = &problem.samples.surface_normals;
    let nearest = problem.nearest_sample_per_center_async(&spheres.centers).await;
    let eps = 1e-4 * mesh_scale;
    let mut moved = 0;
    for (c, &i) in spheres.centers.iter_mut().zip(&nearest) {
        let s = surface[i as usize];
        let n = normals[i as usize];
        let v = *c - s;
        let sd = v.x * n.x + v.y * n.y + v.z * n.z;
        if sd > 0.0 {
            *c -= n * (sd + eps);
            moved += 1;
        }
    }
    moved
}

/// Pack a mesh in one call: create a session, run all iterations, finalize.
pub fn pack(config: Config, mesh: Arc<Mesh>) -> Result<PackResult> {
    let mut s = Session::new(config, mesh)?;
    s.run(|_| ControlFlow::Continue(()))?;
    s.finalize();
    Ok(s.result())
}

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Session>();
    assert_send_sync::<Mesh>();
    assert_send_sync::<Config>();
};

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::Preset;
    use crate::shapes;
    use glam::DVec3;

    fn small_config(preset: Preset, iterations: usize, n: usize) -> Config {
        let mut c = Config::from_preset(preset);
        c.model.num_spheres = n;
        c.model.num_inside_samples = 800;
        c.model.num_surface_samples = 800;
        c.training.iterations = iterations;
        c.random_seed = Some(7);
        c
    }

    fn cube() -> Arc<Mesh> {
        Arc::new(shapes::box_mesh(DVec3::ZERO, DVec3::new(1.0, 0.8, 0.6)))
    }

    #[cfg(feature = "union")]
    #[test]
    fn overlapping_bodies_are_packed_as_their_union() {
        let raw = crate::mesh_prep::tests::two_overlapping_boxes();
        let mut s = Session::new(small_config(Preset::B, 30, 10), raw.clone()).unwrap();
        assert_eq!(s.mesh_prep().action, "unioned");
        assert!((s.mesh().volume() - 1.568).abs() < 0.01 * 1.568);
        assert!((raw.volume() - 2.0).abs() < 1e-12, "the input mesh is untouched");
        // Interior samples now cover the former overlap.
        let mid = DVec3::new(0.7, 0.6, 0.55);
        assert!(s.problem().samples.inside.iter().any(|p| (*p - mid).length() < 0.15));
        s.run(|_| ControlFlow::Continue(())).unwrap();
        s.finalize();
        let v: serde_json::Value = serde_json::from_str(&s.result().to_json_string()).unwrap();
        assert_eq!(v["mesh_prep"]["action"], "unioned");
        assert_eq!(v["mesh_prep"]["n_bodies"], 2);
        assert_eq!(v["config"]["model"]["union_overlapping_bodies"], true);

        // A second session on the same mesh reuses the cached union.
        let s2 = Session::new(small_config(Preset::B, 1, 4), raw.clone()).unwrap();
        assert!(Arc::ptr_eq(s.mesh(), s2.mesh()));

        let mut off = small_config(Preset::B, 1, 4);
        off.model.union_overlapping_bodies = false;
        let s3 = Session::new(off, raw.clone()).unwrap();
        assert_eq!(s3.mesh_prep().action, "disabled");
        assert!(Arc::ptr_eq(s3.mesh(), &raw));
        assert_eq!(v["config"]["model"]["convex_hull"], false);

        // Convex hulls without the union: the two boxes side by side.
        let mut hull = small_config(Preset::B, 1, 4);
        hull.model.union_overlapping_bodies = false;
        hull.model.convex_hull = true;
        let s4 = Session::new(hull, raw.clone()).unwrap();
        assert_eq!((s4.mesh_prep().action.as_str(), s4.mesh_prep().n_hulled), ("hulled", 2));
        assert_eq!(s4.mesh().faces().len(), raw.faces().len());
        assert!(s4.result().config.model.convex_hull);
    }

    #[test]
    fn single_body_meshes_are_packed_as_given_without_a_cycle() {
        let mesh = cube();
        let s = Session::new(small_config(Preset::B, 1, 4), mesh.clone()).unwrap();
        assert_eq!(s.mesh_prep().action, "unchanged");
        assert_eq!(s.mesh_prep().reason, "single body");
        assert!(Arc::ptr_eq(s.mesh(), &mesh));
        drop(s);
        // The cached preparation holds no reference to the mesh itself.
        assert_eq!(Arc::strong_count(&mesh), 1);
    }

    #[test]
    fn short_run_improves_and_stays_inside() {
        let mut cfg = small_config(Preset::B, 40, 8);
        cfg.training.density_control_enabled = false;
        let mut s = Session::new(cfg, cube()).unwrap();
        let mut losses = Vec::new();
        let mut calls = 0;
        let out = s
            .run(|info| {
                calls += 1;
                losses.push(info.total_loss);
                ControlFlow::Continue(())
            })
            .unwrap();
        assert_eq!(out, RunOutcome::Completed);
        assert_eq!(calls, 40);
        assert!(losses.iter().all(|l| l.is_finite()));
        assert!(losses[39] < losses[0], "{} !< {}", losses[39], losses[0]);
        assert!(s.spheres().centers.iter().all(|c| c.is_finite()));
        assert!(s.step().is_err(), "no iterations remain");
        assert_eq!(s.iteration(), 40);
        assert_eq!(s.history().len(), 40);
    }

    #[test]
    fn callback_can_cancel() {
        let mut s = Session::new(small_config(Preset::V, 50, 6), cube()).unwrap();
        let out = s.run(|_| ControlFlow::Break(())).unwrap();
        assert_eq!(out, RunOutcome::Cancelled);
        assert_eq!(s.iteration(), 1);
        assert!(!s.is_done());
    }

    #[test]
    fn same_seed_is_bit_identical() {
        let run = || {
            let mut s = Session::new(small_config(Preset::S, 25, 6), cube()).unwrap();
            s.run(|_| ControlFlow::Continue(())).unwrap();
            s.spheres().clone()
        };
        assert_eq!(run(), run());
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn identical_across_thread_counts() {
        let run = |threads| {
            let pool = rayon::ThreadPoolBuilder::new().num_threads(threads).build().unwrap();
            pool.install(|| {
                let mut s = Session::new(small_config(Preset::B, 20, 10), cube()).unwrap();
                s.run(|_| ControlFlow::Continue(())).unwrap();
                s.result()
            })
        };
        assert_eq!(run(1), run(4));
    }

    #[test]
    fn identical_across_devices() {
        let mut gpu_cfg = small_config(Preset::B, 20, 10);
        gpu_cfg.model.device = "gpu".to_string();
        let mut on_gpu = match Session::new(gpu_cfg, cube()) {
            Ok(s) => s,
            Err(e) => {
                eprintln!("skipping: {e}");
                return;
            }
        };
        assert!(matches!(on_gpu.device(), ResolvedDevice::Gpu { .. }));
        let mut on_cpu = Session::new(small_config(Preset::B, 20, 10), cube()).unwrap();
        assert_eq!(*on_cpu.device(), ResolvedDevice::Cpu, "auto picks the CPU for a tiny problem");
        on_gpu.run(|_| ControlFlow::Continue(())).unwrap();
        on_cpu.run(|_| ControlFlow::Continue(())).unwrap();
        assert_eq!(on_gpu.gpu_error(), None, "the GPU must really have answered the searches");
        // Only the wall-clock timings may differ.
        let mut ha = on_gpu.history().to_json();
        let mut hb = on_cpu.history().to_json();
        for h in [&mut ha, &mut hb] {
            h.as_object_mut().unwrap().remove("time_per_iteration");
        }
        assert_eq!(ha, hb);
        on_gpu.finalize();
        on_cpu.finalize();
        let mut a = on_gpu.result();
        let mut b = on_cpu.result();
        a.config.model.device = String::new();
        b.config.model.device = String::new();
        assert_eq!(a, b);
    }

    /// The async core driven by an executor equals the sync API, on the CPU
    /// and (when present) on a GPU, including density control.
    #[test]
    fn step_async_matches_step() {
        for device in ["cpu", "gpu"] {
            let mut cfg = small_config(Preset::B, 30, 10);
            cfg.training.density_control_min_interval = 12;
            cfg.training.density_control_warmup_steps = 3;
            cfg.model.device = device.to_string();
            let Ok(mut a) = Session::new(cfg.clone(), cube()) else {
                eprintln!("skipping {device}: no adapter");
                continue;
            };
            let mut b = Session::new(cfg, cube()).unwrap();
            assert!(!a.needs_async());
            while !a.is_done() {
                let x = pollster::block_on(a.step_async()).unwrap();
                let y = b.step().unwrap();
                assert_eq!(
                    (x.iteration, x.total_loss, x.num_spheres),
                    (y.iteration, y.total_loss, y.num_spheres)
                );
            }
            assert!(b.is_done());
            assert!(a.density_passes() > 0, "the test must cover density control");
            assert_eq!(a.gpu_error(), None, "{device}");
            let mut ha = a.history().to_json();
            let mut hb = b.history().to_json();
            for h in [&mut ha, &mut hb] {
                h.as_object_mut().unwrap().remove("time_per_iteration");
            }
            assert_eq!(ha, hb, "{device}");
            assert_eq!(a.result(), b.result(), "{device}");
        }
    }

    #[test]
    fn step_async_future_is_send() {
        fn assert_send<T: Send>(_: &T) {}
        let mut s = Session::new(small_config(Preset::B, 4, 2), cube()).unwrap();
        let f = s.step_async();
        assert_send(&f);
    }

    #[test]
    fn grad_magnitudes_are_recorded_before_clipping() {
        let mut cfg = small_config(Preset::V, 1, 6);
        cfg.training.grad_clip_norm = 1e-9;
        let mut s = Session::new(cfg, cube()).unwrap();
        let info = s.step().unwrap();
        assert!(info.radius_grad_mag > 1e-9);
    }

    #[test]
    fn density_control_fires_at_min_interval_and_count_is_preserved() {
        let mut cfg = small_config(Preset::B, 30, 10);
        cfg.training.density_control_min_interval = 12;
        cfg.training.density_control_warmup_steps = 3;
        let mut s = Session::new(cfg, cube()).unwrap();
        let mut fired = Vec::new();
        s.run(|info| {
            if info.density_control.is_some() {
                fired.push(info.iteration);
            }
            assert_eq!(info.num_spheres, 10);
            ControlFlow::Continue(())
        })
        .unwrap();
        assert_eq!(fired, vec![12, 24]);
        assert_eq!(s.density_passes(), 2);
        assert_eq!(s.history().density_control_events.len(), 2);
    }

    #[test]
    fn finalize_prunes_escaped_centers_and_is_idempotent() {
        let mut s = Session::new(small_config(Preset::B, 5, 6), cube()).unwrap();
        s.spheres.centers[2] = DVec3::new(3.0, 3.0, 3.0);
        s.spheres.centers[4] = DVec3::new(-1.0, 0.2, 0.2);
        assert_eq!(s.finalize(), 2);
        assert_eq!(s.spheres().len(), 4);
        assert_eq!(s.finalize(), 2);
        assert_eq!(s.state(), SessionState::Finalized);
        assert!(s.step().is_err());
        let r = s.result();
        assert_eq!(r.num_spheres, 4);
        assert_eq!(r.centers.len(), 4);
    }

    #[test]
    fn finalize_keeps_one_sphere_when_all_escaped() {
        let mut s = Session::new(small_config(Preset::B, 5, 3), cube()).unwrap();
        for c in &mut s.spheres.centers {
            *c = DVec3::splat(9.0);
        }
        assert_eq!(s.finalize(), 2);
        assert_eq!(s.spheres().len(), 1);
    }

    #[test]
    fn projection_moves_escaped_centers_inside() {
        let mut s = Session::new(small_config(Preset::B, 5, 4), cube()).unwrap();
        s.spheres.centers[0] = DVec3::new(0.5, 0.4, 0.9); // above the top face at z = 0.6
        let moved = project_centers_inside(&s.problem, &mut s.spheres, s.mesh.scale());
        assert_eq!(moved, 1);
        let z = s.spheres.centers[0].z;
        assert!(z < 0.6 && z > 0.59, "{z}");
    }

    #[test]
    fn per_sphere_mass_initializes_to_equal_shares() {
        let mut cfg = small_config(Preset::ObjMass, 3, 5);
        cfg.model.per_sphere_mass = true;
        let s = Session::new(cfg, cube()).unwrap();
        let m = s.spheres().masses(1000.0);
        let expect = 0.48 * 1000.0 / 5.0;
        assert!(m.iter().all(|x| (x - expect).abs() < 1e-9));
        let mut s = s;
        s.run(|_| ControlFlow::Continue(())).unwrap();
        assert!(s.result().per_sphere_mass);
    }

    #[test]
    fn early_stopping_converges() {
        let mut cfg = small_config(Preset::B, 500, 4);
        cfg.training.early_stopping = true;
        cfg.training.convergence_patience = 5;
        cfg.training.verbose_frequency = 5;
        cfg.training.convergence_threshold = 0.5;
        cfg.training.density_control_enabled = false;
        let mut s = Session::new(cfg, cube()).unwrap();
        let out = s.run(|_| ControlFlow::Continue(())).unwrap();
        assert_eq!(out, RunOutcome::Converged);
        assert!(s.iteration() < 500);
        assert_eq!(s.state(), SessionState::Converged);
    }

    #[test]
    fn zero_iterations_is_immediately_done() {
        let s = Session::new(small_config(Preset::B, 0, 4), cube()).unwrap();
        assert!(s.is_done());
    }

    #[test]
    fn result_json_has_python_keys() {
        let mut s = Session::new(small_config(Preset::B, 2, 4), cube()).unwrap();
        s.run(|_| ControlFlow::Continue(())).unwrap();
        let v: serde_json::Value = serde_json::from_str(&s.result().to_json_string()).unwrap();
        let mut keys: Vec<&str> = v.as_object().unwrap().keys().map(|k| k.as_str()).collect();
        keys.sort();
        assert_eq!(
            keys,
            [
                "centers",
                "config",
                "masses",
                "mesh_path",
                "mesh_prep",
                "num_spheres",
                "per_sphere_mass",
                "radii"
            ]
        );
        let mut ck: Vec<&str> = v["config"].as_object().unwrap().keys().map(|k| k.as_str()).collect();
        ck.sort();
        assert_eq!(
            ck,
            ["model", "output_filename", "random_seed", "results_dir", "training", "visualization"]
        );
        assert_eq!(v["config"]["training"]["sqem_weight"], serde_json::json!(3000.0));
        let back = PackResult::from_json_str(&s.result().to_json_string()).unwrap();
        assert_eq!(back, s.result());
    }
}
