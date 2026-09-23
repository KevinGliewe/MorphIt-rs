//! Annealed, count-preserving partial re-packing (`density_control.py`).
//!
//! Each pass scores spheres by marginal coverage value, culls the `k` least
//! valuable (always including collapsed or escaped ones), reseeds exactly `k`
//! new spheres at the worst-covered interior points, and warms the newcomers up
//! for a few optimizer steps while the survivors stay frozen. The number of
//! spheres culled per pass shrinks as the temperature cools.

use crate::par::*;
use glam::DVec3;

use crate::config::{LossWeights, TrainingConfig};
use crate::distances::{dist, row_minima};
use crate::loss::{Problem, evaluate_async};
use crate::math::{argmax_first, torch_median};
use crate::mesh::Mesh;
use crate::metrics::History;
use crate::optim::AdamGroup;
use crate::state::Spheres;

const INITIAL_TEMPERATURE: f64 = 0.4;
const MIN_TEMPERATURE: f64 = 0.04;

/// What one re-packing pass did.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RepackOutcome {
    pub added: usize,
    pub removed: usize,
    /// Spheres force-culled for being tiny or having their center outside the mesh.
    pub bad: usize,
}

/// Density-control state carried across a run.
#[derive(Clone, Debug, PartialEq)]
pub struct DensityController {
    target: usize,
    temperature: f64,
    cooling: f64,
    last_iteration: usize,
    warmup_steps: usize,
    warmup_lr: f64,
}

impl DensityController {
    pub fn new(target: usize, t: &TrainingConfig) -> Self {
        DensityController {
            target,
            temperature: INITIAL_TEMPERATURE,
            cooling: t.density_control_cooling_factor,
            last_iteration: 0,
            warmup_steps: t.density_control_warmup_steps,
            warmup_lr: t.center_lr * 5.0,
        }
    }

    /// Current replacement fraction.
    pub fn temperature(&self) -> f64 {
        self.temperature
    }

    /// Iteration of the last pass (0 before the first).
    pub fn last_iteration(&self) -> usize {
        self.last_iteration
    }

    pub fn mark(&mut self, iteration: usize) {
        self.last_iteration = iteration;
    }

    /// `should_perform_density_control`: respects the minimum interval, forces a
    /// pass after twice the interval, otherwise fires on a loss plateau or small
    /// gradients over the last `patience` iterations.
    pub fn should_trigger(&self, t: &TrainingConfig, history: &History, iteration: usize) -> bool {
        let min_interval = t.density_control_min_interval;
        let patience = t.density_control_patience;
        let threshold = t.density_control_grad_threshold;
        let since = iteration.saturating_sub(self.last_iteration);
        if since < min_interval {
            return false;
        }
        if since > min_interval * 2 {
            return true;
        }
        let n = history.len();
        if patience == 0 || n < patience {
            return false;
        }
        let recent = &history.total_loss[n - patience..];
        let loss_change = (recent[0] - recent[patience - 1]).abs() / recent[0].abs().max(1e-5);
        let plateau = loss_change < 0.01;
        let pos = &history.position_grad_mag[n - patience..];
        let rad = &history.radius_grad_mag[n - patience..];
        let grads_small = pos.iter().filter(|&&g| g < threshold).count() > patience / 2
            && rad.iter().filter(|&&g| g < threshold).count() > patience / 2;
        plateau || grads_small
    }

    /// `adaptive_density_control`: one count-preserving re-packing pass.
    pub fn repack(
        &mut self,
        problem: &Problem,
        mesh: &Mesh,
        spheres: &mut Spheres,
        t: &TrainingConfig,
        weights: &LossWeights,
    ) -> RepackOutcome {
        crate::exec::block_on(self.repack_async(problem, mesh, spheres, t, weights))
    }

    /// [`DensityController::repack`], awaiting the GPU readbacks of the warmup.
    pub async fn repack_async(
        &mut self,
        problem: &Problem,
        mesh: &Mesh,
        spheres: &mut Spheres,
        t: &TrainingConfig,
        weights: &LossWeights,
    ) -> RepackOutcome {
        let n = spheres.len();
        let radii = spheres.radii();
        let bad = identify_bad(mesh, spheres, &radii, t);
        let n_bad = bad.iter().filter(|&&b| b).count();

        let k_temp = ((self.target as f64 * self.temperature) as usize).max(1);
        let k = k_temp.max(n_bad).min(self.target.saturating_sub(1)).min(n.saturating_sub(1));
        if k == 0 {
            tracing::info!(n, "skipping density control: sphere count too low to cull");
            return RepackOutcome { added: 0, removed: 0, bad: n_bad };
        }

        let mut values = marginal_values(problem, &spheres.centers, &radii, t);
        for (v, &b) in values.iter_mut().zip(&bad) {
            if b {
                *v = f64::NEG_INFINITY;
            }
        }
        // Survivors in descending value order (stable: lower index first on ties).
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by(|&a, &b| values[b].total_cmp(&values[a]).then(a.cmp(&b)));
        let keep = &order[..n - k];

        let masses = spheres.per_sphere_mass().then(|| spheres.masses(problem.density));
        let surv_centers: Vec<DVec3> = keep.iter().map(|&i| spheres.centers[i]).collect();
        let surv_radii: Vec<f64> = keep.iter().map(|&i| radii[i]).collect();
        let surv_masses: Option<Vec<f64>> = masses.as_ref().map(|m| keep.iter().map(|&i| m[i]).collect());

        let (new_centers, new_radii) =
            farthest_point_reseed(problem, mesh.scale(), &surv_centers, &surv_radii, k);

        let mut all_centers = surv_centers;
        all_centers.extend(&new_centers);
        let mut all_radii = surv_radii;
        all_radii.extend(&new_radii);
        let all_masses = surv_masses.map(|mut m| {
            let init = torch_median(&m).expect("at least one survivor");
            m.extend(std::iter::repeat_n(init, new_centers.len()));
            m
        });
        let n_surv = n - k;
        *spheres = Spheres::from_real(all_centers, &all_radii, all_masses.as_deref());

        warmup_async(problem, spheres, weights, n_surv, self.warmup_steps, self.warmup_lr).await;

        let old = self.temperature;
        self.temperature = (self.temperature * self.cooling).max(MIN_TEMPERATURE);
        tracing::info!(
            culled = k,
            bad = n_bad,
            temperature_before = old,
            temperature_after = self.temperature,
            "density control re-packed spheres"
        );
        RepackOutcome { added: new_centers.len(), removed: k, bad: n_bad }
    }
}

/// Spheres to cull regardless of value: radius below
/// `min_radius_fraction * scale`, or center outside the mesh.
pub fn identify_bad(mesh: &Mesh, spheres: &Spheres, radii: &[f64], t: &TrainingConfig) -> Vec<bool> {
    let min_radius = t.density_control_min_radius_fraction * mesh.scale();
    let inside = mesh.contains_many(&spheres.centers);
    radii.iter().zip(inside).map(|(&r, ins)| r < min_radius || !ins).collect()
}

/// Marginal value of each sphere: the coverage hole that opens if it is removed,
/// accumulated over interior samples (weighted by the coverage weight) and
/// surface samples (weighted by surface + boundary + sqem weights).
pub fn marginal_values(problem: &Problem, centers: &[DVec3], radii: &[f64], t: &TrainingConfig) -> Vec<f64> {
    let n = centers.len();
    let w_int = t.coverage_weight;
    let w_surf = t.surface_weight + t.boundary_weight + t.sqem_weight;
    let score = |samples: &[DVec3]| -> Vec<f64> {
        let rows: Vec<(usize, f64)> = samples
            .par_iter()
            .map(|&p| {
                let (mut i1, mut d1, mut d2) = (0usize, f64::INFINITY, f64::INFINITY);
                for (j, (&c, &r)) in centers.iter().zip(radii).enumerate() {
                    let d = dist(p, c) - r;
                    if d < d1 {
                        d2 = d1;
                        (i1, d1) = (j, d);
                    } else if d < d2 {
                        d2 = d;
                    }
                }
                let second = if n >= 2 { d2 } else { 1e6 };
                (i1, second - d1)
            })
            .collect();
        let mut out = vec![0.0; n];
        for (i, gap) in rows {
            out[i] += gap;
        }
        out
    };
    let mut values = vec![0.0; n];
    if w_int > 0.0 {
        for (v, s) in values.iter_mut().zip(score(&problem.samples.inside)) {
            *v += w_int * s;
        }
    }
    if w_surf > 0.0 {
        for (v, s) in values.iter_mut().zip(score(&problem.samples.surface)) {
            *v += w_surf * s;
        }
    }
    values
}

/// Place exactly `k` new spheres by iterative farthest-point sampling over the
/// interior samples, sized to fill the local gap (at least the survivor median,
/// at most 1.5x the median and 0.3x the mesh scale).
pub fn farthest_point_reseed(
    problem: &Problem,
    mesh_scale: f64,
    surv_centers: &[DVec3],
    surv_radii: &[f64],
    k: usize,
) -> (Vec<DVec3>, Vec<f64>) {
    let cand = &problem.samples.inside;
    let mut min_d: Vec<f64> = if surv_centers.is_empty() {
        vec![1e6; cand.len()]
    } else {
        row_minima(cand, surv_centers, surv_radii).into_iter().map(|r| r.gap).collect()
    };
    let (default_r, cap) = match torch_median(surv_radii) {
        Some(m) => (m, m * 1.5),
        None => (1e-3, 1e6),
    };
    let mut centers = Vec::with_capacity(k);
    let mut radii = Vec::with_capacity(k);
    for _ in 0..k {
        let (w, gap) = argmax_first(&min_d);
        let c = cand[w];
        let r = gap.max(default_r).min(cap).min(mesh_scale * 0.3);
        centers.push(c);
        radii.push(r);
        min_d.par_iter_mut().zip(cand.par_iter()).for_each(|(m, &p)| {
            let d = dist(p, c) - r;
            if d < *m {
                *m = d;
            }
        });
    }
    (centers, radii)
}

/// A few Adam steps on the freshly placed spheres only (survivors' gradients
/// are zeroed and their values restored afterwards). Masses are not updated.
pub fn warmup(
    problem: &Problem,
    spheres: &mut Spheres,
    weights: &LossWeights,
    n_surv: usize,
    steps: usize,
    lr: f64,
) {
    crate::exec::block_on(warmup_async(problem, spheres, weights, n_surv, steps, lr))
}

/// [`warmup`], awaiting the GPU readbacks.
pub async fn warmup_async(
    problem: &Problem,
    spheres: &mut Spheres,
    weights: &LossWeights,
    n_surv: usize,
    steps: usize,
    lr: f64,
) {
    if steps == 0 {
        return;
    }
    let saved_centers = spheres.centers[..n_surv].to_vec();
    let saved_radii = spheres.raw_radii[..n_surv].to_vec();
    let n = spheres.len();
    let mut opt_c = AdamGroup::new(lr, 3 * n);
    let mut opt_r = AdamGroup::new(lr * 0.5, n);
    let mut last = f64::NAN;
    for _ in 0..steps {
        let e = evaluate_async(problem, spheres, weights).await;
        last = e.total;
        let mut gc = e.grads.centers;
        let mut gr = e.grads.raw_radii;
        gc[..n_surv].fill(DVec3::ZERO);
        gr[..n_surv].fill(0.0);
        opt_c.step_vec3(&mut spheres.centers, &gc);
        opt_r.step(&mut spheres.raw_radii, &gr);
    }
    spheres.centers[..n_surv].copy_from_slice(&saved_centers);
    spheres.raw_radii[..n_surv].copy_from_slice(&saved_radii);
    tracing::debug!(steps, final_loss = last, "density-control warmup complete");
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::{Config, LossId, Preset};
    use crate::loss::PhysicsTargets;
    use crate::loss::fd::{toy_problem, toy_spheres};
    use crate::metrics::IterationRecord;
    use crate::shapes;

    fn history(losses: &[f64], grads: f64) -> History {
        let mut h = History::default();
        let w = [0.0; LossId::COUNT];
        for (i, &l) in losses.iter().enumerate() {
            h.record(IterationRecord {
                iteration: i,
                total_loss: l,
                weighted: &w,
                position_grad_mag: grads,
                radius_grad_mag: grads,
                radii: &[0.1],
                seconds: 0.0,
            });
        }
        h
    }

    #[test]
    fn trigger_schedule() {
        let t = Config::default().training;
        let dc = DensityController::new(25, &t);
        let h = history(&vec![1.0; 170], 1.0);
        assert!(!dc.should_trigger(&t, &h, 159));
        assert!(dc.should_trigger(&t, &h, 160), "patience 1: a single-point window is a plateau");
        let mut dc2 = dc.clone();
        dc2.mark(160);
        assert!(!dc2.should_trigger(&t, &h, 319));
        // Forced after twice the interval even without history.
        assert!(dc2.should_trigger(&t, &History::default(), 160 + 321));
    }

    #[test]
    fn trigger_with_longer_patience() {
        let mut t = Config::default().training;
        t.density_control_patience = 5;
        let dc = DensityController::new(25, &t);
        let falling: Vec<f64> = (0..200).map(|i| 100.0 - i as f64 * 0.4).collect();
        assert!(!dc.should_trigger(&t, &history(&falling, 1.0), 170));
        assert!(dc.should_trigger(&t, &history(&falling, 1e-6), 170), "small gradients");
        let flat = vec![3.0; 200];
        assert!(dc.should_trigger(&t, &history(&flat, 1.0), 170), "plateau");
        assert!(!dc.should_trigger(&t, &history(&flat[..3], 1.0), 170), "not enough history");
    }

    #[test]
    fn bad_spheres_are_tiny_or_outside() {
        let mesh = shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        let sp = Spheres::from_real(
            vec![DVec3::new(0.3, 0.4, 0.6), DVec3::new(0.3, 0.4, 1.6), DVec3::new(0.7, 0.2, 0.3)],
            &[0.2, 0.2, 1e-5],
            None,
        );
        let t = Config::default().training;
        assert_eq!(identify_bad(&mesh, &sp, &sp.radii(), &t), vec![false, true, true]);
    }

    #[test]
    fn marginal_values_match_brute_force() {
        let p = toy_problem(1, 50, 50, 1.0);
        let sp = toy_spheres(2, 6, false);
        let r = sp.radii();
        let t = Config::from_preset(Preset::B).training;
        let v = marginal_values(&p, &sp.centers, &r, &t);
        let mut expect = [0.0; 6];
        for (samples, w) in [
            (&p.samples.inside, t.coverage_weight),
            (&p.samples.surface, t.surface_weight + t.boundary_weight + t.sqem_weight),
        ] {
            for s in samples.iter() {
                let mut d: Vec<(f64, usize)> = sp
                    .centers
                    .iter()
                    .zip(&r)
                    .enumerate()
                    .map(|(j, (c, r))| ((*s - *c).length() - r, j))
                    .collect();
                d.sort_by(|a, b| a.0.total_cmp(&b.0));
                expect[d[0].1] += w * (d[1].0 - d[0].0);
            }
        }
        for j in 0..6 {
            assert!((v[j] - expect[j]).abs() < 1e-9 * expect[j].abs().max(1.0));
        }
    }

    #[test]
    fn reseed_places_exactly_k_inside_samples_with_bounded_radii() {
        let p = toy_problem(3, 400, 100, 1.0);
        let surv = vec![DVec3::new(0.2, 0.2, 0.2), DVec3::new(0.8, 0.9, 0.5)];
        let surv_r = vec![0.1, 0.2];
        let (c, r) = farthest_point_reseed(&p, 1.8, &surv, &surv_r, 5);
        assert_eq!(c.len(), 5);
        for (ci, ri) in c.iter().zip(&r) {
            assert!(p.samples.inside.contains(ci));
            assert!(*ri >= 0.1 - 1e-15 && *ri <= 0.15 + 1e-15, "{ri}");
        }
        // Points are distinct.
        for a in 0..5 {
            for b in a + 1..5 {
                assert_ne!(c[a], c[b]);
            }
        }
    }

    #[test]
    fn repack_preserves_count_and_cools() {
        let mesh = shapes::box_mesh(DVec3::ZERO, DVec3::new(1.0, 1.2, 0.8));
        let mut p = toy_problem(4, 300, 300, 1000.0);
        p.targets = PhysicsTargets::from_mesh(&mesh, 1000.0);
        let cfg = Config::from_preset(Preset::B);
        let mut sp = toy_spheres(5, 10, false);
        // Make sure at least one sphere is inside and valuable.
        for c in &mut sp.centers {
            *c = c.clamp(DVec3::splat(0.05), DVec3::new(0.95, 1.15, 0.75));
        }
        let mut dc = DensityController::new(10, &cfg.training);
        let out = dc.repack(&p, &mesh, &mut sp, &cfg.training, &cfg.training.loss_weights());
        assert_eq!(sp.len(), 10);
        assert_eq!(out.added, out.removed);
        assert_eq!(out.removed, 4, "floor(10 * 0.4)");
        assert!((dc.temperature() - 0.34).abs() < 1e-12);
        for _ in 0..30 {
            dc.repack(&p, &mesh, &mut sp, &cfg.training, &cfg.training.loss_weights());
        }
        assert_eq!(dc.temperature(), MIN_TEMPERATURE);
        assert_eq!(sp.len(), 10);
    }

    #[test]
    fn k_is_clamped() {
        let mesh = shapes::box_mesh(DVec3::ZERO, DVec3::new(1.0, 1.2, 0.8));
        let p = toy_problem(6, 100, 100, 1.0);
        let cfg = Config::from_preset(Preset::B);
        let mut single = toy_spheres(7, 1, false);
        let mut dc = DensityController::new(1, &cfg.training);
        let out = dc.repack(&p, &mesh, &mut single, &cfg.training, &cfg.training.loss_weights());
        assert_eq!((out.added, out.removed), (0, 0));
        assert_eq!(dc.temperature(), INITIAL_TEMPERATURE, "a skipped pass does not cool");
    }

    #[test]
    fn warmup_moves_only_new_spheres_and_restores_survivors() {
        let p = toy_problem(8, 200, 200, 1000.0);
        let w = Preset::Obj.weights();
        let mut sp = toy_spheres(9, 6, true);
        let before = sp.clone();
        warmup(&p, &mut sp, &w, 4, 10, 0.01);
        assert_eq!(&sp.centers[..4], &before.centers[..4]);
        assert_eq!(&sp.raw_radii[..4], &before.raw_radii[..4]);
        assert_eq!(sp.raw_masses, before.raw_masses, "masses are not part of the warmup optimizer");
        assert_ne!(&sp.centers[4..], &before.centers[4..]);
    }
}
