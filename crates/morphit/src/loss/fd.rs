//! Test-only helpers: toy problems and a central finite-difference gradient check.

use glam::DVec3;
use rand::{Rng, SeedableRng};

use super::{PhysicsTargets, Problem, Samples, evaluate};
use crate::config::{LossId, LossWeights};
use crate::math::inv_softplus;
use crate::sampling::{MorphRng, sample_inside, sample_surface};
use crate::shapes;
use crate::state::Spheres;

/// A box-shaped problem with `m` interior and `s` surface samples.
pub(crate) fn toy_problem(seed: u64, m: usize, s: usize, density: f64) -> Problem {
    let mesh = shapes::box_mesh(DVec3::ZERO, DVec3::new(1.0, 1.2, 0.8));
    let mut rng = MorphRng::seed_from_u64(seed);
    let inside = sample_inside(&mesh, m, &mut rng).unwrap();
    let (surface, ids) = sample_surface(&mesh, s, &mut rng);
    let surface_normals = ids.iter().map(|&f| mesh.face_normals()[f as usize]).collect();
    Problem::cpu(
        Samples { inside, surface, surface_normals },
        PhysicsTargets::from_mesh(&mesh, density),
        density,
    )
}

/// `n` random spheres roughly inside the toy box; with `psm`, learned masses.
pub(crate) fn toy_spheres(seed: u64, n: usize, psm: bool) -> Spheres {
    let mut rng = MorphRng::seed_from_u64(seed);
    let centers = (0..n)
        .map(|_| {
            DVec3::new(rng.random_range(-0.1..1.1), rng.random_range(-0.1..1.3), rng.random_range(-0.1..0.9))
        })
        .collect();
    let radii: Vec<f64> = (0..n).map(|_| rng.random_range(0.08..0.35)).collect();
    let masses: Vec<f64> = (0..n).map(|_| rng.random_range(10.0..200.0)).collect();
    Spheres {
        centers,
        raw_radii: radii.iter().map(|&r| inv_softplus(r)).collect(),
        raw_masses: psm.then(|| masses.iter().map(|&m| inv_softplus(m)).collect()),
    }
}

/// Only `id` weighted (by 1).
pub(crate) fn only(id: LossId) -> LossWeights {
    let mut w = LossWeights::default();
    w[id] = 1.0;
    w
}

/// Compare analytic raw gradients with central differences of the total loss.
pub(crate) fn assert_fd(problem: &Problem, spheres: &Spheres, weights: &LossWeights) {
    let h = 1e-6;
    let base = evaluate(problem, spheres, weights);
    let f = |sp: &Spheres| evaluate(problem, sp, weights).total;
    let check = |what: String, analytic: f64, fd: f64| {
        let tol = 1e-7 + 1e-5 * analytic.abs().max(fd.abs());
        assert!((analytic - fd).abs() <= tol, "{what}: analytic {analytic:e} vs fd {fd:e}");
    };
    for j in 0..spheres.len() {
        for k in 0..3 {
            let mut plus = spheres.clone();
            plus.centers[j][k] += h;
            let mut minus = spheres.clone();
            minus.centers[j][k] -= h;
            let fd = (f(&plus) - f(&minus)) / (2.0 * h);
            check(format!("center[{j}][{k}]"), base.grads.centers[j][k], fd);
        }
        let mut plus = spheres.clone();
        plus.raw_radii[j] += h;
        let mut minus = spheres.clone();
        minus.raw_radii[j] -= h;
        let fd = (f(&plus) - f(&minus)) / (2.0 * h);
        check(format!("raw_radius[{j}]"), base.grads.raw_radii[j], fd);
        if spheres.raw_masses.is_some() {
            let mut plus = spheres.clone();
            plus.raw_masses.as_mut().unwrap()[j] += h;
            let mut minus = spheres.clone();
            minus.raw_masses.as_mut().unwrap()[j] -= h;
            let fd = (f(&plus) - f(&minus)) / (2.0 * h);
            check(format!("raw_mass[{j}]"), base.grads.raw_masses.as_ref().unwrap()[j], fd);
        }
    }
}
