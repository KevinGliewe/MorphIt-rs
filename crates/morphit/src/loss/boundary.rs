//! Boundary penalty: mean depth by which spheres engulf surface samples.
//! `losses.py::_compute_boundary_penalty`.
//!
//! `L = (1/(S N)) sum_{i,j} relu(r_j - |s_i - c_j|)` over all pairs (no
//! minimum). Only the pairs the search reports as candidates can be active;
//! each is re-tested exactly here. Computed per sphere in parallel over its
//! candidate samples in ascending order, then reduced in sphere order, so the
//! result depends neither on the thread count nor on the backend.

use crate::par::*;
use glam::DVec3;

use super::{Grads, LossInput};
use crate::distances::dist;

pub(crate) fn boundary(inp: &LossInput, w: f64, g: &mut Grads) -> f64 {
    let surface = &inp.problem.samples.surface;
    let s = surface.len();
    let n = inp.centers.len();
    let coef = w / (s as f64 * n as f64);
    let pairs = inp.boundary;
    let cols: Vec<(f64, f64, DVec3)> = (0..n)
        .into_par_iter()
        .map(|j| {
            let c = inp.centers[j];
            let r = inp.radii[j];
            let mut sum = 0.0;
            let mut dr = 0.0;
            let mut dc = DVec3::ZERO;
            for &i in pairs.of(j) {
                let sp = &surface[i as usize];
                let d = dist(*sp, c);
                let v = -(d - r);
                if v > 0.0 {
                    sum += v;
                    dr += coef;
                    if d > 0.0 {
                        dc -= (c - *sp) * (coef / d);
                    }
                }
            }
            (sum, dr, dc)
        })
        .collect();
    let mut total = 0.0;
    for (j, (sum, dr, dc)) in cols.into_iter().enumerate() {
        total += sum;
        g.radii[j] += dr;
        g.centers[j] += dc;
    }
    total / (s as f64 * n as f64)
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "parallel")]
    use super::super::evaluate;
    use super::super::fd::*;
    use crate::config::LossId;

    #[test]
    fn fd_check() {
        let p = toy_problem(13, 30, 80, 1.0);
        for seed in 0..3 {
            assert_fd(&p, &toy_spheres(seed, 5, false), &only(LossId::Boundary));
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn deterministic_across_threads() {
        let p = toy_problem(14, 30, 2000, 1.0);
        let sp = toy_spheres(1, 40, false);
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let many = rayon::ThreadPoolBuilder::new().num_threads(5).build().unwrap();
        let a = one.install(|| evaluate(&p, &sp, &only(LossId::Boundary)));
        let b = many.install(|| evaluate(&p, &sp, &only(LossId::Boundary)));
        assert_eq!(a, b);
    }
}
