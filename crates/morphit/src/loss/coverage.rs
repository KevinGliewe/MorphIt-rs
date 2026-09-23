//! Coverage loss: mean positive gap between interior samples and the nearest
//! sphere surface (L1, paper form). `losses.py::_compute_coverage_loss`.
//!
//! `L = (1/M) sum_i relu(min_j(|p_i - c_j| - r_j))`. Only the argmin sphere of
//! an uncovered sample receives gradient: `dL/dr = -1/M`,
//! `dL/dc = (1/M) (c - p) / |c - p|`.

use super::{Grads, LossInput};

pub(crate) fn coverage(inp: &LossInput, w: f64, g: &mut Grads) -> f64 {
    let rows = inp.inside_min;
    let samples = &inp.problem.samples.inside;
    let m = rows.len() as f64;
    let coef = w / m;
    let mut sum = 0.0;
    for (row, p) in rows.iter().zip(samples) {
        if row.gap > 0.0 {
            sum += row.gap;
            let j = row.j as usize;
            g.radii[j] -= coef;
            if row.d > 0.0 {
                g.centers[j] += (inp.centers[j] - *p) * (coef / row.d);
            }
        }
    }
    sum / m
}

#[cfg(test)]
mod tests {
    use super::super::fd::*;
    use super::super::{Problem, Samples, evaluate};
    use crate::config::LossId;
    use crate::state::Spheres;
    use glam::DVec3;

    #[test]
    fn hand_computed_value() {
        let p = toy_problem(1, 10, 10, 1.0);
        let samples = Samples {
            inside: vec![DVec3::new(0.0, 0.0, 0.0), DVec3::new(2.0, 0.0, 0.0), DVec3::new(0.5, 0.0, 0.0)],
            surface: vec![DVec3::ZERO],
            surface_normals: vec![DVec3::Z],
        };
        let p = Problem::cpu(samples, p.targets, p.density);
        // One sphere at the origin with radius 1: gaps are -1, 1, -0.5 -> mean relu = 1/3.
        let sp = Spheres::from_real(vec![DVec3::ZERO], &[1.0], None);
        let e = evaluate(&p, &sp, &only(LossId::Coverage));
        assert!((e.raw[0] - 1.0 / 3.0).abs() < 1e-12);
        // Uncovered sample at +x pulls the center toward it and grows the radius.
        assert!(e.grads.centers[0].x < 0.0);
        assert!(e.grads.raw_radii[0] < 0.0);
    }

    #[test]
    fn fd_check() {
        let p = toy_problem(11, 60, 60, 1.0);
        for seed in 0..3 {
            assert_fd(&p, &toy_spheres(seed, 5, false), &only(LossId::Coverage));
        }
    }
}
