//! Containment loss: penalizes spheres nested inside other spheres.
//! `losses.py::_compute_containment_loss`.
//!
//! `depth[j,k] = r_k - (|c_j - c_k| + r_j)` (sphere `j` inside sphere `k`),
//! `L = (1/N^2) sum_{j != k} relu(depth)^2`.

use super::{Grads, LossInput};

pub(crate) fn containment(inp: &LossInput, w: f64, g: &mut Grads) -> f64 {
    let n = inp.centers.len();
    let nn = (n * n) as f64;
    let r = inp.radii;
    let c = inp.centers;
    let mut sum = 0.0;
    for j in 0..n {
        for k in 0..n {
            if j == k {
                continue;
            }
            let d = inp.pair[j * n + k];
            let depth = r[k] - (d + r[j]);
            if depth > 0.0 {
                sum += depth * depth;
                let a = 2.0 * w * depth / nn;
                g.radii[k] += a;
                g.radii[j] -= a;
                if d > 0.0 {
                    let u = (c[j] - c[k]) / d;
                    g.centers[j] -= u * a;
                    g.centers[k] += u * a;
                }
            }
        }
    }
    sum / nn
}

#[cfg(test)]
mod tests {
    use super::super::evaluate;
    use super::super::fd::*;
    use crate::config::LossId;
    use crate::state::Spheres;
    use glam::DVec3;

    #[test]
    fn hand_computed_value() {
        let p = toy_problem(1, 10, 10, 1.0);
        // Small sphere (r=0.2) 0.3 from the center of a big one (r=1): depth 0.5, squared 0.25, / 4.
        let sp = Spheres::from_real(vec![DVec3::ZERO, DVec3::new(0.3, 0.0, 0.0)], &[1.0, 0.2], None);
        let e = evaluate(&p, &sp, &only(LossId::Containment));
        assert!((e.raw[LossId::Containment.index()] - 0.0625).abs() < 1e-12);
    }

    #[test]
    fn coincident_centers_have_zero_center_gradient() {
        let p = toy_problem(1, 10, 10, 1.0);
        let sp = Spheres::from_real(vec![DVec3::ONE, DVec3::ONE], &[0.5, 0.1], None);
        let e = evaluate(&p, &sp, &only(LossId::Containment));
        assert!(e.grads.centers.iter().all(|c| *c == DVec3::ZERO));
        assert!(e.grads.raw_radii.iter().all(|g| g.is_finite()));
    }

    #[test]
    fn fd_check() {
        let p = toy_problem(16, 20, 20, 1.0);
        // Nested configuration so the loss is active.
        let sp = Spheres::from_real(
            vec![DVec3::new(0.5, 0.5, 0.4), DVec3::new(0.55, 0.45, 0.42), DVec3::new(0.3, 0.7, 0.4)],
            &[0.4, 0.1, 0.05],
            None,
        );
        assert_fd(&p, &sp, &only(LossId::Containment));
        for seed in 0..3 {
            assert_fd(&p, &toy_spheres(seed, 6, false), &only(LossId::Containment));
        }
    }
}
