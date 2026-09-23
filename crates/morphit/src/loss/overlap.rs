//! Overlap penalty: mean linear overlap depth over all ordered sphere pairs.
//! `losses.py::_compute_overlap_penalty`.
//!
//! `L = (1/N^2) sum_{j != k} relu(r_j + r_k - |c_j - c_k|)`. The mean runs over
//! all `N^2` entries (the diagonal is masked by `+1e6`), and each unordered pair
//! appears twice.

use super::{Grads, LossInput};

pub(crate) fn overlap(inp: &LossInput, w: f64, g: &mut Grads) -> f64 {
    let n = inp.centers.len();
    let nn = (n * n) as f64;
    let r = inp.radii;
    let c = inp.centers;
    let a = 2.0 * w / nn;
    let mut sum = 0.0;
    for j in 0..n {
        for k in j + 1..n {
            let d = inp.pair[j * n + k];
            let o = r[j] + r[k] - d;
            if o > 0.0 {
                sum += 2.0 * o;
                g.radii[j] += a;
                g.radii[k] += a;
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
        // Two spheres of radius 1 at distance 1.5 overlap by 0.5; mean over 4 entries = 2*0.5/4.
        let sp = Spheres::from_real(vec![DVec3::ZERO, DVec3::new(1.5, 0.0, 0.0)], &[1.0, 1.0], None);
        let e = evaluate(&p, &sp, &only(LossId::Overlap));
        assert!((e.raw[LossId::Overlap.index()] - 0.25).abs() < 1e-12);
        // Pushes the spheres apart.
        assert!(e.grads.centers[0].x > 0.0 && e.grads.centers[1].x < 0.0);
    }

    #[test]
    fn coincident_centers_have_finite_zero_center_gradient() {
        let p = toy_problem(1, 10, 10, 1.0);
        let sp = Spheres::from_real(vec![DVec3::ONE, DVec3::ONE], &[0.3, 0.2], None);
        let e = evaluate(&p, &sp, &only(LossId::Overlap));
        assert!(e.grads.centers.iter().all(|c| *c == DVec3::ZERO));
        assert!(e.grads.raw_radii.iter().all(|g| g.is_finite() && *g > 0.0));
    }

    #[test]
    fn fd_check() {
        let p = toy_problem(12, 30, 30, 1.0);
        for seed in 0..3 {
            assert_fd(&p, &toy_spheres(seed, 6, false), &only(LossId::Overlap));
        }
    }
}
