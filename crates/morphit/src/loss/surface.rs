//! Surface loss: mean absolute gap between surface samples and the nearest
//! sphere surface (L1). `losses.py::_compute_surface_loss`.
//!
//! `L = (1/S) sum_i |min_j(|s_i - c_j| - r_j)|`; the argmin sphere receives
//! `sign(gap) / S` through the gap (with `sign(0) = 0`).

use super::{Grads, LossInput};
use crate::math::sgn;

pub(crate) fn surface(inp: &LossInput, w: f64, g: &mut Grads) -> f64 {
    let rows = inp.surface_min;
    let samples = &inp.problem.samples.surface;
    let s = rows.len() as f64;
    let mut sum = 0.0;
    for (row, p) in rows.iter().zip(samples) {
        sum += row.gap.abs();
        let sg = sgn(row.gap);
        if sg != 0.0 {
            let j = row.j as usize;
            let coef = w * sg / s;
            g.radii[j] -= coef;
            if row.d > 0.0 {
                g.centers[j] += (inp.centers[j] - *p) * (coef / row.d);
            }
        }
    }
    sum / s
}

#[cfg(test)]
mod tests {
    use super::super::fd::*;
    use crate::config::LossId;

    #[test]
    fn fd_check() {
        let p = toy_problem(15, 30, 60, 1.0);
        for seed in 0..3 {
            assert_fd(&p, &toy_spheres(seed, 5, false), &only(LossId::Surface));
        }
    }
}
