//! Soft Hausdorff loss: mean squared gap of the worst-covered 15% of surface
//! samples. `losses.py::_compute_hausdorff_surface_loss`.
//!
//! `h_i = relu(min_j(|s_i - c_j| - r_j))`, `k = max(1, floor(0.15 S))`,
//! `L = (1/k) sum_{i in topk(h)} h_i^2`. Ties at the k-th place resolve by
//! lower sample index (PyTorch leaves the order unspecified).

use super::{Grads, LossInput};
use crate::math::relu;

pub(crate) fn topk_count(s: usize) -> usize {
    ((s as f64 * 0.15) as usize).max(1)
}

pub(crate) fn hausdorff(inp: &LossInput, w: f64, g: &mut Grads) -> f64 {
    let rows = inp.surface_min;
    let samples = &inp.problem.samples.surface;
    let s = rows.len();
    let k = topk_count(s).min(s);
    let h: Vec<f64> = rows.iter().map(|r| relu(r.gap)).collect();
    let mut order: Vec<usize> = (0..s).collect();
    order.sort_by(|&a, &b| h[b].total_cmp(&h[a]).then(a.cmp(&b)));
    let kf = k as f64;
    let mut sum = 0.0;
    for &i in &order[..k] {
        let hi = h[i];
        sum += hi * hi;
        let row = rows[i];
        if row.gap > 0.0 {
            let a = 2.0 * w * hi / kf;
            let j = row.j as usize;
            g.radii[j] -= a;
            if row.d > 0.0 {
                g.centers[j] += (inp.centers[j] - samples[i]) * (a / row.d);
            }
        }
    }
    sum / kf
}

#[cfg(test)]
mod tests {
    use super::super::fd::*;
    use super::*;
    use crate::config::LossId;

    #[test]
    fn k_matches_python_truncation() {
        assert_eq!(topk_count(5000), 750);
        assert_eq!(topk_count(60), 9);
        assert_eq!(topk_count(3), 1);
        assert_eq!(topk_count(1), 1);
    }

    #[test]
    fn fd_check() {
        let p = toy_problem(19, 30, 60, 1.0);
        for seed in 0..3 {
            assert_fd(&p, &toy_spheres(seed, 5, false), &only(LossId::Hausdorff));
        }
    }
}
