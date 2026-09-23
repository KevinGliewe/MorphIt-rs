//! SQEM loss: squared signed distance along the surface normal to the
//! best-fitting sphere. `losses.py::_compute_sqem_loss`.
//!
//! `signed[i,j] = n_i . (s_i - c_j) - r_j`, `j* = argmin_j(|s_i - c_j| - r_j)`,
//! `L = (1/S) sum_i signed[i, j*]^2`. The argmin only selects the sphere
//! (`torch.gather`), so no gradient flows through the distance matrix.

use super::{Grads, LossInput};

pub(crate) fn sqem(inp: &LossInput, w: f64, g: &mut Grads) -> f64 {
    let rows = inp.surface_min;
    let samples = &inp.problem.samples.surface;
    let normals = &inp.problem.samples.surface_normals;
    let s = rows.len() as f64;
    let mut sum = 0.0;
    for ((row, sp), nrm) in rows.iter().zip(samples).zip(normals) {
        let j = row.j as usize;
        let diff = *sp - inp.centers[j];
        let signed = (diff.x * nrm.x + diff.y * nrm.y + diff.z * nrm.z) - inp.radii[j];
        sum += signed * signed;
        let a = 2.0 * w * signed / s;
        g.centers[j] -= *nrm * a;
        g.radii[j] -= a;
    }
    sum / s
}

#[cfg(test)]
mod tests {
    use super::super::evaluate;
    use super::super::fd::*;
    use crate::config::LossId;
    use glam::DVec3;

    #[test]
    fn fd_check() {
        let p = toy_problem(17, 30, 60, 1.0);
        for seed in 0..3 {
            assert_fd(&p, &toy_spheres(seed, 5, false), &only(LossId::Sqem));
        }
    }

    #[test]
    fn spheres_never_selected_get_no_gradient() {
        let p = toy_problem(18, 20, 40, 1.0);
        let mut sp = toy_spheres(3, 4, false);
        // A far-away, tiny sphere is never the argmin of any surface sample.
        sp.centers.push(DVec3::splat(50.0));
        sp.raw_radii.push(crate::math::inv_softplus(0.01));
        let e = evaluate(&p, &sp, &only(LossId::Sqem));
        assert_eq!(e.grads.centers[4], DVec3::ZERO);
        assert_eq!(e.grads.raw_radii[4], 0.0);
    }
}
