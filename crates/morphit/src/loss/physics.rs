//! Physics losses: total mass, center of mass and inertia tensor, each
//! normalized by the mesh ground truth. `losses.py::_compute_{mass,com,inertia}_loss`.

use glam::{DMat3, DVec3};

use super::{Grads, LossInput, frobenius_sq};

/// `L = (sum m - M)^2 / (M^2 + eps)`.
pub(crate) fn mass(inp: &LossInput, w: f64, g: &mut Grads) -> f64 {
    let t = &inp.problem.targets;
    let total: f64 = inp.masses.iter().sum();
    let diff = total - t.mass;
    let coef = 2.0 * w * diff / t.mass_norm_sq;
    for gm in &mut g.masses {
        *gm += coef;
    }
    diff * diff / t.mass_norm_sq
}

fn weighted_com(centers: &[DVec3], masses: &[f64]) -> (f64, DVec3) {
    let total: f64 = masses.iter().sum();
    let mut acc = DVec3::ZERO;
    for (c, &m) in centers.iter().zip(masses) {
        acc += *c * m;
    }
    (total, acc / total)
}

/// `L = |com - com_mesh|^2 / (scale^2 + eps)` with `com = sum m c / sum m`.
pub(crate) fn com(inp: &LossInput, w: f64, g: &mut Grads) -> f64 {
    let t = &inp.problem.targets;
    let (total, com) = weighted_com(inp.centers, inp.masses);
    let err = com - t.com;
    let value = err.length_squared() / t.com_norm_sq;
    let e = err * (2.0 * w / t.com_norm_sq);
    for (j, (&c, &m)) in inp.centers.iter().zip(inp.masses).enumerate() {
        g.centers[j] += e * (m / total);
        g.masses[j] += e.dot(c - com) / total;
    }
    value
}

/// Frobenius-relative inertia error about the sphere body's own COM:
/// `I = (sum 0.4 m r^2 + sum m |d|^2) I3 - sum m d d^T` with `d = c - com`,
/// `L = |I - I_mesh|_F^2 / (|I_mesh|_F^2 + eps)`.
///
/// With `G = dL/dI` and `H = tr(G) I3 - G`: `dL/dr_j = 0.8 tr(G) m_j r_j`,
/// `dL/dc_j = 2 m_j H d_j`, `dL/dm_j = tr(G)(0.4 r_j^2 + |d_j|^2) - d_j^T G d_j`.
/// The dependence of the COM on `c` and `m` cancels because `sum m_j d_j = 0`.
pub(crate) fn inertia(inp: &LossInput, w: f64, g: &mut Grads) -> f64 {
    let t = &inp.problem.targets;
    let (_, com) = weighted_com(inp.centers, inp.masses);
    let mut body = 0.0;
    let mut trace = 0.0;
    let mut outer = DMat3::ZERO;
    for ((&c, &m), &r) in inp.centers.iter().zip(inp.masses).zip(inp.radii) {
        body += 0.4 * m * (r * r);
        let d = c - com;
        trace += m * d.length_squared();
        let md = d * m;
        outer += DMat3::from_cols(md * d.x, md * d.y, md * d.z);
    }
    let inertia = DMat3::IDENTITY * (body + trace) - outer;
    let diff = inertia - t.inertia;
    let value = frobenius_sq(&diff) / t.inertia_norm_sq;

    let gm = diff * (2.0 * w / t.inertia_norm_sq);
    let tg = gm.x_axis.x + gm.y_axis.y + gm.z_axis.z;
    let h = DMat3::IDENTITY * tg - gm;
    for (j, ((&c, &m), &r)) in inp.centers.iter().zip(inp.masses).zip(inp.radii).enumerate() {
        let d = c - com;
        g.radii[j] += 0.8 * tg * m * r;
        g.centers[j] += (h * d) * (2.0 * m);
        g.masses[j] += tg * (0.4 * r * r + d.length_squared()) - d.dot(gm * d);
    }
    value
}

#[cfg(test)]
mod tests {
    use super::super::evaluate;
    use super::super::fd::*;
    use crate::config::LossId;
    use crate::state::{FOUR_THIRDS_PI, Spheres};
    use glam::DVec3;

    #[test]
    fn fd_check_mass_com_inertia() {
        for density in [1.0, 1000.0] {
            let p = toy_problem(22, 10, 10, density);
            for id in [LossId::Mass, LossId::Com, LossId::Inertia] {
                for psm in [false, true] {
                    for seed in 0..2 {
                        let mut sp = toy_spheres(seed, 5, psm);
                        if psm {
                            // Keep learned masses on the same scale as the target.
                            let scale = density / 1000.0;
                            for m in sp.raw_masses.as_mut().unwrap() {
                                *m = crate::math::inv_softplus(crate::math::softplus(*m) * scale);
                            }
                        }
                        assert_fd(&p, &sp, &only(id));
                    }
                }
            }
        }
    }

    #[test]
    fn single_sphere_matching_a_ball_has_small_errors() {
        // A mesh ball vs one sphere of equal volume at its center.
        let mesh = crate::shapes::uv_sphere(DVec3::new(0.2, 0.0, 0.1), 0.5, 64, 128);
        let mut p = toy_problem(1, 10, 10, 1000.0);
        p.targets = super::super::PhysicsTargets::from_mesh(&mesh, 1000.0);
        let r = (mesh.volume() / FOUR_THIRDS_PI).cbrt();
        let sp = Spheres::from_real(vec![mesh.center_mass()], &[r], None);
        let mut w = only(LossId::Mass);
        w[LossId::Com] = 1.0;
        w[LossId::Inertia] = 1.0;
        let e = evaluate(&p, &sp, &w);
        assert!(e.raw[LossId::Mass.index()] < 1e-20);
        assert!(e.raw[LossId::Com.index()] < 1e-20);
        assert!(e.raw[LossId::Inertia.index()] < 1e-4);
    }
}
