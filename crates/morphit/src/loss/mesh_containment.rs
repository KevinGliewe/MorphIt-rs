//! Mesh containment loss: penalizes sphere centers outside the mesh.
//! `losses.py::_compute_mesh_containment_loss`.
//!
//! For each center, take its nearest surface sample `s` with normal `n`;
//! `sd = n . (c - s)` is positive outside. `L = (1/N) sum_j relu(sd_j)^2`.
//! Only centers receive gradient (the nearest-sample index is constant).

use super::{Grads, LossInput};

pub(crate) fn mesh_containment(inp: &LossInput, w: f64, g: &mut Grads) -> f64 {
    let surface = &inp.problem.samples.surface;
    let normals = &inp.problem.samples.surface_normals;
    let n = inp.centers.len() as f64;
    let mut sum = 0.0;
    for (j, (&c, &i)) in inp.centers.iter().zip(inp.nearest_surface).enumerate() {
        let s = surface[i as usize];
        let nrm = normals[i as usize];
        let v = c - s;
        let sd = v.x * nrm.x + v.y * nrm.y + v.z * nrm.z;
        if sd > 0.0 {
            sum += sd * sd;
            g.centers[j] += nrm * (2.0 * w * sd / n);
        }
    }
    sum / n
}

#[cfg(test)]
mod tests {
    use super::super::evaluate;
    use super::super::fd::*;
    use crate::config::LossId;

    #[test]
    fn fd_check_and_no_radius_gradient() {
        let p = toy_problem(20, 20, 200, 1.0);
        for seed in 0..3 {
            let sp = toy_spheres(seed, 6, false);
            assert_fd(&p, &sp, &only(LossId::MeshContainment));
            let e = evaluate(&p, &sp, &only(LossId::MeshContainment));
            assert!(e.grads.raw_radii.iter().all(|&g| g == 0.0));
        }
    }

    #[test]
    fn escaped_center_is_pulled_back() {
        let p = toy_problem(21, 20, 400, 1.0);
        let mut sp = toy_spheres(1, 1, false);
        sp.centers[0] = glam::DVec3::new(0.5, 0.6, 1.5); // above the box top (z = 0.8)
        let e = evaluate(&p, &sp, &only(LossId::MeshContainment));
        assert!(e.raw[LossId::MeshContainment.index()] > 0.0);
        assert!(e.grads.centers[0].z > 0.0, "gradient points outward, descent moves inward");
    }
}
