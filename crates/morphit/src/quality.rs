//! Approximation-quality metrics, reproducing `scripts/debug_quick_eval.py`.
//!
//! All metrics use held-out samples drawn from their own seeded RNG, so they
//! are independent of the samples used during optimization.

use crate::par::*;
use glam::{DMat3, DVec3};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::distances::row_minima;
use crate::mesh::Mesh;
use crate::sampling::{MorphRng, sample_surface};
use crate::state::FOUR_THIRDS_PI;

/// Sampling settings for [`evaluate_packing`]; defaults match `debug_quick_eval.py`.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityOptions {
    pub seed: u64,
    pub surface_samples: usize,
    pub volume_samples: usize,
    /// Volume samples are drawn from the mesh bounding box scaled by this factor.
    pub bounds_expand: f64,
    /// Density used to derive sphere masses when none are given.
    pub density: f64,
}

impl Default for QualityOptions {
    fn default() -> Self {
        QualityOptions {
            seed: 0,
            surface_samples: 50_000,
            volume_samples: 50_000,
            bounds_expand: 1.5,
            density: 1000.0,
        }
    }
}

/// One row of the `debug_quick_eval` table.
#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct QualityMetrics {
    pub actual_n: usize,
    /// Spheres whose center lies outside the mesh.
    pub n_out: usize,
    /// Spheres with radius below 0.001 x mesh scale.
    pub n_tiny: usize,
    /// Volume covered by spheres and inside the mesh, relative to the mesh volume.
    pub r_in: f64,
    /// Volume covered by spheres but outside the mesh, relative to the mesh volume.
    pub r_out: f64,
    /// Union volume of the spheres relative to the mesh volume.
    pub r_uni: f64,
    /// Mean absolute surface distance, millimetres (mesh units x 1000).
    pub d_avg_mm: f64,
    /// Maximum absolute surface distance, millimetres.
    pub d_max_mm: f64,
    pub mass_abs: f64,
    pub mass_rel: f64,
    pub com_abs: f64,
    pub com_rel: f64,
    pub i_abs: f64,
    pub i_rel: f64,
}

/// Score a sphere packing against its mesh.
pub fn evaluate_packing(
    mesh: &Mesh,
    centers: &[DVec3],
    radii: &[f64],
    masses: Option<&[f64]>,
    opts: &QualityOptions,
) -> QualityMetrics {
    assert_eq!(centers.len(), radii.len());
    let mut rng = MorphRng::seed_from_u64(opts.seed);

    // Surface distance.
    let (surface, _) = sample_surface(mesh, opts.surface_samples, &mut rng);
    let gaps: Vec<f64> = row_minima(&surface, centers, radii).into_iter().map(|r| r.gap.abs()).collect();
    let d_max = gaps.iter().copied().fold(0.0, f64::max);
    let d_avg = gaps.iter().sum::<f64>() / gaps.len().max(1) as f64;

    // Volume overlap by Monte Carlo over the expanded bounding box.
    let (lo, hi) = mesh.bounds();
    let c = (lo + hi) * 0.5;
    let he = (hi - lo) * 0.5 * opts.bounds_expand;
    let box_lo = c - he;
    let size = he * 2.0;
    let sample_volume = size.x * size.y * size.z;
    let pts: Vec<DVec3> = (0..opts.volume_samples)
        .map(|_| box_lo + size * DVec3::new(rng.random(), rng.random(), rng.random()))
        .collect();
    let in_mesh = mesh.contains_robust_many(&pts);
    let r2: Vec<f64> = radii.iter().map(|r| r * r).collect();
    let in_sphere: Vec<bool> = pts
        .par_iter()
        .map(|p| centers.iter().zip(&r2).any(|(c, r2)| (*p - *c).length_squared() <= *r2))
        .collect();
    let n = pts.len().max(1) as f64;
    let sphere_count = in_sphere.iter().filter(|&&b| b).count() as f64;
    let both_count = in_sphere.iter().zip(&in_mesh).filter(|(s, m)| **s && **m).count() as f64;
    let sphere_vol = sphere_count / n * sample_volume;
    let both_vol = both_count / n * sample_volume;
    let mv = if mesh.volume() > 0.0 { mesh.volume() } else { 1e-12 };

    // Escapes.
    let inside = mesh.contains_robust_many(centers);
    let n_out = inside.iter().filter(|&&b| !b).count();
    let min_radius = 0.001 * mesh.scale();
    let n_tiny = radii.iter().filter(|&&r| r < min_radius).count();

    // Physics about the sphere body's own center of mass.
    let derived: Vec<f64>;
    let masses = match masses {
        Some(m) => m,
        None => {
            derived = radii.iter().map(|r| opts.density * FOUR_THIRDS_PI * r * r * r).collect();
            &derived
        }
    };
    let total: f64 = masses.iter().sum();
    let mut com = DVec3::ZERO;
    for (c, m) in centers.iter().zip(masses) {
        com += *c * *m;
    }
    com /= total;
    let mut inertia = DMat3::IDENTITY * masses.iter().zip(radii).map(|(m, r)| 0.4 * m * r * r).sum::<f64>();
    for (c, &m) in centers.iter().zip(masses) {
        let d = *c - com;
        inertia += (DMat3::IDENTITY * d.dot(d) - DMat3::from_cols(d * d.x, d * d.y, d * d.z)) * m;
    }
    let mesh_mass = mesh.volume() * opts.density;
    let mesh_inertia = mesh.moment_inertia() * opts.density;
    let fro =
        |m: DMat3| (m.x_axis.length_squared() + m.y_axis.length_squared() + m.z_axis.length_squared()).sqrt();
    let mass_abs = (total - mesh_mass).abs();
    let com_abs = (com - mesh.center_mass()).length();
    let i_abs = fro(inertia - mesh_inertia);

    QualityMetrics {
        actual_n: centers.len(),
        n_out,
        n_tiny,
        r_in: both_vol / mv,
        r_out: (sphere_vol - both_vol) / mv,
        r_uni: sphere_vol / mv,
        d_avg_mm: d_avg * 1000.0,
        d_max_mm: d_max * 1000.0,
        mass_abs,
        mass_rel: mass_abs / mesh_mass.max(1e-20),
        com_abs,
        com_rel: com_abs / mesh.scale().max(1e-20),
        i_abs,
        i_rel: i_abs / fro(mesh_inertia).max(1e-20),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes;

    #[test]
    fn sphere_inside_a_box() {
        let cube = shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        let r = 0.3;
        let q = evaluate_packing(&cube, &[DVec3::splat(0.5)], &[r], None, &QualityOptions::default());
        let expect = FOUR_THIRDS_PI * r * r * r;
        assert!((q.r_uni - expect).abs() < 0.01, "{}", q.r_uni);
        assert!((q.r_in - expect).abs() < 0.01);
        assert!(q.r_out.abs() < 1e-12);
        assert_eq!((q.n_out, q.n_tiny, q.actual_n), (0, 0, 1));
        // Distance from the cube surface to the sphere: 0.2 at face centers, up to
        // |corner - center| - r = 0.566 at corners.
        assert!(q.d_max_mm > 500.0 && q.d_max_mm < 567.0);
        assert!(q.d_avg_mm > 200.0);
    }

    #[test]
    fn sphere_centered_on_a_face_is_half_outside() {
        let cube = shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        let r = 0.25;
        let q = evaluate_packing(&cube, &[DVec3::new(0.5, 0.5, 1.0)], &[r], None, &QualityOptions::default());
        let half = 0.5 * FOUR_THIRDS_PI * r * r * r;
        assert!((q.r_in - half).abs() < 0.006, "{}", q.r_in);
        assert!((q.r_out - half).abs() < 0.006, "{}", q.r_out);
        assert_eq!(q.n_out, 0, "a center exactly on the face is classified inside by the robust test");
    }

    #[test]
    fn physics_of_a_matching_ball() {
        let ball = shapes::uv_sphere(DVec3::new(0.1, 0.2, 0.3), 0.5, 64, 128);
        let r = (ball.volume() / FOUR_THIRDS_PI).cbrt();
        let opts = QualityOptions { surface_samples: 2000, volume_samples: 2000, ..Default::default() };
        let q = evaluate_packing(&ball, &[ball.center_mass()], &[r], None, &opts);
        assert!(q.mass_rel < 1e-12);
        assert!(q.com_rel < 1e-12);
        assert!(q.i_rel < 0.01);
        let q2 = evaluate_packing(&ball, &[ball.center_mass()], &[r], Some(&[ball.volume() * 2000.0]), &opts);
        assert!((q2.mass_rel - 1.0).abs() < 1e-12);
    }
}
