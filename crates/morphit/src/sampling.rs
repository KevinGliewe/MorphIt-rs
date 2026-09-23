//! Random initialization: voxel-grid sphere centers, log-normal radii, and
//! interior / surface sample points (`morphit.py::_initialize_*`).
//!
//! Every function draws from the caller's RNG so a seed determines the result.

use glam::DVec3;
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, LogNormal};

use crate::error::{Error, Result};
use crate::mesh::Mesh;

/// RNG used throughout the crate.
pub type MorphRng = ChaCha8Rng;

/// Refuse to materialize voxel grids larger than this (pathologically thin meshes).
const MAX_GRID_POINTS: usize = 20_000_000;

/// Result of the voxel-grid initializer.
#[derive(Clone, Debug)]
pub struct VoxelInit {
    pub centers: Vec<DVec3>,
    /// Final voxel edge length.
    pub voxel_size: f64,
    /// Inside cells found before random sub-selection.
    pub candidates: usize,
}

/// `_voxel_sample_centers`: place `n` centers at inside cell centers of a
/// uniform grid whose cell size is auto-tuned so at least `n` cells are inside.
pub fn voxel_grid_centers(mesh: &Mesh, n: usize, rng: &mut MorphRng) -> VoxelInit {
    let safety = 1.0;
    let max_iters = 10;
    let (lo, hi) = mesh.bounds();
    let extent = hi - lo;
    let scale = mesh.scale();

    let mut voxel = (mesh.volume() / (n as f64 * safety)).cbrt();
    voxel = voxel.max(scale * 1e-3).min(scale);

    let mut inside: Vec<DVec3> = Vec::new();
    for _ in 0..max_iters {
        let n_axis = [0, 1, 2].map(|i| ((extent[i] / voxel - 1e-9).ceil() as usize).max(1));
        let total = n_axis[0].saturating_mul(n_axis[1]).saturating_mul(n_axis[2]);
        if total > MAX_GRID_POINTS {
            tracing::warn!(voxel, total, "voxel grid too large; falling back to volume sampling");
            break;
        }
        let axes: Vec<Vec<f64>> =
            (0..3).map(|i| (0..n_axis[i]).map(|k| lo[i] + (k as f64 + 0.5) * voxel).collect()).collect();
        let mut grid = Vec::with_capacity(total);
        for &x in &axes[0] {
            for &y in &axes[1] {
                for &z in &axes[2] {
                    grid.push(DVec3::new(x, y, z));
                }
            }
        }
        let mask = mesh.contains_robust_many(&grid);
        inside = grid.into_iter().zip(mask).filter_map(|(p, m)| m.then_some(p)).collect();
        if inside.len() >= n {
            break;
        }
        voxel *= 0.75;
    }

    if inside.len() < n {
        // Too thin or concave for voxels: top up with uniform volume samples,
        // then with surface samples pulled 5% toward the centroid.
        let needed = n - inside.len();
        let count = (needed * 3).max(100);
        let mut fallback: Vec<DVec3> =
            (0..count).map(|_| lo + DVec3::new(rng.random(), rng.random(), rng.random()) * extent).collect();
        let mask = mesh.contains_robust_many(&fallback);
        fallback = fallback.into_iter().zip(mask).filter_map(|(p, m)| m.then_some(p)).collect();
        if fallback.len() < needed {
            let (surf, _) = sample_surface(mesh, needed * 2, rng);
            let com = mesh.center_mass();
            fallback.extend(surf.into_iter().map(|s| s + 0.05 * (com - s)));
        }
        inside.extend(fallback.into_iter().take(needed));
    }

    let candidates = inside.len();
    if candidates > n {
        let idx = rand::seq::index::sample(rng, candidates, n).into_vec();
        inside = idx.into_iter().map(|i| inside[i]).collect();
    }
    tracing::debug!(voxel, candidates, selected = n, "voxel init");
    VoxelInit { centers: inside, voxel_size: voxel, candidates }
}

/// `_initialize_radii_with_variation`: log-normal radii scaled so the total
/// sphere volume equals `volume`.
pub fn lognormal_radii(volume: f64, n: usize, sigma: f64, rng: &mut MorphRng) -> Vec<f64> {
    let target = volume / n as f64;
    let mean_radius = (3.0 * target / (4.0 * std::f64::consts::PI)).cbrt();
    let dist = LogNormal::new(0.0, sigma).expect("sigma validated as finite and >= 0");
    let samples: Vec<f64> = (0..n).map(|_| dist.sample(rng)).collect();
    let cube_sum: f64 = samples.iter().map(|x| x * x * x).sum();
    let factor = (n as f64 / cube_sum).cbrt();
    samples.into_iter().map(|x| mean_radius * (x * factor)).collect()
}

/// `_initialize_inside_samples`: rejection-sample `n` interior points using the
/// axis-aligned containment test, in batches of `min(2n, 10000)`.
pub fn sample_inside(mesh: &Mesh, n: usize, rng: &mut MorphRng) -> Result<Vec<DVec3>> {
    let (lo, hi) = mesh.bounds();
    let extent = hi - lo;
    let batch = (n * 2).clamp(1, 10_000);
    let max_batches = 10_000;
    let mut points = Vec::with_capacity(n);
    for _ in 0..max_batches {
        let cand: Vec<DVec3> =
            (0..batch).map(|_| lo + extent * DVec3::new(rng.random(), rng.random(), rng.random())).collect();
        let mask = mesh.contains_many(&cand);
        points.extend(cand.into_iter().zip(mask).filter_map(|(p, m)| m.then_some(p)));
        if points.len() >= n {
            points.truncate(n);
            return Ok(points);
        }
    }
    Err(Error::Mesh(format!(
        "found only {} of {n} interior sample points; is the mesh closed?",
        points.len()
    )))
}

/// `trimesh.sample.sample_surface`: area-weighted uniform surface samples.
/// Returns the points and the index of the face each one lies on.
pub fn sample_surface(mesh: &Mesh, n: usize, rng: &mut MorphRng) -> (Vec<DVec3>, Vec<u32>) {
    let areas = mesh.face_areas();
    let mut cum = Vec::with_capacity(areas.len());
    let mut acc = 0.0;
    for a in areas {
        acc += a;
        cum.push(acc);
    }
    let total = acc;
    let picks: Vec<f64> = (0..n).map(|_| rng.random::<f64>() * total).collect();
    let face_ids: Vec<u32> =
        picks.iter().map(|&p| cum.partition_point(|&c| c < p).min(areas.len() - 1) as u32).collect();
    let faces = mesh.faces();
    let verts = mesh.vertices();
    let points = face_ids
        .iter()
        .map(|&f| {
            let [i0, i1, i2] = faces[f as usize];
            let origin = verts[i0 as usize];
            let e1 = verts[i1 as usize] - origin;
            let e2 = verts[i2 as usize] - origin;
            let (mut a, mut b): (f64, f64) = (rng.random(), rng.random());
            if a + b > 1.0 {
                a = (a - 1.0).abs();
                b = (b - 1.0).abs();
            }
            e1 * a + e2 * b + origin
        })
        .collect();
    (points, face_ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes;
    use rand::SeedableRng;

    fn rng(seed: u64) -> MorphRng {
        MorphRng::seed_from_u64(seed)
    }

    #[test]
    fn cube_lattices_are_exact() {
        let cube = shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        for k in [3usize, 4] {
            let init = voxel_grid_centers(&cube, k * k * k, &mut rng(0));
            assert_eq!(init.centers.len(), k * k * k);
            assert_eq!(init.candidates, k * k * k, "k={k}");
            assert!((init.voxel_size - 1.0 / k as f64).abs() < 1e-12);
            let mut got: Vec<[i64; 3]> =
                init.centers.iter().map(|c| (*c * k as f64 - 0.5).round().as_i64vec3().to_array()).collect();
            got.sort();
            let mut want = Vec::new();
            for x in 0..k as i64 {
                for y in 0..k as i64 {
                    for z in 0..k as i64 {
                        want.push([x, y, z]);
                    }
                }
            }
            assert_eq!(got, want);
        }
    }

    #[test]
    fn thin_slab_shrinks_voxels() {
        let slab = shapes::box_mesh(DVec3::ZERO, DVec3::new(1.0, 1.0, 0.02));
        let init = voxel_grid_centers(&slab, 20, &mut rng(1));
        assert_eq!(init.centers.len(), 20);
        assert!(init.centers.iter().all(|c| slab.contains_robust(*c)));
    }

    #[test]
    fn fallback_fills_the_budget() {
        // Asking for more spheres than 10 shrink rounds can produce forces the fallback.
        let cube = shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        let init = voxel_grid_centers(&cube, 5000, &mut rng(2));
        assert_eq!(init.centers.len(), 5000);
    }

    #[test]
    fn radii_preserve_volume() {
        let r = lognormal_radii(2.5, 37, 0.3, &mut rng(3));
        let vol: f64 = r.iter().map(|r| 4.0 / 3.0 * std::f64::consts::PI * r * r * r).sum();
        assert!((vol - 2.5).abs() < 1e-9);
        let flat = lognormal_radii(1.0, 8, 0.0, &mut rng(3));
        assert!(flat.windows(2).all(|w| (w[0] - w[1]).abs() < 1e-15));
    }

    #[test]
    fn inside_samples_are_inside() {
        let s = shapes::uv_sphere(DVec3::ZERO, 1.0, 24, 48);
        let pts = sample_inside(&s, 3000, &mut rng(4)).unwrap();
        assert_eq!(pts.len(), 3000);
        assert!(pts.iter().all(|p| p.length() < 1.0));
    }

    #[test]
    fn surface_samples_lie_on_their_faces() {
        let cube = shapes::box_mesh(DVec3::ZERO, DVec3::new(2.0, 1.0, 1.0));
        let (pts, ids) = sample_surface(&cube, 20_000, &mut rng(5));
        for (p, &f) in pts.iter().zip(&ids) {
            let [a, _, _] = cube.faces()[f as usize].map(|i| cube.vertices()[i as usize]);
            let n = cube.face_normals()[f as usize];
            assert!((p - a).dot(n).abs() < 1e-12);
        }
        // Faces of area 1 (2 of the 12 triangles of area 0.5 per x-face pair) vs area 2 faces:
        // expect the share of samples per face proportional to area.
        let mut per_face = [0usize; 12];
        for &f in &ids {
            per_face[f as usize] += 1;
        }
        let total_area = cube.area();
        for (f, &c) in per_face.iter().enumerate() {
            let expect = 20_000.0 * cube.face_areas()[f] / total_area;
            assert!((c as f64 - expect).abs() < 5.0 * expect.sqrt(), "face {f}: {c} vs {expect}");
        }
    }

    #[test]
    fn seeds_are_deterministic() {
        let s = shapes::uv_sphere(DVec3::ZERO, 1.0, 12, 24);
        let a = sample_inside(&s, 100, &mut rng(9)).unwrap();
        let b = sample_inside(&s, 100, &mut rng(9)).unwrap();
        assert_eq!(a, b);
        let c = sample_inside(&s, 100, &mut rng(10)).unwrap();
        assert_ne!(a, c);
    }
}
