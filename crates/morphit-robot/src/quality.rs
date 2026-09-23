//! Packing quality as the web UI reports it (the API's `_quality_metrics`):
//! unsigned distances from surface samples to the nearest sphere surface,
//! and the fraction of the mesh interior covered by spheres, estimated by
//! Monte-Carlo sampling. Random streams differ from numpy's, so values agree
//! with Python statistically, not digit for digit.

use morphit::Mesh;
use morphit::glam::DVec3;
use morphit::par::*;
use morphit::sampling::{MorphRng, sample_surface};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Surface samples for the distance metrics.
pub const ANALYZE_SURFACE_SAMPLES: usize = 8000;
/// Interior points to collect for coverage.
pub const ANALYZE_VOLUME_SAMPLES: usize = 8000;
/// Upper bound on bounding-box batches of [`ANALYZE_VOLUME_SAMPLES`] draws.
pub const ANALYZE_VOLUME_MAX_BATCHES: usize = 15;

/// Uniform points in the bounding box, kept when inside the mesh, in batches
/// until enough are inside or the batch budget is spent. Returns the inside
/// points and the volume estimate `bbox_volume * hits / draws`.
pub fn mc_interior_samples(mesh: &Mesh, seed: u64) -> (Vec<DVec3>, f64) {
    let (lo, hi) = mesh.bounds();
    let ext = hi - lo;
    let bbox_vol = ext.x * ext.y * ext.z;
    if bbox_vol <= 0.0 {
        return (Vec::new(), 0.0);
    }
    let mut rng = MorphRng::seed_from_u64(seed);
    let mut inside = Vec::new();
    let mut drawn = 0usize;
    for _ in 0..ANALYZE_VOLUME_MAX_BATCHES {
        let batch: Vec<DVec3> = (0..ANALYZE_VOLUME_SAMPLES)
            .map(|_| lo + ext * DVec3::new(rng.random(), rng.random(), rng.random()))
            .collect();
        drawn += batch.len();
        let mask = mesh.contains_many(&batch);
        inside.extend(batch.iter().zip(&mask).filter(|(_, m)| **m).map(|(p, _)| *p));
        if inside.len() >= ANALYZE_VOLUME_SAMPLES {
            break;
        }
    }
    let volume = bbox_vol * inside.len() as f64 / drawn as f64;
    (inside, volume)
}

/// Signed distance from each point to the nearest sphere surface
/// (`min_k |x - c_k| - r_k`; negative inside a sphere).
pub fn sphere_surface_dists(points: &[DVec3], centers: &[DVec3], radii: &[f64]) -> Vec<f64> {
    points
        .par_iter()
        .map(|&x| centers.iter().zip(radii).map(|(c, r)| x.distance(*c) - r).fold(f64::INFINITY, f64::min))
        .collect()
}

/// Sphere set echoed back for heat maps.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SphereSet {
    pub centers: Vec<[f64; 3]>,
    pub radii: Vec<f64>,
}

/// Metrics of one packed mesh (key order as in the Python response).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct LinkQuality {
    pub link_name: String,
    pub collision_index: usize,
    pub num_spheres: usize,
    pub d_min: f64,
    pub d_mean: f64,
    pub d_max: f64,
    /// Fraction of the interior samples inside some sphere.
    pub coverage: f64,
    pub area: f64,
    pub volume_est: f64,
    pub spheres: SphereSet,
}

impl LinkQuality {
    /// Whether every number is finite (Python refuses NaN in JSON).
    pub fn is_finite(&self) -> bool {
        [self.d_min, self.d_mean, self.d_max, self.coverage, self.area, self.volume_est]
            .iter()
            .all(|v| v.is_finite())
    }
}

/// Totals over several links.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Overall {
    pub num_spheres: usize,
    pub d_min: f64,
    /// Area-weighted mean of the per-link `d_mean`.
    pub d_mean: f64,
    pub d_max: f64,
    /// Volume-weighted mean of the per-link coverage.
    pub coverage: f64,
}

/// Quality of `centers`/`radii` against `mesh` (pass the prepared mesh).
pub fn quality_metrics(
    mesh: &Mesh,
    link_name: &str,
    collision_index: usize,
    centers: &[[f64; 3]],
    radii: &[f64],
) -> LinkQuality {
    let c: Vec<DVec3> = centers.iter().map(|p| DVec3::from_array(*p)).collect();
    let mut rng = MorphRng::seed_from_u64(0);
    let (surf, _) = sample_surface(mesh, ANALYZE_SURFACE_SAMPLES, &mut rng);
    let d: Vec<f64> = sphere_surface_dists(&surf, &c, radii).into_iter().map(f64::abs).collect();
    let (interior, volume_est) = mc_interior_samples(mesh, 0);
    let coverage = if interior.is_empty() {
        0.0
    } else {
        let covered = sphere_surface_dists(&interior, &c, radii).iter().filter(|&&v| v <= 0.0).count();
        covered as f64 / interior.len() as f64
    };
    let (d_min, d_max, d_mean) = if d.is_empty() {
        (f64::NAN, f64::NAN, f64::NAN)
    } else {
        (
            d.iter().copied().fold(f64::INFINITY, f64::min),
            d.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            d.iter().sum::<f64>() / d.len() as f64,
        )
    };
    LinkQuality {
        link_name: link_name.to_string(),
        collision_index,
        num_spheres: radii.len(),
        d_min,
        d_mean,
        d_max,
        coverage,
        area: mesh.area(),
        volume_est,
        spheres: SphereSet { centers: centers.to_vec(), radii: radii.to_vec() },
    }
}

/// Combine links: sums and extremes, area-weighted `d_mean`, volume-weighted
/// coverage (uniform weights when the areas or volumes sum to zero).
pub fn aggregate_overall(links: &[LinkQuality]) -> Overall {
    let n = links.len().max(1) as f64;
    let weights = |v: Vec<f64>| {
        let s: f64 = v.iter().sum();
        if s > 0.0 { v.iter().map(|x| x / s).collect::<Vec<_>>() } else { vec![1.0 / n; v.len()] }
    };
    let aw = weights(links.iter().map(|l| l.area).collect());
    let vw = weights(links.iter().map(|l| l.volume_est).collect());
    Overall {
        num_spheres: links.iter().map(|l| l.num_spheres).sum(),
        d_min: links.iter().map(|l| l.d_min).fold(f64::INFINITY, f64::min),
        d_mean: links.iter().zip(&aw).map(|(l, w)| w * l.d_mean).sum(),
        d_max: links.iter().map(|l| l.d_max).fold(f64::NEG_INFINITY, f64::max),
        coverage: links.iter().zip(&vw).map(|(l, w)| w * l.coverage).sum(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use morphit::shapes;

    #[test]
    fn cube_covered_by_its_circumsphere() {
        let mesh = shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        let (inside, vol) = mc_interior_samples(&mesh, 0);
        assert!(inside.len() >= ANALYZE_VOLUME_SAMPLES);
        assert!((vol - 1.0).abs() < 1e-12, "cube fills its box: {vol}");
        let r = 3f64.sqrt() / 2.0 + 1e-9;
        let q = quality_metrics(&mesh, "cube", 0, &[[0.5; 3]], &[r]);
        assert_eq!(q.coverage, 1.0);
        assert_eq!(q.num_spheres, 1);
        assert!((q.area - 6.0).abs() < 1e-12);
        // Surface points lie between the face centers (r - 0.5) and corners (0).
        assert!(q.d_max <= r - 0.5 + 1e-9 && q.d_min >= 0.0);
        assert!(q.is_finite());
        let json = serde_json::to_string(&q).unwrap();
        assert!(json.starts_with(r#"{"link_name":"cube","collision_index":0,"num_spheres":1,"d_min":"#));
        assert!(json.contains(r#""spheres":{"centers":[[0.5,0.5,0.5]],"radii":["#));

        // Half the cube: a small sphere at a corner covers little.
        let small = quality_metrics(&mesh, "cube", 0, &[[0.0; 3]], &[0.5]);
        assert!(small.coverage > 0.0 && small.coverage < 0.1, "{}", small.coverage);
    }

    #[test]
    fn overall_weights() {
        let mesh = shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        let mut a = quality_metrics(&mesh, "a", 0, &[[0.5; 3]], &[0.5]);
        let mut b = a.clone();
        a.area = 1.0;
        a.volume_est = 3.0;
        a.d_mean = 1.0;
        a.coverage = 1.0;
        b.area = 3.0;
        b.volume_est = 1.0;
        b.d_mean = 2.0;
        b.coverage = 0.0;
        b.d_max = 10.0;
        let o = aggregate_overall(&[a.clone(), b]);
        assert_eq!(o.num_spheres, 2);
        assert!((o.d_mean - 1.75).abs() < 1e-12);
        assert!((o.coverage - 0.75).abs() < 1e-12);
        assert_eq!(o.d_max, 10.0);
        a.area = 0.0;
        a.volume_est = 0.0;
        let o = aggregate_overall(&[a]);
        assert_eq!(o.d_mean, 1.0);
        assert_eq!(serde_json::to_value(&o).unwrap().as_object().unwrap().len(), 5);
    }
}
