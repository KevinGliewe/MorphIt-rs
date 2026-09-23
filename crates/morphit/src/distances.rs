//! Distance data shared by the loss terms, computed once per evaluation.
//!
//! Heavy work runs in parallel over independent rows; every reduction that
//! crosses rows happens later in a fixed serial order, so results are
//! bit-identical regardless of the number of threads.

use crate::par::*;
use glam::DVec3;

/// Euclidean distance, always evaluated as `|sample - center|` so every call
/// site produces bit-identical values for the same pair.
#[inline]
pub fn dist(sample: DVec3, center: DVec3) -> f64 {
    (sample - center).length()
}

/// Per-sample minimum of `|p - c_j| - r_j` over spheres.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RowMin {
    /// `min_j (d_j - r_j)`.
    pub gap: f64,
    /// Distance to the argmin sphere.
    pub d: f64,
    /// Argmin sphere (first index on ties, like `torch.min(dim)` on CPU).
    pub j: u32,
}

/// Nearest sphere for one sample (first index on ties).
#[inline]
pub(crate) fn row_min(p: DVec3, centers: &[DVec3], radii: &[f64]) -> RowMin {
    let mut best = RowMin { gap: f64::INFINITY, d: 0.0, j: 0 };
    let mut first = true;
    for (j, (c, r)) in centers.iter().zip(radii).enumerate() {
        let d = dist(p, *c);
        let gap = d - r;
        if first || gap < best.gap {
            best = RowMin { gap, d, j: j as u32 };
            first = false;
        }
    }
    best
}

/// Row minima for every sample (parallel over samples).
pub fn row_minima(samples: &[DVec3], centers: &[DVec3], radii: &[f64]) -> Vec<RowMin> {
    samples.par_iter().map(|&p| row_min(p, centers, radii)).collect()
}

/// Index of the nearest sample to one center (first index on ties).
#[inline]
pub(crate) fn nearest_one(c: DVec3, samples: &[DVec3]) -> u32 {
    let mut best = 0u32;
    let mut best_d = f64::INFINITY;
    for (i, &s) in samples.iter().enumerate() {
        let d = dist(s, c);
        if i == 0 || d < best_d {
            best = i as u32;
            best_d = d;
        }
    }
    best
}

/// For every center, the index of the nearest sample (first index on ties).
pub fn nearest_sample_per_center(centers: &[DVec3], samples: &[DVec3]) -> Vec<u32> {
    centers.par_iter().map(|&c| nearest_one(c, samples)).collect()
}

/// Symmetric `N x N` center distance matrix (row-major, zero diagonal).
pub fn pairwise(centers: &[DVec3]) -> Vec<f64> {
    let n = centers.len();
    let mut out = vec![0.0; n * n];
    out.par_chunks_mut(n.max(1)).enumerate().for_each(|(j, row)| {
        for (k, v) in row.iter_mut().enumerate() {
            if k != j {
                // Same argument order for (j,k) and (k,j) so the matrix is exactly symmetric.
                let (a, b) = if j < k { (j, k) } else { (k, j) };
                *v = dist(centers[a], centers[b]);
            }
        }
    });
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    fn random_points(n: usize, seed: u64) -> Vec<DVec3> {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        (0..n).map(|_| DVec3::new(rng.random(), rng.random(), rng.random())).collect()
    }

    #[test]
    fn minima_match_brute_force() {
        let samples = random_points(200, 1);
        let centers = random_points(7, 2);
        let radii: Vec<f64> = (0..7).map(|j| 0.05 * j as f64).collect();
        let mins = row_minima(&samples, &centers, &radii);
        for (i, s) in samples.iter().enumerate() {
            let mut best = (f64::INFINITY, 0, 0.0);
            for (j, c) in centers.iter().enumerate() {
                let d = (*s - *c).length();
                if d - radii[j] < best.0 {
                    best = (d - radii[j], j, d);
                }
            }
            assert_eq!(mins[i].j as usize, best.1);
            assert_eq!(mins[i].gap, best.0);
            assert_eq!(mins[i].d, best.2);
        }
    }

    #[test]
    fn ties_take_first_index() {
        let centers = vec![DVec3::X, DVec3::X, DVec3::Y];
        let m = row_minima(&[DVec3::ZERO], &centers, &[0.0, 0.0, 0.0]);
        assert_eq!(m[0].j, 0);
        let near = nearest_sample_per_center(&[DVec3::ZERO], &[DVec3::X, DVec3::Y, DVec3::X]);
        assert_eq!(near, vec![0]);
    }

    #[test]
    fn pairwise_is_symmetric() {
        let c = random_points(9, 3);
        let p = pairwise(&c);
        for j in 0..9 {
            assert_eq!(p[j * 9 + j], 0.0);
            for k in 0..9 {
                assert_eq!(p[j * 9 + k], p[k * 9 + j]);
            }
        }
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn identical_across_thread_counts() {
        let samples = random_points(3000, 4);
        let centers = random_points(40, 5);
        let radii = vec![0.1; 40];
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let four = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let a = one.install(|| row_minima(&samples, &centers, &radii));
        let b = four.install(|| row_minima(&samples, &centers, &radii));
        assert_eq!(a, b);
        let a = one.install(|| nearest_sample_per_center(&centers, &samples));
        let b = four.install(|| nearest_sample_per_center(&centers, &samples));
        assert_eq!(a, b);
    }
}
