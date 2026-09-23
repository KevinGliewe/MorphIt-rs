//! The three distance searches the optimizer spends its time on, behind one
//! interface with a CPU backend and (feature `gpu`) a wgpu backend.
//!
//! 1. nearest sphere per sample: `min_j (|p_i - c_j| - r_j)` ([`Search::row_minima`]),
//! 2. the surface samples each sphere engulfs, for the boundary loss
//!    ([`Search::boundary_pairs`]),
//! 3. nearest surface sample per sphere ([`Search::nearest_sample_per_center`]).
//!
//! **Every backend returns bit-identical results.** The GPU only proposes, in
//! f32, which index wins each row and by how much; the CPU recomputes the
//! winner with the same f64 [`dist`] the CPU backend uses and rescans any row
//! whose f32 margin is below a proven error bound. The loss code never sees
//! which backend ran.

#[cfg(feature = "gpu")]
pub(crate) mod gpu;
#[cfg(feature = "gpu")]
pub use gpu::{AUTO_GPU_MIN_WORK, init_gpu, list_devices};

/// GPU adapters (none in a build without the `gpu` feature).
#[cfg(not(feature = "gpu"))]
pub fn list_devices() -> Vec<crate::device::GpuInfo> {
    Vec::new()
}

/// Discover the GPUs (none in a build without the `gpu` feature).
#[cfg(not(feature = "gpu"))]
pub async fn init_gpu() -> Vec<crate::device::GpuInfo> {
    Vec::new()
}

/// What [`Search::queries`] should compute.
#[derive(Clone, Copy, Debug, Default)]
pub struct Wants {
    pub inside_min: bool,
    pub surface_min: bool,
    pub pairs: bool,
}

/// Results of [`Search::queries`]; unrequested parts are empty.
#[derive(Debug, Default)]
pub struct QueryResults {
    pub inside_min: Vec<RowMin>,
    pub surface_min: Vec<RowMin>,
    pub pairs: BoundaryPairs,
}

use crate::par::*;
use glam::DVec3;

use crate::device::{Device, ResolvedDevice};
use crate::distances::{self, RowMin, dist};
#[cfg(not(feature = "gpu"))]
use crate::error::Error;
use crate::error::Result;
use crate::loss::Samples;

/// Which fixed sample set a query runs over.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum SampleSet {
    Inside,
    Surface,
}

/// Surface samples each sphere may engulf, as CSR by sphere: for sphere `j`,
/// `samples[offsets[j]..offsets[j + 1]]` lists candidate sample indices in
/// ascending order. The list may contain samples the sphere does not actually
/// engulf (the GPU adds a tolerance); consumers apply the exact test.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct BoundaryPairs {
    pub offsets: Vec<usize>,
    pub samples: Vec<u32>,
}

impl BoundaryPairs {
    /// Candidate samples of sphere `j`.
    pub fn of(&self, j: usize) -> &[u32] {
        &self.samples[self.offsets[j]..self.offsets[j + 1]]
    }

    /// Total number of candidate pairs.
    pub fn len(&self) -> usize {
        self.samples.len()
    }

    pub fn is_empty(&self) -> bool {
        self.samples.is_empty()
    }

    /// Build from per-sphere lists.
    pub(crate) fn from_lists(lists: Vec<Vec<u32>>) -> Self {
        let mut offsets = Vec::with_capacity(lists.len() + 1);
        let mut samples = Vec::with_capacity(lists.iter().map(Vec::len).sum());
        offsets.push(0);
        for l in lists {
            samples.extend_from_slice(&l);
            offsets.push(samples.len());
        }
        BoundaryPairs { offsets, samples }
    }
}

enum Backend {
    Cpu,
    #[cfg(feature = "gpu")]
    Gpu(Box<gpu::SessionSearch>),
}

/// Distance-search engine bound to one problem's sample sets.
pub struct Search {
    backend: Backend,
    resolved: ResolvedDevice,
}

impl std::fmt::Debug for Search {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Search").field("device", &self.resolved).finish()
    }
}

impl Search {
    /// CPU backend.
    pub fn cpu() -> Search {
        Search { backend: Backend::Cpu, resolved: ResolvedDevice::Cpu }
    }

    /// Backend for `device`. `num_spheres` is the planned sphere count, used by
    /// [`Device::Auto`] to decide whether the problem is large enough for a GPU.
    pub fn new(samples: &Samples, device: Device, num_spheres: usize) -> Result<Search> {
        match device {
            Device::Cpu => Ok(Search::cpu()),
            #[cfg(feature = "gpu")]
            Device::Auto | Device::Gpu(_) => gpu::open(samples, device, num_spheres),
            #[cfg(not(feature = "gpu"))]
            Device::Auto => Ok(Search::cpu()),
            #[cfg(not(feature = "gpu"))]
            Device::Gpu(_) => {
                let _ = (samples, num_spheres);
                Err(Error::config("model.device", "this build has no GPU support (feature `gpu` is off)"))
            }
        }
    }

    #[cfg(feature = "gpu")]
    pub(crate) fn with_gpu(session: gpu::SessionSearch, resolved: ResolvedDevice) -> Search {
        Search { backend: Backend::Gpu(Box::new(session)), resolved }
    }

    /// The device answering the queries.
    pub fn device(&self) -> &ResolvedDevice {
        &self.resolved
    }

    /// Why a GPU search fell back to the CPU (a device or shader error), if
    /// it did. Results are unaffected either way.
    pub fn gpu_error(&self) -> Option<String> {
        match &self.backend {
            Backend::Cpu => None,
            #[cfg(feature = "gpu")]
            Backend::Gpu(g) => g.last_error(),
        }
    }

    /// Whether queries must be awaited: a WebGPU backend in the browser,
    /// whose readbacks resolve only after yielding to the event loop. The
    /// sync methods panic in that case; use the `_async` variants.
    pub fn needs_async(&self) -> bool {
        match &self.backend {
            Backend::Cpu => false,
            #[cfg(feature = "gpu")]
            Backend::Gpu(_) => cfg!(target_arch = "wasm32"),
        }
    }

    /// The sample-side queries of one loss evaluation in a single pass
    /// (one GPU round trip).
    pub fn queries(&self, samples: &Samples, centers: &[DVec3], radii: &[f64], want: Wants) -> QueryResults {
        crate::exec::block_on(self.queries_async(samples, centers, radii, want))
    }

    /// [`Search::queries`], awaiting the GPU readback.
    pub async fn queries_async(
        &self,
        samples: &Samples,
        centers: &[DVec3],
        radii: &[f64],
        want: Wants,
    ) -> QueryResults {
        match &self.backend {
            Backend::Cpu => QueryResults {
                inside_min: if want.inside_min {
                    distances::row_minima(&samples.inside, centers, radii)
                } else {
                    vec![]
                },
                surface_min: if want.surface_min {
                    distances::row_minima(&samples.surface, centers, radii)
                } else {
                    vec![]
                },
                pairs: if want.pairs {
                    cpu_boundary_pairs(&samples.surface, centers, radii)
                } else {
                    BoundaryPairs::default()
                },
            },
            #[cfg(feature = "gpu")]
            Backend::Gpu(g) => g.queries_async(samples, centers, radii, want).await,
        }
    }

    /// Per sample of `set`: the sphere minimizing `|p - c_j| - r_j` (first index
    /// on ties), the distance to it and the minimum. `samples` must be the
    /// slice this search was built with for `set`.
    pub fn row_minima(
        &self,
        set: SampleSet,
        samples: &[DVec3],
        centers: &[DVec3],
        radii: &[f64],
    ) -> Vec<RowMin> {
        match &self.backend {
            Backend::Cpu => {
                let _ = set;
                distances::row_minima(samples, centers, radii)
            }
            #[cfg(feature = "gpu")]
            Backend::Gpu(g) => g.row_minima(set, samples, centers, radii),
        }
    }

    /// Candidate `(sphere, surface sample)` pairs with `r_j - |s_i - c_j| > 0`,
    /// possibly plus a tolerance band (see [`BoundaryPairs`]).
    pub fn boundary_pairs(&self, surface: &[DVec3], centers: &[DVec3], radii: &[f64]) -> BoundaryPairs {
        match &self.backend {
            Backend::Cpu => cpu_boundary_pairs(surface, centers, radii),
            #[cfg(feature = "gpu")]
            Backend::Gpu(g) => g.boundary_pairs(surface, centers, radii),
        }
    }

    /// Per center: index of the nearest surface sample (first index on ties).
    pub fn nearest_sample_per_center(&self, surface: &[DVec3], centers: &[DVec3]) -> Vec<u32> {
        crate::exec::block_on(self.nearest_sample_per_center_async(surface, centers))
    }

    /// [`Search::nearest_sample_per_center`], awaiting the GPU readback.
    pub async fn nearest_sample_per_center_async(&self, surface: &[DVec3], centers: &[DVec3]) -> Vec<u32> {
        match &self.backend {
            Backend::Cpu => distances::nearest_sample_per_center(centers, surface),
            #[cfg(feature = "gpu")]
            Backend::Gpu(g) => g.nearest_sample_per_center_async(surface, centers).await,
        }
    }
}

/// Exact CPU candidate lists: every `i` with `r_j - dist(s_i, c_j) > 0`.
pub(crate) fn cpu_boundary_pairs(surface: &[DVec3], centers: &[DVec3], radii: &[f64]) -> BoundaryPairs {
    let lists: Vec<Vec<u32>> = centers
        .par_iter()
        .zip(radii)
        .map(|(&c, &r)| {
            let mut l = Vec::new();
            for (i, &s) in surface.iter().enumerate() {
                if -(dist(s, c) - r) > 0.0 {
                    l.push(i as u32);
                }
            }
            l
        })
        .collect();
    BoundaryPairs::from_lists(lists)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use rand::{Rng, SeedableRng};

    pub(crate) fn random_points(n: usize, seed: u64, extent: f64) -> Vec<DVec3> {
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
        (0..n).map(|_| DVec3::new(rng.random(), rng.random(), rng.random()) * extent).collect()
    }

    #[test]
    fn cpu_pairs_match_brute_force() {
        let surface = random_points(500, 1, 1.0);
        let centers = random_points(9, 2, 1.0);
        let radii: Vec<f64> = (0..9).map(|j| 0.05 + 0.04 * j as f64).collect();
        let p = cpu_boundary_pairs(&surface, &centers, &radii);
        assert_eq!(p.offsets.len(), 10);
        let mut total = 0;
        for (j, (&c, &r)) in centers.iter().zip(&radii).enumerate() {
            let want: Vec<u32> = (0..500u32).filter(|&i| (surface[i as usize] - c).length() < r).collect();
            assert_eq!(p.of(j), &want[..], "sphere {j}");
            total += want.len();
        }
        assert_eq!(p.len(), total);
        assert!(total > 0);
    }

    #[test]
    #[cfg(feature = "parallel")]
    fn cpu_pairs_identical_across_thread_counts() {
        let surface = random_points(3000, 4, 1.0);
        let centers = random_points(40, 5, 1.0);
        let radii = vec![0.1; 40];
        let one = rayon::ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let four = rayon::ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let a = one.install(|| cpu_boundary_pairs(&surface, &centers, &radii));
        let b = four.install(|| cpu_boundary_pairs(&surface, &centers, &radii));
        assert_eq!(a, b);
    }

    #[test]
    fn empty_lists_are_well_formed() {
        let p = BoundaryPairs::from_lists(vec![vec![], vec![]]);
        assert!(p.is_empty());
        assert_eq!(p.of(1), &[] as &[u32]);
    }

    #[test]
    fn cpu_device_reports_cpu() {
        let s = Search::new(&Samples::default(), Device::Cpu, 5).unwrap();
        assert_eq!(*s.device(), ResolvedDevice::Cpu);
        #[cfg(not(feature = "gpu"))]
        {
            assert!(Search::new(&Samples::default(), Device::Gpu(None), 5).is_err());
            assert_eq!(
                *Search::new(&Samples::default(), Device::Auto, 5).unwrap().device(),
                ResolvedDevice::Cpu
            );
        }
    }
}
