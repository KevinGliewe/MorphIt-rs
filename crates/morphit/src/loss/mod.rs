//! The eleven MorphIt loss terms with hand-derived gradients (`losses.py`).
//!
//! Each term returns its unweighted value and accumulates `weight * dL/dx`
//! with respect to the *real* quantities (centers, radii, masses) into
//! [`Grads`]. [`Grads::into_raw`] then applies the chain rule once into the raw
//! softplus parameters, which is where PyTorch's autograd ends up as well.
//!
//! Conventions that mirror PyTorch autograd:
//! - `d|x - y| / dx = (x - y) / |x - y|`, and zero when the distance is zero.
//! - `relu'(0) = 0`, `abs'(0) = 0`.
//! - `min`, `argmin` and `topk` route gradient only to the selected index
//!   (first index on ties).

mod boundary;
mod containment;
mod coverage;
mod hausdorff;
mod mesh_containment;
mod overlap;
mod physics;
mod sqem;
mod surface;

#[cfg(test)]
pub(crate) mod fd;

use glam::{DMat3, DVec3};

use crate::config::{LossId, LossWeights};
use crate::device::Device;
use crate::distances::RowMin;
use crate::error::Result;
use crate::math::softplus_grad;
use crate::mesh::Mesh;
use crate::search::{BoundaryPairs, SampleSet, Search, Wants};
use crate::state::{FOUR_THIRDS_PI, Spheres};

/// Fixed sample points the losses are evaluated on.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Samples {
    /// Points inside the mesh (coverage loss, density control, reseeding).
    pub inside: Vec<DVec3>,
    /// Points on the mesh surface.
    pub surface: Vec<DVec3>,
    /// Outward face normal at each surface sample.
    pub surface_normals: Vec<DVec3>,
}

/// Ground-truth physics of the mesh and the normalization constants of the
/// mass, COM and inertia losses (`MorphItLosses.__init__`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PhysicsTargets {
    pub mass: f64,
    pub com: DVec3,
    /// Inertia about the center of mass, scaled by density.
    pub inertia: DMat3,
    pub mass_norm_sq: f64,
    pub com_norm_sq: f64,
    pub inertia_norm_sq: f64,
}

impl PhysicsTargets {
    const EPS: f64 = 1e-20;

    pub fn from_mesh(mesh: &Mesh, density: f64) -> Self {
        let mass = mesh.volume() * density;
        let inertia = mesh.moment_inertia() * density;
        let inertia_sq = frobenius_sq(&inertia);
        PhysicsTargets {
            mass,
            com: mesh.center_mass(),
            inertia,
            mass_norm_sq: mass * mass + Self::EPS,
            com_norm_sq: mesh.scale() * mesh.scale() + Self::EPS,
            inertia_norm_sq: inertia_sq + Self::EPS,
        }
    }
}

pub(crate) fn frobenius_sq(m: &DMat3) -> f64 {
    m.x_axis.length_squared() + m.y_axis.length_squared() + m.z_axis.length_squared()
}

/// Everything constant during an optimization: samples, physics targets,
/// density, and the search engine bound to the samples.
#[derive(Debug)]
pub struct Problem {
    pub samples: Samples,
    pub targets: PhysicsTargets,
    pub density: f64,
    pub search: Search,
}

impl Problem {
    /// Bind the samples to a search engine on `device`; `num_spheres` lets
    /// `Device::Auto` judge the problem size.
    pub fn new(
        samples: Samples,
        targets: PhysicsTargets,
        density: f64,
        device: Device,
        num_spheres: usize,
    ) -> Result<Problem> {
        let search = Search::new(&samples, device, num_spheres)?;
        Ok(Problem { samples, targets, density, search })
    }

    /// CPU-only problem.
    pub fn cpu(samples: Samples, targets: PhysicsTargets, density: f64) -> Problem {
        Problem { samples, targets, density, search: Search::cpu() }
    }

    /// Nearest sphere per sample of `set`.
    pub fn row_minima(&self, set: SampleSet, centers: &[DVec3], radii: &[f64]) -> Vec<RowMin> {
        let samples = match set {
            SampleSet::Inside => &self.samples.inside,
            SampleSet::Surface => &self.samples.surface,
        };
        self.search.row_minima(set, samples, centers, radii)
    }

    /// Surface samples each sphere may engulf.
    pub fn boundary_pairs(&self, centers: &[DVec3], radii: &[f64]) -> BoundaryPairs {
        self.search.boundary_pairs(&self.samples.surface, centers, radii)
    }

    /// Nearest surface sample per center.
    pub fn nearest_sample_per_center(&self, centers: &[DVec3]) -> Vec<u32> {
        self.search.nearest_sample_per_center(&self.samples.surface, centers)
    }

    /// [`Problem::nearest_sample_per_center`], awaiting the GPU readback.
    pub async fn nearest_sample_per_center_async(&self, centers: &[DVec3]) -> Vec<u32> {
        self.search.nearest_sample_per_center_async(&self.samples.surface, centers).await
    }
}

/// Gradients with respect to real centers, radii and masses.
#[derive(Clone, Debug, PartialEq)]
pub struct Grads {
    pub centers: Vec<DVec3>,
    pub radii: Vec<f64>,
    pub masses: Vec<f64>,
}

impl Grads {
    pub fn zeros(n: usize) -> Self {
        Grads { centers: vec![DVec3::ZERO; n], radii: vec![0.0; n], masses: vec![0.0; n] }
    }

    /// Chain rule into raw parameter space.
    ///
    /// Learned masses: `d_mu = d_m * softplus'(mu)`. Derived masses
    /// (`m = density * 4/3 pi r^3`): `d_r += d_m * density * 4 pi r^2`.
    /// Then `d_rho = d_r * softplus'(rho)`.
    pub fn into_raw(self, spheres: &Spheres, radii: &[f64], density: f64) -> RawGrads {
        let Grads { centers, radii: mut d_r, masses: d_m } = self;
        let raw_masses = match &spheres.raw_masses {
            Some(mu) => Some(mu.iter().zip(&d_m).map(|(&m, &g)| g * softplus_grad(m)).collect()),
            None => {
                for ((dr, &dm), &r) in d_r.iter_mut().zip(&d_m).zip(radii) {
                    *dr += dm * (density * FOUR_THIRDS_PI * 3.0 * (r * r));
                }
                None
            }
        };
        let raw_radii = spheres.raw_radii.iter().zip(&d_r).map(|(&rho, &g)| g * softplus_grad(rho)).collect();
        RawGrads { centers, raw_radii, raw_masses }
    }
}

/// Gradients with respect to the optimized (raw) parameters.
#[derive(Clone, Debug, PartialEq)]
pub struct RawGrads {
    pub centers: Vec<DVec3>,
    pub raw_radii: Vec<f64>,
    pub raw_masses: Option<Vec<f64>>,
}

/// Values and gradients of one loss evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct Evaluation {
    /// Unweighted value per [`LossId`] (zero for skipped losses).
    pub raw: [f64; LossId::COUNT],
    /// `weight * value` per [`LossId`].
    pub weighted: [f64; LossId::COUNT],
    /// Sum of `weighted` in [`LossId`] order.
    pub total: f64,
    pub grads: RawGrads,
}

/// Read-only inputs shared by all loss terms.
pub(crate) struct LossInput<'a> {
    pub problem: &'a Problem,
    pub centers: &'a [DVec3],
    pub radii: &'a [f64],
    pub masses: &'a [f64],
    pub inside_min: &'a [RowMin],
    pub boundary: &'a BoundaryPairs,
    pub surface_min: &'a [RowMin],
    pub nearest_surface: &'a [u32],
    pub pair: &'a [f64],
}

/// Evaluate all active (non-zero weight) losses and their gradients.
///
/// Zero-weight losses are skipped entirely, as in `compute_all_losses(weights=...)`.
/// The flatness loss is not implemented; its weight must be zero.
pub fn evaluate(problem: &Problem, spheres: &Spheres, weights: &LossWeights) -> Evaluation {
    crate::exec::block_on(evaluate_async(problem, spheres, weights))
}

/// [`evaluate`], awaiting the GPU readbacks (needed for WebGPU in the browser).
pub async fn evaluate_async(problem: &Problem, spheres: &Spheres, weights: &LossWeights) -> Evaluation {
    let n = spheres.len();
    let radii = spheres.radii();
    let masses = spheres.masses(problem.density);
    let centers = &spheres.centers;

    let need_inside = weights.active(LossId::Coverage);
    let need_pairs = weights.active(LossId::Boundary);
    let need_surface_min =
        weights.active(LossId::Surface) || weights.active(LossId::Sqem) || weights.active(LossId::Hausdorff);
    let need_nearest = weights.active(LossId::MeshContainment);
    let need_pair = weights.active(LossId::Overlap) || weights.active(LossId::Containment);

    let want = Wants { inside_min: need_inside, surface_min: need_surface_min, pairs: need_pairs };
    let q = problem.search.queries_async(&problem.samples, centers, &radii, want).await;
    let (inside_min, surface_min, boundary) = (q.inside_min, q.surface_min, q.pairs);
    let nearest_surface =
        if need_nearest { problem.nearest_sample_per_center_async(centers).await } else { Vec::new() };
    let pair = if need_pair { crate::distances::pairwise(centers) } else { Vec::new() };

    let inp = LossInput {
        problem,
        centers,
        radii: &radii,
        masses: &masses,
        inside_min: &inside_min,
        boundary: &boundary,
        surface_min: &surface_min,
        nearest_surface: &nearest_surface,
        pair: &pair,
    };

    let mut g = Grads::zeros(n);
    let mut raw = [0.0; LossId::COUNT];
    for id in LossId::ALL {
        let w = weights[id];
        if w == 0.0 {
            continue;
        }
        raw[id.index()] = match id {
            LossId::Coverage => coverage::coverage(&inp, w, &mut g),
            LossId::Overlap => overlap::overlap(&inp, w, &mut g),
            LossId::Boundary => boundary::boundary(&inp, w, &mut g),
            LossId::Surface => surface::surface(&inp, w, &mut g),
            LossId::Containment => containment::containment(&inp, w, &mut g),
            LossId::Sqem => sqem::sqem(&inp, w, &mut g),
            LossId::Hausdorff => hausdorff::hausdorff(&inp, w, &mut g),
            LossId::MeshContainment => mesh_containment::mesh_containment(&inp, w, &mut g),
            LossId::Mass => physics::mass(&inp, w, &mut g),
            LossId::Com => physics::com(&inp, w, &mut g),
            LossId::Inertia => physics::inertia(&inp, w, &mut g),
            LossId::Flatness => {
                debug_assert!(false, "flatness loss is not implemented; Config::validate rejects it");
                0.0
            }
        };
    }
    let mut weighted = [0.0; LossId::COUNT];
    let mut total = 0.0;
    for id in LossId::ALL {
        let v = weights[id] * raw[id.index()];
        weighted[id.index()] = v;
        total += v;
    }
    let grads = g.into_raw(spheres, &radii, problem.density);
    Evaluation { raw, weighted, total, grads }
}

#[cfg(test)]
mod tests {
    use super::fd::*;
    use super::*;
    use crate::config::Preset;

    #[test]
    fn into_raw_chains_through_softplus_and_derived_mass() {
        let spheres = Spheres::from_real(vec![DVec3::ZERO, DVec3::ONE], &[0.2, 0.5], None);
        let radii = spheres.radii();
        let g = Grads { centers: vec![DVec3::X, DVec3::Y], radii: vec![1.0, -2.0], masses: vec![0.5, 0.25] };
        let raw = g.into_raw(&spheres, &radii, 10.0);
        for j in 0..2 {
            let dm_dr = 10.0 * 4.0 * std::f64::consts::PI * radii[j] * radii[j];
            let d_r = [1.0, -2.0][j] + [0.5, 0.25][j] * dm_dr;
            let expect = d_r * softplus_grad(spheres.raw_radii[j]);
            assert!((raw.raw_radii[j] - expect).abs() < 1e-12);
        }
        assert!(raw.raw_masses.is_none());
        assert_eq!(raw.centers, vec![DVec3::X, DVec3::Y]);
    }

    #[test]
    fn zero_weight_losses_are_skipped() {
        let p = toy_problem(1, 60, 60, 1000.0);
        let sp = toy_spheres(2, 5, false);
        let mut w = LossWeights::default();
        w[LossId::Coverage] = 2.0;
        let e = evaluate(&p, &sp, &w);
        for id in LossId::ALL {
            if id != LossId::Coverage {
                assert_eq!(e.raw[id.index()], 0.0, "{id:?}");
            }
        }
        assert!(e.raw[0] > 0.0);
        assert_eq!(e.total, 2.0 * e.raw[0]);
    }

    #[test]
    fn every_preset_passes_fd_check() {
        let p = toy_problem(3, 60, 60, 1000.0);
        for preset in Preset::ALL {
            for psm in [false, true] {
                let sp = toy_spheres(4, 5, psm);
                assert_fd(&p, &sp, &preset.weights());
            }
        }
    }

    #[test]
    fn all_losses_at_once_pass_fd_check() {
        let p = toy_problem(5, 60, 60, 1.0);
        let mut w = LossWeights([1.0; LossId::COUNT]);
        w[LossId::Flatness] = 0.0;
        for psm in [false, true] {
            assert_fd(&p, &toy_spheres(6, 6, psm), &w);
        }
    }
}
