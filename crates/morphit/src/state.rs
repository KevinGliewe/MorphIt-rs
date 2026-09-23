//! Optimizable sphere parameters.
//!
//! Radii (and optional per-sphere masses) are stored in raw, pre-softplus space
//! exactly like the Python `nn.Parameter`s; always read them through
//! [`Spheres::radii`] and [`Spheres::masses`].

use glam::DVec3;

use crate::math::{inv_softplus, softplus};

/// `(4/3) * pi` as written in `morphit.py`.
pub const FOUR_THIRDS_PI: f64 = (4.0 / 3.0) * std::f64::consts::PI;

/// Sphere set: centers plus raw radii and optional raw per-sphere masses.
#[derive(Clone, Debug, PartialEq)]
pub struct Spheres {
    pub centers: Vec<DVec3>,
    /// Pre-softplus radii.
    pub raw_radii: Vec<f64>,
    /// Pre-softplus masses when per-sphere mass is learned, else `None`.
    pub raw_masses: Option<Vec<f64>>,
}

impl Spheres {
    /// Build from real (post-softplus) radii and optional real masses.
    pub fn from_real(centers: Vec<DVec3>, radii: &[f64], masses: Option<&[f64]>) -> Self {
        assert_eq!(centers.len(), radii.len());
        if let Some(m) = masses {
            assert_eq!(m.len(), radii.len());
        }
        Spheres {
            centers,
            raw_radii: radii.iter().map(|&r| inv_softplus(r)).collect(),
            raw_masses: masses.map(|m| m.iter().map(|&x| inv_softplus(x)).collect()),
        }
    }

    pub fn len(&self) -> usize {
        self.centers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.centers.is_empty()
    }

    pub fn per_sphere_mass(&self) -> bool {
        self.raw_masses.is_some()
    }

    /// Real radii, `softplus(raw_radii)`.
    pub fn radii(&self) -> Vec<f64> {
        self.raw_radii.iter().map(|&x| softplus(x)).collect()
    }

    /// Real masses: learned (`softplus(raw_masses)`) or `density * 4/3 pi r^3`.
    pub fn masses(&self, density: f64) -> Vec<f64> {
        match &self.raw_masses {
            Some(m) => m.iter().map(|&x| softplus(x)).collect(),
            None => self.radii().into_iter().map(|r| density * FOUR_THIRDS_PI * (r * r * r)).collect(),
        }
    }

    /// New set with the spheres at `idx`, in that order.
    pub fn select(&self, idx: &[usize]) -> Spheres {
        Spheres {
            centers: idx.iter().map(|&i| self.centers[i]).collect(),
            raw_radii: idx.iter().map(|&i| self.raw_radii[i]).collect(),
            raw_masses: self.raw_masses.as_ref().map(|m| idx.iter().map(|&i| m[i]).collect()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn real_round_trip() {
        let s = Spheres::from_real(vec![DVec3::ZERO, DVec3::ONE], &[0.1, 0.25], Some(&[2.0, 3.0]));
        let r = s.radii();
        assert!((r[0] - 0.1).abs() < 1e-14 && (r[1] - 0.25).abs() < 1e-14);
        let m = s.masses(1000.0);
        assert!((m[0] - 2.0).abs() < 1e-13 && (m[1] - 3.0).abs() < 1e-13);
    }

    #[test]
    fn derived_masses_use_density() {
        let s = Spheres::from_real(vec![DVec3::ZERO], &[0.5], None);
        let m = s.masses(1000.0);
        assert!((m[0] - 1000.0 * FOUR_THIRDS_PI * 0.125).abs() < 1e-9);
        assert!(!s.per_sphere_mass());
    }

    #[test]
    fn select_reorders() {
        let s =
            Spheres::from_real(vec![DVec3::X, DVec3::Y, DVec3::Z], &[0.1, 0.2, 0.3], Some(&[1.0, 2.0, 3.0]));
        let t = s.select(&[2, 0]);
        assert_eq!(t.centers, vec![DVec3::Z, DVec3::X]);
        assert_eq!(t.raw_radii, vec![s.raw_radii[2], s.raw_radii[0]]);
        assert_eq!(
            t.raw_masses.unwrap(),
            vec![s.raw_masses.as_ref().unwrap()[2], s.raw_masses.as_ref().unwrap()[0]]
        );
    }
}
