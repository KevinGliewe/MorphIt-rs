//! Adam and gradient-norm clipping with PyTorch's exact update rules.

use glam::DVec3;

use crate::config::TrainingConfig;
use crate::loss::RawGrads;
use crate::state::Spheres;

pub const BETA1: f64 = 0.9;
pub const BETA2: f64 = 0.999;
pub const EPS: f64 = 1e-8;

/// One Adam parameter group (`torch.optim.Adam`, no weight decay, no amsgrad).
#[derive(Clone, Debug, PartialEq)]
pub struct AdamGroup {
    pub lr: f64,
    step: u64,
    m: Vec<f64>,
    v: Vec<f64>,
}

impl AdamGroup {
    pub fn new(lr: f64, len: usize) -> Self {
        AdamGroup { lr, step: 0, m: vec![0.0; len], v: vec![0.0; len] }
    }

    /// Number of steps taken since creation or the last reset.
    pub fn steps(&self) -> u64 {
        self.step
    }

    /// Forget all moment estimates (a fresh optimizer, as after density control).
    pub fn reset(&mut self, len: usize) {
        self.step = 0;
        self.m = vec![0.0; len];
        self.v = vec![0.0; len];
    }

    /// One Adam update, following `torch.optim.adam._single_tensor_adam`.
    pub fn step(&mut self, params: &mut [f64], grads: &[f64]) {
        assert_eq!(params.len(), grads.len());
        assert_eq!(params.len(), self.m.len(), "Adam state size mismatch; reset after resizing");
        self.step += 1;
        let t = self.step as f64;
        let bias1 = 1.0 - BETA1.powf(t);
        let bias2 = 1.0 - BETA2.powf(t);
        let step_size = self.lr / bias1;
        let bias2_sqrt = bias2.sqrt();
        let weight = 1.0 - BETA1;
        for i in 0..params.len() {
            let g = grads[i];
            // exp_avg.lerp_(grad, 1 - beta1)
            self.m[i] += weight * (g - self.m[i]);
            // exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
            self.v[i] = self.v[i] * BETA2 + (1.0 - BETA2) * g * g;
            let denom = self.v[i].sqrt() / bias2_sqrt + EPS;
            params[i] += -step_size * (self.m[i] / denom);
        }
    }

    /// Adam update for a slice of vectors (flattened as x, y, z per entry).
    pub fn step_vec3(&mut self, params: &mut [DVec3], grads: &[DVec3]) {
        let mut p = flatten(params);
        let g = flatten(grads);
        self.step(&mut p, &g);
        for (dst, src) in params.iter_mut().zip(p.chunks_exact(3)) {
            *dst = DVec3::new(src[0], src[1], src[2]);
        }
    }
}

fn flatten(v: &[DVec3]) -> Vec<f64> {
    v.iter().flat_map(|c| c.to_array()).collect()
}

/// `torch.nn.utils.clip_grad_norm_` for a single parameter: scales the gradient
/// by `min(1, max_norm / (norm + 1e-6))`. Returns the norm before clipping.
pub fn clip_grad_norm(grads: &mut [f64], max_norm: f64) -> f64 {
    let norm = grads.iter().map(|g| g * g).sum::<f64>().sqrt();
    let coef = (max_norm / (norm + 1e-6)).min(1.0);
    for g in grads {
        *g *= coef;
    }
    norm
}

/// [`clip_grad_norm`] over all components of a vector gradient.
pub fn clip_grad_norm_vec3(grads: &mut [DVec3], max_norm: f64) -> f64 {
    let norm = grads.iter().map(|g| g.length_squared()).sum::<f64>().sqrt();
    let coef = (max_norm / (norm + 1e-6)).min(1.0);
    for g in grads {
        *g *= coef;
    }
    norm
}

/// The trainer's optimizer: one group each for centers, raw radii and
/// (with per-sphere mass) raw masses.
#[derive(Clone, Debug, PartialEq)]
pub struct Optimizer {
    pub centers: AdamGroup,
    pub radii: AdamGroup,
    pub masses: Option<AdamGroup>,
}

impl Optimizer {
    pub fn new(t: &TrainingConfig, spheres: &Spheres) -> Self {
        let n = spheres.len();
        Optimizer {
            centers: AdamGroup::new(t.center_lr, 3 * n),
            radii: AdamGroup::new(t.radius_lr, n),
            masses: spheres.per_sphere_mass().then(|| AdamGroup::new(t.mass_lr, n)),
        }
    }

    /// Fresh state sized for the current spheres (`_reset_optimizer`).
    pub fn reset(&mut self, spheres: &Spheres) {
        let n = spheres.len();
        self.centers.reset(3 * n);
        self.radii.reset(n);
        match (&mut self.masses, spheres.per_sphere_mass()) {
            (Some(g), true) => g.reset(n),
            (None, false) => {}
            _ => unreachable!("per-sphere mass mode cannot change during a run"),
        }
    }

    pub fn step(&mut self, spheres: &mut Spheres, grads: &RawGrads) {
        self.centers.step_vec3(&mut spheres.centers, &grads.centers);
        self.radii.step(&mut spheres.raw_radii, &grads.raw_radii);
        if let (Some(group), Some(params), Some(g)) =
            (&mut self.masses, &mut spheres.raw_masses, &grads.raw_masses)
        {
            group.step(params, g);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_step_moves_by_lr_times_sign() {
        let mut g = AdamGroup::new(0.01, 3);
        let mut p = vec![1.0, 2.0, 3.0];
        g.step(&mut p, &[0.5, -3.0, 1e-3]);
        assert!((p[0] - (1.0 - 0.01)).abs() < 1e-8);
        assert!((p[1] - (2.0 + 0.01)).abs() < 1e-8);
        // Tiny gradients are damped by eps.
        assert!((p[2] - (3.0 - 0.01 * 1e-3 / (1e-3 + 1e-8))).abs() < 1e-12);
    }

    #[test]
    fn three_steps_match_hand_computation() {
        let lr = 0.1;
        let grads = [0.3, -0.2, 0.7];
        let mut g = AdamGroup::new(lr, 1);
        let mut p = [0.0];
        let (mut m, mut v, mut x) = (0.0f64, 0.0f64, 0.0f64);
        for (t, &gr) in grads.iter().enumerate() {
            g.step(&mut p, &[gr]);
            let t = (t + 1) as f64;
            m = 0.9 * m + 0.1 * gr;
            v = 0.999 * v + 0.001 * gr * gr;
            let mh = m / (1.0 - 0.9f64.powf(t));
            let vh = v / (1.0 - 0.999f64.powf(t));
            x -= lr * mh / (vh.sqrt() + 1e-8);
            assert!((p[0] - x).abs() < 1e-12, "step {t}");
        }
        assert_eq!(g.steps(), 3);
        g.reset(1);
        assert_eq!(g.steps(), 0);
    }

    #[test]
    fn clipping() {
        let mut g = vec![3.0, 4.0];
        let n = clip_grad_norm(&mut g, 1.0);
        assert_eq!(n, 5.0);
        assert!((g[0] - 3.0 / (5.0 + 1e-6)).abs() < 1e-15);
        let mut small = vec![0.1, 0.2];
        clip_grad_norm(&mut small, 1.0);
        assert_eq!(small, vec![0.1, 0.2]);
        let mut zero = vec![0.0; 4];
        clip_grad_norm(&mut zero, 1.0);
        assert!(zero.iter().all(|&x| x == 0.0));
        let mut v = vec![DVec3::new(3.0, 0.0, 0.0), DVec3::new(0.0, 4.0, 0.0)];
        assert_eq!(clip_grad_norm_vec3(&mut v, 0.5), 5.0);
        assert!((v[1].y - 4.0 * 0.5 / (5.0 + 1e-6)).abs() < 1e-15);
    }
}
