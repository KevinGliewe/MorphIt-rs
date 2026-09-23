//! Training history (`convergence_tracker.py` without plotting) and the
//! convergence test used for early stopping and density-control triggers.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value, json};

use crate::config::LossId;

/// One density-control pass.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DensityEvent {
    pub iteration: usize,
    pub spheres_added: usize,
    pub spheres_removed: usize,
}

/// Per-iteration metrics.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct History {
    pub iterations: Vec<usize>,
    pub total_loss: Vec<f64>,
    /// Weighted loss per [`LossId`], per iteration.
    pub component_losses: Vec<[f64; LossId::COUNT]>,
    pub position_grad_mag: Vec<f64>,
    pub radius_grad_mag: Vec<f64>,
    pub num_spheres: Vec<usize>,
    pub min_radius: Vec<f64>,
    pub max_radius: Vec<f64>,
    pub mean_radius: Vec<f64>,
    pub time_per_iteration: Vec<f64>,
    pub density_control_events: Vec<DensityEvent>,
}

/// Result of [`History::analyze_convergence`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ConvergenceAnalysis {
    pub converged: bool,
    pub loss_change: f64,
    pub grad_small: bool,
}

/// Values recorded for one iteration.
#[derive(Clone, Copy, Debug)]
pub struct IterationRecord<'a> {
    pub iteration: usize,
    pub total_loss: f64,
    pub weighted: &'a [f64; LossId::COUNT],
    pub position_grad_mag: f64,
    pub radius_grad_mag: f64,
    pub radii: &'a [f64],
    pub seconds: f64,
}

impl History {
    pub fn len(&self) -> usize {
        self.total_loss.len()
    }

    pub fn is_empty(&self) -> bool {
        self.total_loss.is_empty()
    }

    pub fn record(&mut self, r: IterationRecord<'_>) {
        self.iterations.push(r.iteration);
        self.total_loss.push(r.total_loss);
        self.component_losses.push(*r.weighted);
        self.position_grad_mag.push(r.position_grad_mag);
        self.radius_grad_mag.push(r.radius_grad_mag);
        let n = r.radii.len();
        self.num_spheres.push(n);
        let (mut lo, mut hi, mut sum) = (f64::INFINITY, f64::NEG_INFINITY, 0.0);
        for &x in r.radii {
            lo = lo.min(x);
            hi = hi.max(x);
            sum += x;
        }
        self.min_radius.push(lo);
        self.max_radius.push(hi);
        self.mean_radius.push(if n > 0 { sum / n as f64 } else { 0.0 });
        self.time_per_iteration.push(r.seconds);
    }

    /// `ConvergenceTracker.analyze_convergence`: over the last `window` iterations,
    /// converged when the relative loss change is below `threshold` or more than
    /// half of both gradient magnitudes are below 1e-4.
    pub fn analyze_convergence(&self, window: usize, threshold: f64) -> ConvergenceAnalysis {
        let n = self.len();
        if window == 0 || n < window {
            return ConvergenceAnalysis { converged: false, loss_change: f64::NAN, grad_small: false };
        }
        let recent = &self.total_loss[n - window..];
        let loss_change = (recent[0] - recent[window - 1]).abs() / recent[0].abs().max(1e-5);
        let pos = &self.position_grad_mag[n - window..];
        let rad = &self.radius_grad_mag[n - window..];
        let grad_small = pos.iter().filter(|&&g| g < 1e-4).count() > window / 2
            && rad.iter().filter(|&&g| g < 1e-4).count() > window / 2;
        ConvergenceAnalysis { converged: loss_change < threshold || grad_small, loss_change, grad_small }
    }

    /// JSON in the layout of `ConvergenceTracker.metrics`.
    pub fn to_json(&self) -> Value {
        let mut comps = Map::new();
        for id in LossId::ALL {
            let series: Vec<f64> = self.component_losses.iter().map(|c| c[id.index()]).collect();
            comps.insert(id.name().to_string(), json!(series));
        }
        json!({
            "total_loss": self.total_loss,
            "component_losses": comps,
            "gradient_info": {
                "position_grad_mag": self.position_grad_mag,
                "radius_grad_mag": self.radius_grad_mag,
            },
            "sphere_stats": {
                "num_spheres": self.num_spheres,
                "min_radius": self.min_radius,
                "max_radius": self.max_radius,
                "mean_radius": self.mean_radius,
            },
            "iterations": self.iterations,
            "time_per_iteration": self.time_per_iteration,
            "density_control_events": self.density_control_events,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push(h: &mut History, loss: f64, pos: f64, rad: f64) {
        let w = [0.0; LossId::COUNT];
        h.record(IterationRecord {
            iteration: h.len(),
            total_loss: loss,
            weighted: &w,
            position_grad_mag: pos,
            radius_grad_mag: rad,
            radii: &[0.1, 0.3],
            seconds: 0.0,
        });
    }

    #[test]
    fn needs_enough_data() {
        let mut h = History::default();
        push(&mut h, 1.0, 1.0, 1.0);
        assert!(!h.analyze_convergence(5, 0.01).converged);
    }

    #[test]
    fn plateau_and_small_gradients() {
        let mut h = History::default();
        for i in 0..10 {
            push(&mut h, 1.0 - 0.0001 * i as f64, 1.0, 1.0);
        }
        let a = h.analyze_convergence(5, 0.01);
        assert!(a.converged && !a.grad_small);
        let mut h = History::default();
        for i in 0..10 {
            push(&mut h, 10.0 - i as f64, 1e-5, 1e-5);
        }
        let a = h.analyze_convergence(5, 0.01);
        assert!(a.converged && a.grad_small);
        let mut h = History::default();
        for i in 0..10 {
            push(&mut h, 10.0 - i as f64, 1.0, 1e-5);
        }
        assert!(!h.analyze_convergence(5, 0.01).converged);
    }

    #[test]
    fn records_sphere_stats_and_serializes() {
        let mut h = History::default();
        push(&mut h, 2.0, 0.1, 0.2);
        h.density_control_events.push(DensityEvent { iteration: 0, spheres_added: 2, spheres_removed: 2 });
        assert_eq!(h.num_spheres, vec![2]);
        assert!((h.mean_radius[0] - 0.2).abs() < 1e-15);
        let j = h.to_json();
        assert_eq!(j["component_losses"]["coverage_loss"][0], json!(0.0));
        assert_eq!(j["density_control_events"][0]["spheres_added"], json!(2));
    }
}
