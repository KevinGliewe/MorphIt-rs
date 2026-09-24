//! Configuration, mirroring `config.py` field for field so result JSON stays
//! compatible with the Python tooling.

use std::fmt;
use std::ops::{Index, IndexMut};
use std::str::FromStr;

use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::{Error, Result};

/// The twelve loss terms, in the order Python's `compute_all_losses` returns them.
/// The total loss is accumulated in this order.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum LossId {
    Coverage = 0,
    Overlap,
    Boundary,
    Surface,
    Containment,
    Sqem,
    Hausdorff,
    MeshContainment,
    Mass,
    Com,
    Inertia,
    /// Not implemented in the port; its weight must be zero (it is zero in every preset).
    Flatness,
}

impl LossId {
    pub const COUNT: usize = 12;
    pub const ALL: [LossId; 12] = [
        LossId::Coverage,
        LossId::Overlap,
        LossId::Boundary,
        LossId::Surface,
        LossId::Containment,
        LossId::Sqem,
        LossId::Hausdorff,
        LossId::MeshContainment,
        LossId::Mass,
        LossId::Com,
        LossId::Inertia,
        LossId::Flatness,
    ];

    #[inline]
    pub fn index(self) -> usize {
        self as usize
    }

    /// Key used by Python for this loss in loss dictionaries and training logs.
    pub fn name(self) -> &'static str {
        match self {
            LossId::Coverage => "coverage_loss",
            LossId::Overlap => "overlap_penalty",
            LossId::Boundary => "boundary_penalty",
            LossId::Surface => "surface_loss",
            LossId::Containment => "containment_loss",
            LossId::Sqem => "sqem_loss",
            LossId::Hausdorff => "hausdorff_loss",
            LossId::MeshContainment => "mesh_containment_loss",
            LossId::Mass => "mass_loss",
            LossId::Com => "com_loss",
            LossId::Inertia => "inertia_loss",
            LossId::Flatness => "flatness_loss",
        }
    }

    /// Name of the `TrainingConfig` field holding this loss's weight.
    pub fn weight_key(self) -> &'static str {
        match self {
            LossId::Coverage => "coverage_weight",
            LossId::Overlap => "overlap_weight",
            LossId::Boundary => "boundary_weight",
            LossId::Surface => "surface_weight",
            LossId::Containment => "containment_weight",
            LossId::Sqem => "sqem_weight",
            LossId::Hausdorff => "hausdorff_weight",
            LossId::MeshContainment => "mesh_containment_weight",
            LossId::Mass => "mass_weight",
            LossId::Com => "com_weight",
            LossId::Inertia => "inertia_weight",
            LossId::Flatness => "flatness_weight",
        }
    }
}

/// One weight per [`LossId`].
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct LossWeights(pub [f64; LossId::COUNT]);

impl Index<LossId> for LossWeights {
    type Output = f64;
    fn index(&self, id: LossId) -> &f64 {
        &self.0[id.index()]
    }
}

impl IndexMut<LossId> for LossWeights {
    fn index_mut(&mut self, id: LossId) -> &mut f64 {
        &mut self.0[id.index()]
    }
}

impl LossWeights {
    /// True when the loss contributes (Python skips zero-weight losses entirely).
    #[inline]
    pub fn active(&self, id: LossId) -> bool {
        self[id] != 0.0
    }
}

/// Named loss-weight presets from `LOSS_WEIGHT_CONFIGS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Preset {
    /// Volume-focused, conservative approximation for collision avoidance.
    V,
    /// Surface-focused, precise approximation for contact-rich manipulation.
    S,
    /// Balanced (the Python default).
    B,
    /// Physics-aware: geometry plus mass, center of mass and inertia.
    Obj,
    /// Same weights as `Obj`; meant to be combined with `model.per_sphere_mass = true`.
    ObjMass,
}

impl Preset {
    pub const ALL: [Preset; 5] = [Preset::V, Preset::S, Preset::B, Preset::Obj, Preset::ObjMass];

    pub fn name(self) -> &'static str {
        match self {
            Preset::V => "MorphIt-V",
            Preset::S => "MorphIt-S",
            Preset::B => "MorphIt-B",
            Preset::Obj => "MorphIt-Obj",
            Preset::ObjMass => "MorphIt-Obj-mass",
        }
    }

    /// Loss weights in [`LossId`] order, copied from `config.py::LOSS_WEIGHT_CONFIGS`.
    pub fn weights(self) -> LossWeights {
        // [coverage, overlap, boundary, surface, containment, sqem,
        //  hausdorff, mesh_containment, mass, com, inertia, flatness]
        LossWeights(match self {
            Preset::V => [5000.0, 0.1, 1.0, 0.1, 1.0, 10.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            Preset::S => [0.01, 0.01, 1000.0, 10.0, 1.0, 1000.0, 10.0, 500.0, 0.0, 0.0, 0.0, 0.0],
            Preset::B => [1500.0, 0.3, 1.0, 50.0, 10.0, 3000.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0],
            Preset::Obj | Preset::ObjMass => {
                [100.0, 0.3, 0.0, 50.0, 1.0, 50.0, 10.0, 500.0, 30.0, 10.0, 30.0, 0.0]
            }
        })
    }
}

impl fmt::Display for Preset {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

impl FromStr for Preset {
    type Err = Error;
    fn from_str(s: &str) -> Result<Self> {
        Preset::ALL.into_iter().find(|p| p.name().eq_ignore_ascii_case(s)).ok_or_else(|| {
            let names: Vec<_> = Preset::ALL.iter().map(|p| p.name()).collect();
            Error::config("preset", format!("unknown preset `{s}`; available: {}", names.join(", ")))
        })
    }
}

/// `ModelConfig` from `config.py`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct ModelConfig {
    pub num_spheres: usize,
    /// Informational; a session records the path of the mesh it was built from.
    pub mesh_path: String,
    /// Compute device: `auto` (default), `cpu`, `gpu` or `gpu:N`; Python's
    /// `cuda`/`cuda:N`/`mps` are accepted as aliases for `gpu`. See [`crate::Device`].
    pub device: String,
    /// Log-normal sigma of the initial radius distribution.
    pub initial_radius_variation: f64,
    pub num_inside_samples: usize,
    pub num_surface_samples: usize,
    /// Unused by the optimizer (sphere count is preserved); kept for JSON compatibility.
    pub max_spheres: usize,
    /// Material density (kg/m^3) used by the mass, COM and inertia losses.
    pub density: f64,
    /// Learn per-sphere masses instead of deriving them from radii and density.
    pub per_sphere_mass: bool,
    /// Union overlapping closed bodies before sampling (see [`crate::mesh_prep`]).
    /// CAD exports with several overlapping solids otherwise read as hollow in
    /// the overlap and lose spheres there. `false` packs the mesh exactly as loaded.
    pub union_overlapping_bodies: bool,
    /// Replace every body with its convex hull before the union (see
    /// [`crate::mesh_prep`]): a simpler, closed shape to approximate, for
    /// collision models where concavities do not matter. Not in Python; off
    /// by default.
    pub convex_hull: bool,
}

impl Default for ModelConfig {
    fn default() -> Self {
        ModelConfig {
            num_spheres: 25,
            mesh_path: String::new(),
            device: "auto".into(),
            initial_radius_variation: 0.1,
            num_inside_samples: 5000,
            num_surface_samples: 5000,
            max_spheres: 25,
            density: 1000.0,
            per_sphere_mass: false,
            union_overlapping_bodies: true,
            convex_hull: false,
        }
    }
}

/// `TrainingConfig` from `config.py`. Loss weights default to the un-preset values.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct TrainingConfig {
    pub iterations: usize,
    pub verbose_frequency: usize,
    /// Unused by the port; kept for JSON compatibility.
    pub logging_enabled: bool,
    pub center_lr: f64,
    /// Learning rate in raw (pre-softplus) radius space.
    pub radius_lr: f64,
    /// Learning rate in raw (pre-softplus) mass space; only used with per-sphere mass.
    pub mass_lr: f64,
    pub grad_clip_norm: f64,
    pub coverage_weight: f64,
    pub overlap_weight: f64,
    pub boundary_weight: f64,
    pub surface_weight: f64,
    pub containment_weight: f64,
    pub sqem_weight: f64,
    pub mass_weight: f64,
    pub com_weight: f64,
    pub inertia_weight: f64,
    pub flatness_weight: f64,
    pub hausdorff_weight: f64,
    pub mesh_containment_weight: f64,
    pub early_stopping: bool,
    pub convergence_patience: usize,
    pub convergence_threshold: f64,
    pub density_control_enabled: bool,
    pub density_control_min_interval: usize,
    pub density_control_patience: usize,
    pub density_control_grad_threshold: f64,
    pub density_control_warmup_steps: usize,
    pub density_control_cooling_factor: f64,
    pub density_control_min_radius_fraction: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        TrainingConfig {
            iterations: 300,
            verbose_frequency: 50,
            logging_enabled: false,
            center_lr: 0.002,
            radius_lr: 0.005,
            mass_lr: 0.01,
            grad_clip_norm: 1.0,
            coverage_weight: 10.0,
            overlap_weight: 0.01,
            boundary_weight: 5.0,
            surface_weight: 5.0,
            containment_weight: 5.0,
            sqem_weight: 800.0,
            mass_weight: 0.0,
            com_weight: 0.0,
            inertia_weight: 0.0,
            flatness_weight: 0.0,
            hausdorff_weight: 0.0,
            mesh_containment_weight: 0.0,
            early_stopping: false,
            convergence_patience: 50,
            convergence_threshold: 0.001,
            density_control_enabled: true,
            density_control_min_interval: 160,
            density_control_patience: 1,
            density_control_grad_threshold: 1e-4,
            density_control_warmup_steps: 10,
            density_control_cooling_factor: 0.85,
            density_control_min_radius_fraction: 0.001,
        }
    }
}

impl TrainingConfig {
    /// Current loss weights in [`LossId`] order.
    pub fn loss_weights(&self) -> LossWeights {
        LossWeights([
            self.coverage_weight,
            self.overlap_weight,
            self.boundary_weight,
            self.surface_weight,
            self.containment_weight,
            self.sqem_weight,
            self.hausdorff_weight,
            self.mesh_containment_weight,
            self.mass_weight,
            self.com_weight,
            self.inertia_weight,
            self.flatness_weight,
        ])
    }

    /// Overwrite all twelve weight fields.
    pub fn set_loss_weights(&mut self, w: &LossWeights) {
        self.coverage_weight = w[LossId::Coverage];
        self.overlap_weight = w[LossId::Overlap];
        self.boundary_weight = w[LossId::Boundary];
        self.surface_weight = w[LossId::Surface];
        self.containment_weight = w[LossId::Containment];
        self.sqem_weight = w[LossId::Sqem];
        self.hausdorff_weight = w[LossId::Hausdorff];
        self.mesh_containment_weight = w[LossId::MeshContainment];
        self.mass_weight = w[LossId::Mass];
        self.com_weight = w[LossId::Com];
        self.inertia_weight = w[LossId::Inertia];
        self.flatness_weight = w[LossId::Flatness];
    }
}

/// `VisualizationConfig` from `config.py`. The port does not render; the values
/// are carried only so result JSON matches the Python schema.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct VisualizationConfig {
    pub enabled: bool,
    pub off_screen: bool,
    pub save_video: bool,
    pub video_filename: String,
    pub render_interval: usize,
    pub sphere_color: String,
    pub sphere_opacity: f64,
    pub mesh_color: String,
    pub mesh_line_width: f64,
    pub mesh_opacity: f64,
    pub show_sample_points: bool,
    pub show_surface_points: bool,
    pub sample_points_subsample: usize,
    pub surface_points_subsample: usize,
    pub sample_point_color: String,
    pub surface_point_color: String,
    pub point_size: usize,
    pub camera_position: [f64; 3],
    pub camera_focal_point: [f64; 3],
    pub camera_view_up: [f64; 3],
    pub camera_azimuth: f64,
    pub camera_elevation: f64,
    pub camera_roll: f64,
    pub camera_zoom: f64,
}

impl Default for VisualizationConfig {
    fn default() -> Self {
        VisualizationConfig {
            enabled: false,
            off_screen: true,
            save_video: true,
            video_filename: "sphere_filling.mp4".into(),
            render_interval: 5,
            sphere_color: "blue".into(),
            sphere_opacity: 0.3,
            mesh_color: "white".into(),
            mesh_line_width: 1.5,
            mesh_opacity: 0.8,
            show_sample_points: true,
            show_surface_points: true,
            sample_points_subsample: 100_000,
            surface_points_subsample: 100_000,
            sample_point_color: "red".into(),
            surface_point_color: "green".into(),
            point_size: 5,
            camera_position: [1.0, 1.0, 1.0],
            camera_focal_point: [0.0, 0.0, 0.0],
            camera_view_up: [0.0, 0.0, 1.0],
            camera_azimuth: 80.0,
            camera_elevation: 120.0,
            camera_roll: 120.0,
            camera_zoom: 1.5,
        }
    }
}

/// `MorphItConfig` from `config.py`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Config {
    pub model: ModelConfig,
    pub training: TrainingConfig,
    pub visualization: VisualizationConfig,
    pub results_dir: String,
    pub output_filename: String,
    /// `None` seeds from OS entropy; `Some(seed)` makes the run fully reproducible.
    pub random_seed: Option<u64>,
}

impl Default for Config {
    fn default() -> Self {
        Config {
            model: ModelConfig::default(),
            training: TrainingConfig::default(),
            visualization: VisualizationConfig::default(),
            results_dir: "results/output".into(),
            output_filename: "morphit_results.json".into(),
            random_seed: None,
        }
    }
}

impl Config {
    /// `get_config(preset)`: defaults with the preset's loss weights applied.
    pub fn from_preset(preset: Preset) -> Self {
        let mut c = Config::default();
        c.training.set_loss_weights(&preset.weights());
        c
    }

    /// `get_config(name)` by preset name, e.g. `"MorphIt-B"`.
    pub fn from_preset_name(name: &str) -> Result<Self> {
        Ok(Self::from_preset(name.parse()?))
    }

    /// Parse a full (possibly partial) nested config JSON; missing fields keep their defaults.
    pub fn from_json_str(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| Error::config("<json>", e.to_string()))
    }

    /// Serialize as nested JSON, the same shape Python writes under `"config"`.
    pub fn to_json_value(&self) -> Value {
        serde_json::to_value(self).expect("config serializes")
    }

    /// Read one value by dotted key (`"training.iterations"`) or top-level key (`"random_seed"`).
    pub fn get(&self, key: &str) -> Result<Value> {
        let root = self.to_json_value();
        lookup(&root, key).cloned()
    }

    /// Set one value by dotted key, like Python's `update_config_from_dict`.
    ///
    /// Unknown sections or parameters and type mismatches are errors. An integral
    /// float is accepted for an integer field so C callers can use one setter.
    /// On error the config is left unchanged.
    pub fn set(&mut self, key: &str, value: Value) -> Result<()> {
        let mut root = self.to_json_value();
        let slot = lookup_mut(&mut root, key)?;
        if slot.is_object() {
            return Err(Error::config(key, "is a section, not a parameter"));
        }
        let value = coerce_like(slot, value);
        *slot = value;
        let updated: Config = serde_json::from_value(root).map_err(|e| Error::config(key, e.to_string()))?;
        *self = updated;
        Ok(())
    }

    /// Apply a JSON object of dotted-key updates, e.g. `{"training.iterations": 100}`.
    /// All-or-nothing: on error the config is left unchanged.
    pub fn apply_updates_json(&mut self, json: &str) -> Result<()> {
        let v: Value = serde_json::from_str(json).map_err(|e| Error::config("<json>", e.to_string()))?;
        let obj =
            v.as_object().ok_or_else(|| Error::config("<json>", "expected a JSON object of dotted keys"))?;
        let mut next = self.clone();
        for (k, val) in obj {
            next.set(k, val.clone())?;
        }
        *self = next;
        Ok(())
    }

    /// Check that a session can run with this config.
    pub fn validate(&self) -> Result<()> {
        let m = &self.model;
        let t = &self.training;
        if m.num_spheres == 0 {
            return Err(Error::config("model.num_spheres", "must be at least 1"));
        }
        if m.num_inside_samples == 0 {
            return Err(Error::config("model.num_inside_samples", "must be at least 1"));
        }
        if m.num_surface_samples == 0 {
            return Err(Error::config("model.num_surface_samples", "must be at least 1"));
        }
        crate::device::Device::parse(&m.device)?;
        if !(m.density.is_finite() && m.density > 0.0) {
            return Err(Error::config("model.density", "must be a positive finite number"));
        }
        if !(m.initial_radius_variation.is_finite() && m.initial_radius_variation >= 0.0) {
            return Err(Error::config("model.initial_radius_variation", "must be >= 0"));
        }
        if t.verbose_frequency == 0 {
            return Err(Error::config("training.verbose_frequency", "must be at least 1"));
        }
        for (k, v) in [
            ("training.center_lr", t.center_lr),
            ("training.radius_lr", t.radius_lr),
            ("training.mass_lr", t.mass_lr),
            ("training.grad_clip_norm", t.grad_clip_norm),
        ] {
            if !(v.is_finite() && v >= 0.0) {
                return Err(Error::config(k, "must be a finite number >= 0"));
            }
        }
        let w = t.loss_weights();
        for id in LossId::ALL {
            if !w[id].is_finite() {
                return Err(Error::config(format!("training.{}", id.weight_key()), "must be finite"));
            }
        }
        if t.flatness_weight != 0.0 {
            return Err(Error::config(
                "training.flatness_weight",
                "the flatness loss is not implemented in the Rust port; set it to 0",
            ));
        }
        if t.density_control_patience == 0 {
            return Err(Error::config("training.density_control_patience", "must be at least 1"));
        }
        if t.convergence_patience == 0 {
            return Err(Error::config("training.convergence_patience", "must be at least 1"));
        }
        if !(t.density_control_cooling_factor > 0.0 && t.density_control_cooling_factor <= 1.0) {
            return Err(Error::config("training.density_control_cooling_factor", "must be in (0, 1]"));
        }
        Ok(())
    }
}

fn lookup<'a>(root: &'a Value, key: &str) -> Result<&'a Value> {
    match key.split_once('.') {
        Some((section, param)) => {
            let sec = root
                .get(section)
                .filter(|v| v.is_object())
                .ok_or_else(|| Error::config(key, format!("unknown section `{section}`")))?;
            sec.get(param).ok_or_else(|| {
                Error::config(key, format!("unknown parameter `{param}` in section `{section}`"))
            })
        }
        None => root.get(key).ok_or_else(|| Error::config(key, "unknown parameter")),
    }
}

fn lookup_mut<'a>(root: &'a mut Value, key: &str) -> Result<&'a mut Value> {
    match key.split_once('.') {
        Some((section, param)) => {
            let sec = root
                .get_mut(section)
                .filter(|v| v.is_object())
                .ok_or_else(|| Error::config(key, format!("unknown section `{section}`")))?;
            sec.get_mut(param).ok_or_else(|| {
                Error::config(key, format!("unknown parameter `{param}` in section `{section}`"))
            })
        }
        None => root.get_mut(key).ok_or_else(|| Error::config(key, "unknown parameter")),
    }
}

/// Convert an integral, non-negative float to an integer when the slot currently
/// holds an integer (or is the nullable integer `random_seed`).
fn coerce_like(slot: &Value, value: Value) -> Value {
    let slot_is_int = slot.is_u64() || slot.is_i64() || slot.is_null();
    if slot_is_int && value.is_f64() {
        let f = value.as_f64().unwrap_or(f64::NAN);
        if f.fract() == 0.0 && f >= 0.0 && f <= u64::MAX as f64 {
            return Value::from(f as u64);
        }
    }
    value
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn presets_match_python_tables() {
        let v = Config::from_preset(Preset::V).training;
        assert_eq!(
            (v.coverage_weight, v.overlap_weight, v.boundary_weight, v.surface_weight),
            (5000.0, 0.1, 1.0, 0.1)
        );
        assert_eq!((v.containment_weight, v.sqem_weight, v.hausdorff_weight), (1.0, 10.0, 0.0));
        assert_eq!(v.mesh_containment_weight, 100.0);
        assert_eq!((v.mass_weight, v.com_weight, v.inertia_weight, v.flatness_weight), (0.0, 0.0, 0.0, 0.0));

        let s = Config::from_preset(Preset::S).training;
        assert_eq!(
            (s.coverage_weight, s.overlap_weight, s.boundary_weight, s.surface_weight),
            (0.01, 0.01, 1000.0, 10.0)
        );
        assert_eq!((s.containment_weight, s.sqem_weight, s.hausdorff_weight), (1.0, 1000.0, 10.0));
        assert_eq!(s.mesh_containment_weight, 500.0);

        let b = Config::from_preset(Preset::B).training;
        assert_eq!(
            (b.coverage_weight, b.overlap_weight, b.boundary_weight, b.surface_weight),
            (1500.0, 0.3, 1.0, 50.0)
        );
        assert_eq!((b.containment_weight, b.sqem_weight, b.hausdorff_weight), (10.0, 3000.0, 0.0));
        assert_eq!(b.mesh_containment_weight, 10.0);

        let o = Config::from_preset(Preset::Obj).training;
        assert_eq!(
            (o.coverage_weight, o.overlap_weight, o.boundary_weight, o.surface_weight),
            (100.0, 0.3, 0.0, 50.0)
        );
        assert_eq!((o.containment_weight, o.sqem_weight, o.hausdorff_weight), (1.0, 50.0, 10.0));
        assert_eq!((o.mass_weight, o.com_weight, o.inertia_weight), (30.0, 10.0, 30.0));
        assert_eq!(o.mesh_containment_weight, 500.0);
        assert_eq!(Preset::ObjMass.weights(), Preset::Obj.weights());
        // Python's get_config only changes weights; per-sphere mass stays off.
        assert!(!Config::from_preset(Preset::ObjMass).model.per_sphere_mass);
    }

    #[test]
    fn defaults_match_python() {
        let c = Config::default();
        assert_eq!(c.model.num_spheres, 25);
        assert_eq!(c.model.num_inside_samples, 5000);
        assert_eq!(c.model.density, 1000.0);
        let t = &c.training;
        assert_eq!((t.iterations, t.verbose_frequency), (300, 50));
        assert_eq!((t.center_lr, t.radius_lr, t.mass_lr, t.grad_clip_norm), (0.002, 0.005, 0.01, 1.0));
        assert_eq!((t.density_control_min_interval, t.density_control_patience), (160, 1));
        assert_eq!(t.density_control_warmup_steps, 10);
        assert_eq!(t.density_control_cooling_factor, 0.85);
        assert_eq!(t.density_control_min_radius_fraction, 0.001);
        assert_eq!(t.density_control_grad_threshold, 1e-4);
        assert_eq!(t.sqem_weight, 800.0);
        assert!(c.random_seed.is_none());
    }

    #[test]
    fn preset_names_round_trip() {
        for p in Preset::ALL {
            assert_eq!(p.name().parse::<Preset>().unwrap(), p);
        }
        assert_eq!("morphit-b".parse::<Preset>().unwrap(), Preset::B);
        assert!("MorphIt-X".parse::<Preset>().is_err());
    }

    #[test]
    fn dotted_set_and_get() {
        let mut c = Config::default();
        c.set("training.iterations", json!(100)).unwrap();
        assert_eq!(c.training.iterations, 100);
        c.set("training.iterations", json!(120.0)).unwrap();
        assert_eq!(c.training.iterations, 120);
        c.set("model.density", json!(500)).unwrap();
        assert_eq!(c.model.density, 500.0);
        c.set("random_seed", json!(7)).unwrap();
        assert_eq!(c.random_seed, Some(7));
        c.set("random_seed", Value::Null).unwrap();
        assert_eq!(c.random_seed, None);
        c.set("model.per_sphere_mass", json!(true)).unwrap();
        assert!(c.model.per_sphere_mass);
        assert_eq!(c.get("training.iterations").unwrap(), json!(120));
    }

    #[test]
    fn dotted_set_rejects_bad_keys_and_types() {
        let mut c = Config::default();
        let before = c.clone();
        assert!(c.set("nope.iterations", json!(1)).is_err());
        assert!(c.set("training.nope", json!(1)).is_err());
        assert!(c.set("nope", json!(1)).is_err());
        assert!(c.set("training", json!(1)).is_err());
        assert!(c.set("training.iterations", json!("many")).is_err());
        assert!(c.set("training.iterations", json!(1.5)).is_err());
        assert!(c.set("training.iterations", json!(-3)).is_err());
        assert_eq!(c, before, "failed sets must not modify the config");
    }

    #[test]
    fn apply_updates_is_atomic() {
        let mut c = Config::default();
        let err = c.apply_updates_json(r#"{"training.iterations": 5, "training.bogus": 1}"#);
        assert!(err.is_err());
        assert_eq!(c.training.iterations, 300);
        c.apply_updates_json(r#"{"training.iterations": 5, "model.num_spheres": 9}"#).unwrap();
        assert_eq!((c.training.iterations, c.model.num_spheres), (5, 9));
    }

    #[test]
    fn json_round_trip_and_partial_parse() {
        let c = Config::from_preset(Preset::S);
        let s = serde_json::to_string(&c).unwrap();
        assert_eq!(Config::from_json_str(&s).unwrap(), c);
        let partial = Config::from_json_str(r#"{"model": {"num_spheres": 3}}"#).unwrap();
        assert_eq!(partial.model.num_spheres, 3);
        assert_eq!(partial.training, TrainingConfig::default());
    }

    #[test]
    fn loss_weights_order_matches_loss_ids() {
        let t = Config::from_preset(Preset::Obj).training;
        let w = t.loss_weights();
        let json = serde_json::to_value(&t).unwrap();
        for id in LossId::ALL {
            assert_eq!(json[id.weight_key()].as_f64().unwrap(), w[id], "{id:?}");
        }
        let mut t2 = TrainingConfig::default();
        t2.set_loss_weights(&w);
        assert_eq!(t2.loss_weights(), w);
    }

    #[test]
    fn validate_catches_bad_values() {
        let mut c = Config::default();
        assert!(c.validate().is_ok());
        c.model.num_spheres = 0;
        assert!(c.validate().is_err());
        let mut c = Config::default();
        c.training.flatness_weight = 1.0;
        assert!(c.validate().is_err());
        let mut c = Config::default();
        c.training.verbose_frequency = 0;
        assert!(c.validate().is_err());
    }
}
