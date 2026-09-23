//! Optimizer configuration for API-driven packs: the preset, the sphere and
//! iteration counts, a seed, and the allow-listed `advanced` overrides.

use morphit::Config;
use serde_json::Value;

use crate::{Error, Result};

/// Presets the web UI offers.
pub const ALLOWED_VARIANTS: [&str; 3] = ["MorphIt-V", "MorphIt-S", "MorphIt-B"];

/// Flat `advanced` keys the UI sends, and the dotted config keys they set.
/// Anything else is dropped silently, as in the Python API.
pub const ADVANCED_OVERRIDE_MAP: [(&str, &str); 13] = [
    ("num_inside_samples", "model.num_inside_samples"),
    ("num_surface_samples", "model.num_surface_samples"),
    ("density_control_enabled", "training.density_control_enabled"),
    ("density_control_min_interval", "training.density_control_min_interval"),
    ("density_control_cooling_factor", "training.density_control_cooling_factor"),
    ("coverage_weight", "training.coverage_weight"),
    ("overlap_weight", "training.overlap_weight"),
    ("boundary_weight", "training.boundary_weight"),
    ("surface_weight", "training.surface_weight"),
    ("containment_weight", "training.containment_weight"),
    ("sqem_weight", "training.sqem_weight"),
    ("hausdorff_weight", "training.hausdorff_weight"),
    ("mesh_containment_weight", "training.mesh_containment_weight"),
];

/// `variant must be one of ('MorphIt-V', 'MorphIt-S', 'MorphIt-B')`.
pub fn variant_error() -> String {
    let names: Vec<String> = ALLOWED_VARIANTS.iter().map(|v| crate::py_repr(v)).collect();
    format!("variant must be one of ({})", names.join(", "))
}

/// Decode the `advanced` form field into dotted-key overrides (sorted by
/// key; they are independent). Unknown keys and nulls are dropped.
pub fn parse_advanced(json: &str) -> Result<Vec<(String, Value)>> {
    if json.is_empty() {
        return Ok(Vec::new());
    }
    let raw: Value = serde_json::from_str(json)
        .map_err(|e| Error::Invalid(format!("advanced must be valid JSON: {e}")))?;
    let Value::Object(map) = raw else {
        return Err(Error::Invalid("advanced must be a JSON object".into()));
    };
    Ok(map
        .into_iter()
        .filter(|(_, v)| !v.is_null())
        .filter_map(|(k, v)| {
            ADVANCED_OVERRIDE_MAP
                .iter()
                .find(|(flat, _)| *flat == k)
                .map(|(_, dotted)| (dotted.to_string(), v))
        })
        .collect())
}

/// What a pack request asks for.
#[derive(Clone, Debug)]
pub struct PackParams {
    pub variant: String,
    pub num_spheres: usize,
    pub iterations: usize,
    pub seed: Option<u64>,
    /// Dotted-key overrides from [`parse_advanced`]; applied last.
    pub advanced: Vec<(String, Value)>,
    /// Mesh preparation: merge overlapping closed bodies into their union
    /// before packing (`model.union_overlapping_bodies`, default true).
    pub union_overlapping_bodies: bool,
}

impl PackParams {
    /// Validate like the API does: the variant, `1..=200` spheres and
    /// `1..=1000` iterations, with the Python messages.
    pub fn validate(&self) -> Result<()> {
        if !ALLOWED_VARIANTS.contains(&self.variant.as_str()) {
            return Err(Error::Invalid(variant_error()));
        }
        if !(1..=200).contains(&self.num_spheres) {
            return Err(Error::Invalid("num_spheres must be in [1, 200]".into()));
        }
        if !(1..=1000).contains(&self.iterations) {
            return Err(Error::Invalid("iterations must be in [1, 1000]".into()));
        }
        Ok(())
    }

    /// The optimizer config: preset, counts, logging and visualization off,
    /// output location, seed, `device`, then the overrides. A bad override
    /// value is an [`Error::Invalid`] carrying morphit's message.
    pub fn config(&self, device: &str, results_dir: &str, output_filename: &str) -> Result<Config> {
        let mut c = Config::from_preset_name(&self.variant).map_err(|e| Error::Invalid(e.to_string()))?;
        c.model.num_spheres = self.num_spheres;
        c.model.device = device.to_string();
        c.training.iterations = self.iterations;
        c.model.union_overlapping_bodies = self.union_overlapping_bodies;
        c.training.logging_enabled = false;
        c.visualization.enabled = false;
        c.visualization.off_screen = true;
        c.visualization.save_video = false;
        c.results_dir = results_dir.to_string();
        c.output_filename = output_filename.to_string();
        if self.seed.is_some() {
            c.random_seed = self.seed;
        }
        for (k, v) in &self.advanced {
            c.set(k, v.clone()).map_err(|e| Error::Invalid(e.to_string()))?;
        }
        c.validate().map_err(|e| Error::Invalid(e.to_string()))?;
        Ok(c)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params() -> PackParams {
        PackParams {
            variant: "MorphIt-B".into(),
            num_spheres: 20,
            iterations: 200,
            seed: None,
            advanced: vec![],
            union_overlapping_bodies: true,
        }
    }

    #[test]
    fn advanced_is_filtered_and_mapped() {
        let a = parse_advanced(
            r#"{"coverage_weight": 2.5, "bogus": 1, "sqem_weight": null, "num_inside_samples": 100}"#,
        )
        .unwrap();
        assert_eq!(
            a,
            vec![
                ("training.coverage_weight".to_string(), Value::from(2.5)),
                ("model.num_inside_samples".to_string(), Value::from(100)),
            ]
        );
        assert!(parse_advanced("").unwrap().is_empty());
        assert!(parse_advanced("{}").unwrap().is_empty());
        assert!(parse_advanced("[").unwrap_err().to_string().starts_with("advanced must be valid JSON: "));
        assert_eq!(parse_advanced("[1]").unwrap_err().to_string(), "advanced must be a JSON object");
    }

    #[test]
    fn mesh_prep_follows_the_flag() {
        let mut p = params();
        assert!(p.config("cpu", "", "x.json").unwrap().model.union_overlapping_bodies);
        p.union_overlapping_bodies = false;
        assert!(!p.config("cpu", "", "x.json").unwrap().model.union_overlapping_bodies);
    }

    #[test]
    fn validation_messages() {
        assert!(params().validate().is_ok());
        let mut p = params();
        p.variant = "MorphIt-Obj".into();
        assert_eq!(
            p.validate().unwrap_err().to_string(),
            "variant must be one of ('MorphIt-V', 'MorphIt-S', 'MorphIt-B')"
        );
        let mut p = params();
        p.num_spheres = 201;
        assert_eq!(p.validate().unwrap_err().to_string(), "num_spheres must be in [1, 200]");
        let mut p = params();
        p.iterations = 0;
        assert_eq!(p.validate().unwrap_err().to_string(), "iterations must be in [1, 1000]");
    }

    #[test]
    fn config_applies_everything() {
        let mut p = params();
        p.seed = Some(7);
        p.advanced = parse_advanced(r#"{"coverage_weight": 3, "density_control_enabled": false}"#).unwrap();
        let c = p.config("cpu", "/tmp/x", "a.json").unwrap();
        assert_eq!(c.model.num_spheres, 20);
        assert_eq!(c.training.iterations, 200);
        assert_eq!(c.random_seed, Some(7));
        assert_eq!(c.model.device, "cpu");
        assert_eq!(c.training.coverage_weight, 3.0);
        assert!(!c.training.density_control_enabled);
        assert_eq!(c.output_filename, "a.json");
        p.advanced = vec![("model.num_inside_samples".into(), Value::from("many"))];
        assert!(matches!(p.config("cpu", "", ""), Err(Error::Invalid(_))));
    }
}
