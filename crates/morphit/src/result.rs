//! Packing result in the JSON schema of Python's `MorphIt.save_results`.

use std::path::Path;

use serde::{Deserialize, Serialize};

use crate::config::Config;
use crate::error::{Error, Result};
use crate::mesh_prep::MeshPrepReport;

/// Final (or current) spheres plus the config that produced them.
///
/// Serializes to the same keys as the Python implementation, so existing
/// tooling (`create_object_urdf.py`, the robot pipeline, the web UI) can read it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PackResult {
    pub centers: Vec<[f64; 3]>,
    pub radii: Vec<f64>,
    pub masses: Vec<f64>,
    pub mesh_path: String,
    pub num_spheres: usize,
    pub per_sphere_mass: bool,
    /// What mesh preparation did (absent in files written before it existed).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mesh_prep: Option<MeshPrepReport>,
    pub config: Config,
}

impl PackResult {
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(self).expect("result serializes")
    }

    pub fn from_json_str(s: &str) -> Result<Self> {
        serde_json::from_str(s).map_err(|e| Error::Io(format!("invalid result JSON: {e}")))
    }

    /// Write pretty-printed JSON, creating parent directories as needed.
    pub fn save(&self, path: impl AsRef<Path>) -> Result<()> {
        let path = path.as_ref();
        if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
            std::fs::create_dir_all(dir)
                .map_err(|e| Error::Io(format!("cannot create {}: {e}", dir.display())))?;
        }
        std::fs::write(path, self.to_json_string())
            .map_err(|e| Error::Io(format!("cannot write {}: {e}", path.display())))
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        let s = std::fs::read_to_string(path)
            .map_err(|e| Error::Io(format!("cannot read {}: {e}", path.display())))?;
        Self::from_json_str(&s)
    }
}
