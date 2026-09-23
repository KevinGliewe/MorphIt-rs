//! Plain data handed to JavaScript (camelCase keys; see `types.d.ts`).

use std::collections::BTreeMap;

use morphit::{GpuInfo, InitInfo, LossId, Mesh, StepInfo};
use serde::Serialize;
use wasm_bindgen::prelude::*;

use crate::err;

/// Serialize to a plain JS value: objects for maps, numbers for integers.
pub(crate) fn to_js<T: Serialize + ?Sized>(v: &T) -> Result<JsValue, JsError> {
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible()).map_err(err)
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct GpuInfoDto {
    pub index: usize,
    pub name: String,
    pub backend: String,
    pub kind: String,
    pub software: bool,
}

impl From<GpuInfo> for GpuInfoDto {
    fn from(g: GpuInfo) -> Self {
        GpuInfoDto { index: g.index, name: g.name, backend: g.backend, kind: g.kind, software: g.software }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct DensityDto {
    pub added: usize,
    pub removed: usize,
    pub bad: usize,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct StepInfoDto {
    pub iteration: usize,
    pub total_loss: f64,
    pub weighted_losses: BTreeMap<&'static str, f64>,
    pub raw_losses: BTreeMap<&'static str, f64>,
    pub position_grad_mag: f64,
    pub radius_grad_mag: f64,
    pub num_spheres: usize,
    pub projected: usize,
    pub density_control: Option<DensityDto>,
    pub done: bool,
    pub converged: bool,
    pub seconds: f64,
}

impl From<&StepInfo> for StepInfoDto {
    fn from(s: &StepInfo) -> Self {
        let named =
            |v: &[f64; LossId::COUNT]| LossId::ALL.iter().map(|id| (id.name(), v[id.index()])).collect();
        StepInfoDto {
            iteration: s.iteration,
            total_loss: s.total_loss,
            weighted_losses: named(&s.weighted_losses),
            raw_losses: named(&s.raw_losses),
            position_grad_mag: s.position_grad_mag,
            radius_grad_mag: s.radius_grad_mag,
            num_spheres: s.num_spheres,
            projected: s.projected,
            density_control: s.density_control.map(|d| DensityDto {
                added: d.added,
                removed: d.removed,
                bad: d.bad,
            }),
            done: s.done,
            converged: s.converged,
            seconds: s.seconds,
        }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct InitInfoDto {
    pub seed: u64,
    pub voxel_size: f64,
    pub voxel_candidates: usize,
}

impl From<InitInfo> for InitInfoDto {
    fn from(i: InitInfo) -> Self {
        InitInfoDto { seed: i.seed, voxel_size: i.voxel_size, voxel_candidates: i.voxel_candidates }
    }
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
pub(crate) struct MeshInfoDto {
    pub vertices: usize,
    pub faces: usize,
    pub volume: f64,
    pub area: f64,
    pub scale: f64,
    pub bounds: [[f64; 3]; 2],
    pub center_mass: [f64; 3],
    pub winding_flipped: bool,
    pub source_path: Option<String>,
}

impl From<&Mesh> for MeshInfoDto {
    fn from(m: &Mesh) -> Self {
        let (lo, hi) = m.bounds();
        MeshInfoDto {
            vertices: m.vertices().len(),
            faces: m.faces().len(),
            volume: m.volume(),
            area: m.area(),
            scale: m.scale(),
            bounds: [lo.to_array(), hi.to_array()],
            center_mass: m.center_mass().to_array(),
            winding_flipped: m.winding_flipped(),
            source_path: m.source_path().map(str::to_string),
        }
    }
}

/// A rigid transform: `rotation` column-major 3x3, then `translation`.
#[derive(Serialize)]
pub(crate) struct PoseDto {
    pub rotation: [f64; 9],
    pub translation: [f64; 3],
}

impl From<&morphit_robot::kinematics::Pose> for PoseDto {
    fn from(p: &morphit_robot::kinematics::Pose) -> Self {
        PoseDto { rotation: p.rotation.to_cols_array(), translation: p.translation.to_array() }
    }
}
