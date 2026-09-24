//! Robot mode: a URDF package held in memory, inspected, packed link by
//! link, and assembled into a spherical URDF.

use std::cell::RefCell;
use std::collections::BTreeMap;
use std::sync::Arc;

use morphit::Session;
use morphit_robot::assemble::{MemSpheres, load_spheres_str, rewrite_urdf_text};
use morphit_robot::color::safe_color_rgba;
use morphit_robot::config::{PackParams, parse_advanced};
use morphit_robot::inspect::{CollisionItem, InspectionReport, inspect_urdf_in, select_urdf_in};
use morphit_robot::kinematics::zero_config_link_poses;
use morphit_robot::pack::{json_filename, link_config, pack_mesh_path};
use morphit_robot::vfs::MemPackage;
use serde::{Deserialize, Serialize};
use wasm_bindgen::JsCast;
use wasm_bindgen::prelude::*;

use crate::dto::{PoseDto, to_js};
use crate::err;
use crate::mesh::JsMesh;
use crate::session::JsSession;

/// A robot description package (URDF plus meshes) held in memory.
#[wasm_bindgen(js_name = RobotPackage)]
pub struct JsRobotPackage {
    pkg: Arc<MemPackage>,
    spheres: RefCell<MemSpheres>,
}

#[derive(Deserialize)]
#[serde(rename_all = "camelCase")]
struct PackParamsDto {
    variant: Option<String>,
    num_spheres: Option<usize>,
    iterations: Option<usize>,
    seed: Option<u64>,
    /// The web UI's flat `advanced` overrides (`coverage_weight`, ...).
    advanced: Option<serde_json::Value>,
    /// Mesh preparation: merge overlapping bodies (default true).
    union_overlapping_bodies: Option<bool>,
    /// Mesh preparation: convex hull of each body first (default false).
    convex_hull: Option<bool>,
}

#[derive(Serialize)]
#[serde(rename_all = "camelCase")]
struct AssembleDto {
    urdf: String,
    links_with_collisions_replaced: usize,
    mesh_collisions_replaced: usize,
    primitive_collisions_removed: usize,
    sphere_collisions_removed: usize,
    sphere_children_added: usize,
    /// `link[idx]: reason; ...` for pack items without spheres.
    skipped: String,
}

impl JsRobotPackage {
    fn report(&self, urdf: Option<String>) -> Result<InspectionReport, JsError> {
        let key = select_urdf_in(&self.pkg, urdf.as_deref()).map_err(err)?;
        inspect_urdf_in(&self.pkg, &key).map_err(err)
    }
}

#[wasm_bindgen(js_class = RobotPackage)]
impl JsRobotPackage {
    /// An empty package; add files with `addFile`.
    #[wasm_bindgen(constructor)]
    pub fn new() -> JsRobotPackage {
        JsRobotPackage { pkg: Arc::new(MemPackage::new()), spheres: RefCell::default() }
    }

    /// Add a file under its path relative to the package root (for a picked
    /// folder, the `webkitRelativePath`).
    #[wasm_bindgen(js_name = addFile)]
    pub fn add_file(&mut self, path: &str, bytes: &[u8]) -> Result<String, JsError> {
        Arc::make_mut(&mut self.pkg).insert(path, bytes.to_vec()).map_err(err)
    }

    /// A package from `{ path: Uint8Array }` or a `Map` of the same.
    #[wasm_bindgen(js_name = fromFiles)]
    pub fn from_files(
        #[wasm_bindgen(unchecked_param_type = "Record<string, Uint8Array> | Map<string, Uint8Array>")]
        files: JsValue,
    ) -> Result<JsRobotPackage, JsError> {
        let entries: js_sys::Array = match files.dyn_ref::<js_sys::Map>() {
            Some(map) => js_sys::Array::from(&map.entries()),
            None => js_sys::Object::entries(
                files.dyn_ref().ok_or_else(|| JsError::new("files must be an object or a Map"))?,
            ),
        };
        let mut pkg = JsRobotPackage::new();
        for entry in entries.iter() {
            let pair: js_sys::Array = entry.unchecked_into();
            let path = pair.get(0).as_string().ok_or_else(|| JsError::new("file paths must be strings"))?;
            let bytes = js_sys::Uint8Array::new(&pair.get(1)).to_vec();
            pkg.add_file(&path, &bytes)?;
        }
        Ok(pkg)
    }

    /// A package from a `.zip` archive.
    #[wasm_bindgen(js_name = fromZip)]
    pub fn from_zip(bytes: &[u8]) -> Result<JsRobotPackage, JsError> {
        let pkg = MemPackage::from_zip(bytes).map_err(err)?;
        Ok(JsRobotPackage { pkg: Arc::new(pkg), spheres: RefCell::default() })
    }

    /// Every file path, sorted.
    pub fn files(&self) -> Vec<String> {
        self.pkg.files().map(str::to_string).collect()
    }

    /// Every `.urdf` in the package, sorted.
    pub fn urdfs(&self) -> Vec<String> {
        self.pkg.urdfs()
    }

    /// The URDF to use: by basename when `requested` is given, else the only
    /// one (errors as the web API's).
    #[wasm_bindgen(js_name = selectUrdf)]
    pub fn select_urdf(&self, requested: Option<String>) -> Result<String, JsError> {
        select_urdf_in(&self.pkg, requested.as_deref()).map_err(err)
    }

    /// Classify every `<collision>` (the web API's inspection report; paths
    /// are package paths).
    #[wasm_bindgen(unchecked_return_type = "InspectionReport")]
    pub fn inspect(&self, urdf: Option<String>) -> Result<JsValue, JsError> {
        to_js(&self.report(urdf)?)
    }

    /// World pose of every link with all joints at zero.
    #[wasm_bindgen(js_name = linkPoses, unchecked_return_type = "Record<string, Pose>")]
    pub fn link_poses(&self, urdf: Option<String>) -> Result<JsValue, JsError> {
        let key = select_urdf_in(&self.pkg, urdf.as_deref()).map_err(err)?;
        let text = String::from_utf8_lossy(self.pkg.read(&key).unwrap_or_default()).into_owned();
        let poses = zero_config_link_poses(&text).map_err(err)?;
        let out: BTreeMap<String, PoseDto> = poses.iter().map(|(k, p)| (k.clone(), p.into())).collect();
        to_js(&out)
    }

    /// Load a mesh of the package (e.g. an item's `mesh_path`).
    pub fn mesh(&self, path: &str) -> Result<JsMesh, JsError> {
        self.pkg.load_mesh(path).map(JsMesh::new).map_err(err)
    }

    /// A session packing one `pack` item of `inspect()`. `params`:
    /// `{ variant, numSpheres, iterations, seed, advanced,
    /// unionOverlappingBodies, convexHull }` with the web
    /// API's defaults and limits. Store the finished session's `resultJson()`
    /// with `setLinkResult`.
    #[wasm_bindgen(js_name = packLink)]
    pub fn pack_link(
        &self,
        #[wasm_bindgen(unchecked_param_type = "CollisionItem")] item: JsValue,
        #[wasm_bindgen(unchecked_param_type = "PackParams")] params: JsValue,
        device: Option<String>,
    ) -> Result<JsSession, JsError> {
        let item: CollisionItem = serde_wasm_bindgen::from_value(item).map_err(err)?;
        let p: PackParamsDto = serde_wasm_bindgen::from_value(params).map_err(err)?;
        let advanced = match p.advanced {
            Some(v) if !v.is_null() => parse_advanced(&v.to_string()).map_err(err)?,
            _ => Vec::new(),
        };
        let params = PackParams {
            variant: p.variant.unwrap_or_else(|| "MorphIt-B".into()),
            num_spheres: p.num_spheres.unwrap_or(20),
            iterations: p.iterations.unwrap_or(200),
            seed: p.seed,
            advanced,
            union_overlapping_bodies: p.union_overlapping_bodies.unwrap_or(true),
            convex_hull: p.convex_hull.unwrap_or(false),
        };
        params.validate().map_err(err)?;
        let mesh_path = pack_mesh_path(&item).map_err(err)?;
        let (config, _) =
            link_config(&item, &params, device.as_deref().unwrap_or("auto"), "spheres").map_err(err)?;
        let mesh = Arc::new(self.pkg.load_mesh(mesh_path).map_err(err)?);
        Session::new(config, mesh).map(JsSession::from_session).map_err(err)
    }

    /// Record the spheres of `link[index]` from a result JSON.
    #[wasm_bindgen(js_name = setLinkResult)]
    pub fn set_link_result(&self, link: &str, index: usize, result_json: &str) -> Result<(), JsError> {
        let spheres = load_spheres_str(result_json, &json_filename(link, index)).map_err(err)?;
        self.spheres.borrow_mut().0.insert((link.to_string(), index), spheres);
        Ok(())
    }

    /// Forget every recorded link result.
    #[wasm_bindgen(js_name = clearLinkResults)]
    pub fn clear_link_results(&self) {
        self.spheres.borrow_mut().0.clear();
    }

    /// The URDF with every packed collision replaced by sphere links (the
    /// web API's `assemble`). `baseColor` is `#rrggbb`; `colorVariation`
    /// (0..1) spreads the hue over the packed links.
    #[wasm_bindgen(unchecked_return_type = "AssembleResult")]
    pub fn assemble(
        &self,
        urdf: Option<String>,
        base_color: Option<String>,
        color_variation: Option<f64>,
    ) -> Result<JsValue, JsError> {
        let report = self.report(urdf)?;
        let text = String::from_utf8_lossy(self.pkg.read(&report.urdf_path).unwrap_or_default()).into_owned();
        let color = safe_color_rgba(base_color.as_deref());
        let spheres = self.spheres.borrow();
        let (out, stats) = rewrite_urdf_text(
            &text,
            &report,
            &*spheres,
            color,
            color_variation.unwrap_or(0.0).clamp(0.0, 1.0),
        )
        .map_err(err)?;
        to_js(&AssembleDto {
            urdf: out,
            links_with_collisions_replaced: stats.links_with_collisions_replaced,
            mesh_collisions_replaced: stats.mesh_collisions_replaced,
            primitive_collisions_removed: stats.primitive_collisions_removed,
            sphere_collisions_removed: stats.sphere_collisions_removed,
            sphere_children_added: stats.sphere_children_added,
            skipped: stats.skipped_summary(),
        })
    }
}

impl Default for JsRobotPackage {
    fn default() -> Self {
        Self::new()
    }
}
