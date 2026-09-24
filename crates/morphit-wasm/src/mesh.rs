use std::sync::Arc;

use morphit::glam::DVec3;
use morphit::{Mesh, MeshPrepOptions};
use wasm_bindgen::prelude::*;

use crate::dto::{MeshInfoDto, to_js};
use crate::err;

/// An immutable triangle mesh.
#[wasm_bindgen(js_name = Mesh)]
#[derive(Clone)]
pub struct JsMesh {
    pub(crate) inner: Arc<Mesh>,
}

#[derive(serde::Deserialize, Default)]
#[serde(rename_all = "camelCase")]
struct MeshPrepOptionsDto {
    union_overlapping_bodies: Option<bool>,
    convex_hull: Option<bool>,
}

/// `{ unionOverlappingBodies, convexHull }` (either may be missing) as options.
fn prep_options(options: JsValue) -> Result<MeshPrepOptions, JsError> {
    let o: MeshPrepOptionsDto = if options.is_undefined() || options.is_null() {
        Default::default()
    } else {
        serde_wasm_bindgen::from_value(options).map_err(err)?
    };
    Ok(MeshPrepOptions {
        union_overlapping_bodies: o.union_overlapping_bodies.unwrap_or(true),
        convex_hull: o.convex_hull.unwrap_or(false),
    })
}

impl JsMesh {
    pub(crate) fn new(mesh: Mesh) -> Self {
        JsMesh { inner: Arc::new(mesh) }
    }
}

#[wasm_bindgen(js_class = Mesh)]
impl JsMesh {
    /// Parse a mesh file (`ext`: `obj`, `stl`, `ply` or `dae`). `name` is
    /// recorded as the result's `mesh_path`.
    #[wasm_bindgen(js_name = fromBytes)]
    pub fn from_bytes(bytes: &[u8], ext: &str, name: Option<String>) -> Result<JsMesh, JsError> {
        Mesh::load_from_bytes(bytes, ext, name).map(JsMesh::new).map_err(err)
    }

    /// From flat `x y z` vertex coordinates and `a b c` triangle indices.
    #[wasm_bindgen(js_name = fromArrays)]
    pub fn from_arrays(vertices: &[f64], faces: &[u32]) -> Result<JsMesh, JsError> {
        Mesh::from_flat(vertices, faces).map(JsMesh::new).map_err(err)
    }

    /// File extensions [`JsMesh::from_bytes`] accepts.
    #[wasm_bindgen(js_name = supportedExtensions)]
    pub fn supported_extensions() -> Vec<String> {
        Mesh::supported_extensions().iter().map(|s| s.to_string()).collect()
    }

    #[wasm_bindgen(unchecked_return_type = "MeshInfo")]
    pub fn info(&self) -> Result<JsValue, JsError> {
        to_js(&MeshInfoDto::from(&*self.inner))
    }

    /// Flat vertex coordinates.
    pub fn vertices(&self) -> Vec<f64> {
        self.inner.vertices().iter().flat_map(|v| v.to_array()).collect()
    }

    /// Flat triangle indices.
    pub fn faces(&self) -> Vec<u32> {
        self.inner.faces().iter().flatten().copied().collect()
    }

    /// Per point of flat `x y z` coordinates: 1 inside, 0 outside.
    pub fn contains(&self, points: &[f64]) -> Vec<u8> {
        let pts: Vec<DVec3> = points.chunks_exact(3).map(DVec3::from_slice).collect();
        self.inner.contains_many(&pts).into_iter().map(u8::from).collect()
    }

    /// The mesh a session packs with these mesh preparation options
    /// (`{ unionOverlappingBodies, convexHull }`, default merge on, hull off):
    /// overlapping closed bodies merged, bodies replaced by convex hulls.
    pub fn prepared(
        &self,
        #[wasm_bindgen(unchecked_param_type = "MeshPrepOptions | undefined")] options: JsValue,
    ) -> Result<JsMesh, JsError> {
        Ok(JsMesh { inner: self.inner.prepared_with(prep_options(options)?).0 })
    }

    /// The mesh as Wavefront OBJ text (exact coordinates), e.g. of
    /// `mesh.prepared({ convexHull: true })` for download.
    #[wasm_bindgen(js_name = toObj)]
    pub fn to_obj(&self) -> String {
        self.inner.to_obj()
    }

    /// The mesh as binary STL (single precision).
    #[wasm_bindgen(js_name = toStl)]
    pub fn to_stl(&self) -> Vec<u8> {
        self.inner.to_stl()
    }

    /// What [`JsMesh::prepared`] does with the same options (Python
    /// `MeshPrepReport` keys, plus `convex_hull` and `n_hulled`).
    #[wasm_bindgen(js_name = prepReport, unchecked_return_type = "MeshPrepReport")]
    pub fn prep_report(
        &self,
        #[wasm_bindgen(unchecked_param_type = "MeshPrepOptions | undefined")] options: JsValue,
    ) -> Result<JsValue, JsError> {
        to_js(&self.inner.prepared_with(prep_options(options)?).1)
    }
}
