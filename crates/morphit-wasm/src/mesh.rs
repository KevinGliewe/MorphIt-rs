use std::sync::Arc;

use morphit::Mesh;
use morphit::glam::DVec3;
use wasm_bindgen::prelude::*;

use crate::dto::{MeshInfoDto, to_js};
use crate::err;

/// An immutable triangle mesh.
#[wasm_bindgen(js_name = Mesh)]
#[derive(Clone)]
pub struct JsMesh {
    pub(crate) inner: Arc<Mesh>,
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

    /// The mesh a session packs with `model.union_overlapping_bodies`:
    /// overlapping closed bodies merged into one.
    pub fn prepared(&self) -> JsMesh {
        JsMesh { inner: self.inner.prepared().0 }
    }

    /// What [`JsMesh::prepared`] did (Python `MeshPrepReport` keys).
    #[wasm_bindgen(js_name = prepReport, unchecked_return_type = "MeshPrepReport")]
    pub fn prep_report(&self) -> Result<JsValue, JsError> {
        to_js(&self.inner.prepared().1)
    }
}
