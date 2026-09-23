//! Mesh file parsers. Every parser works on an in-memory buffer so uploads
//! need no temporary file; [`crate::Mesh::load`] reads the file and calls
//! [`parse`].
//!
//! All formats yield a triangle soup (positions + index triples). Polygons are
//! fan-triangulated, and multiple objects, groups or scene nodes are
//! concatenated, as trimesh does with `force="mesh"`.

mod dae;
mod ply;

use std::io::Cursor;

use glam::DVec3;

/// Lower-case extensions (without the dot) that [`parse`] understands.
pub const SUPPORTED_EXTENSIONS: &[&str] = &["obj", "stl", "ply", "dae"];

pub(crate) type Soup = (Vec<DVec3>, Vec<[u32; 3]>);

/// Parse `bytes` as the format named by `ext` (case-insensitive, with or
/// without a leading dot). `Err(None)` means the format is not supported;
/// `Err(Some(msg))` is a parse error.
pub(crate) fn parse(bytes: &[u8], ext: &str) -> Result<Soup, Option<String>> {
    let ext = ext.trim_start_matches('.').to_ascii_lowercase();
    match ext.as_str() {
        "obj" => obj(bytes).map_err(Some),
        "stl" => stl(bytes).map_err(Some),
        "ply" => ply::parse(bytes).map_err(Some),
        "dae" => dae::parse(bytes).map_err(Some),
        _ => Err(None),
    }
}

fn obj(bytes: &[u8]) -> Result<Soup, String> {
    let opts =
        tobj::LoadOptions { triangulate: true, single_index: false, ignore_points: true, ignore_lines: true };
    // Materials are irrelevant; a failing material loader only affects the
    // (ignored) material list, never the geometry.
    let (models, _materials) =
        tobj::load_obj_buf(&mut Cursor::new(bytes), &opts, |_| Err(tobj::LoadError::OpenFileFailed))
            .map_err(|e| e.to_string())?;
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for m in &models {
        let base = vertices.len() as u32;
        let pos = &m.mesh.positions;
        vertices.extend(pos.chunks_exact(3).map(|c| DVec3::new(c[0], c[1], c[2])));
        faces.extend(m.mesh.indices.chunks_exact(3).map(|c| [base + c[0], base + c[1], base + c[2]]));
    }
    Ok((vertices, faces))
}

fn stl(bytes: &[u8]) -> Result<Soup, String> {
    let m = stl_io::read_stl(&mut Cursor::new(bytes)).map_err(|e| e.to_string())?;
    let vertices = m.vertices.iter().map(|v| DVec3::new(v[0] as f64, v[1] as f64, v[2] as f64)).collect();
    let faces =
        m.faces.iter().map(|t| [t.vertices[0] as u32, t.vertices[1] as u32, t.vertices[2] as u32]).collect();
    Ok((vertices, faces))
}

/// Fan-triangulate one polygon given as vertex indices (pycollada and trimesh
/// do the same). Polygons with fewer than three corners are dropped.
pub(crate) fn fan(poly: &[u32], faces: &mut Vec<[u32; 3]>) {
    for k in 1..poly.len().saturating_sub(1) {
        faces.push([poly[0], poly[k], poly[k + 1]]);
    }
}
