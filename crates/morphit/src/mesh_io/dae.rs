//! COLLADA (`.dae`) reader covering what robot description packages use.
//!
//! Mirrors trimesh's loader (pycollada underneath): the default visual scene
//! is walked from its root nodes, each node's `<matrix>`, `<translate>`,
//! `<rotate>` and `<scale>` are composed in document order, `<instance_node>`
//! references are followed, and every `<instance_geometry>` contributes its
//! `<triangles>`, `<polylist>` and `<polygons>` primitives, fan-triangulated,
//! transformed into the scene frame. `<unit>` and `<up_axis>` are metadata
//! only, as in trimesh. Controllers, cameras and lights are ignored.

use std::collections::HashMap;

use glam::{DMat4, DVec3};
use roxmltree::{Document, Node};

use super::{Soup, fan};

type XNode<'a, 'i> = Node<'a, 'i>;

fn children<'a, 'i>(n: XNode<'a, 'i>, tag: &'static str) -> impl Iterator<Item = XNode<'a, 'i>> {
    n.children().filter(move |c| c.is_element() && c.tag_name().name() == tag)
}

fn child<'a, 'i>(n: XNode<'a, 'i>, tag: &'static str) -> Option<XNode<'a, 'i>> {
    children(n, tag).next()
}

fn numbers(n: XNode) -> Result<Vec<f64>, String> {
    n.text()
        .unwrap_or("")
        .split_ascii_whitespace()
        .map(|t| t.parse::<f64>().map_err(|_| format!("bad number `{t}`")))
        .collect()
}

fn indices(n: XNode) -> Result<Vec<u32>, String> {
    n.text()
        .unwrap_or("")
        .split_ascii_whitespace()
        .map(|t| t.parse::<u32>().map_err(|_| format!("bad index `{t}`")))
        .collect()
}

/// `#id` -> `id`.
fn fragment(url: &str) -> &str {
    url.strip_prefix('#').unwrap_or(url)
}

struct Reader<'a, 'i> {
    by_id: HashMap<&'a str, XNode<'a, 'i>>,
    vertices: Vec<DVec3>,
    faces: Vec<[u32; 3]>,
    /// Position arrays already decoded, keyed by `<source>` id.
    positions: HashMap<&'a str, Vec<DVec3>>,
}

impl<'a, 'i> Reader<'a, 'i> {
    fn lookup(&self, url: &str) -> Result<XNode<'a, 'i>, String> {
        self.by_id.get(fragment(url)).copied().ok_or_else(|| format!("broken reference `{url}`"))
    }

    fn local_matrix(node: XNode) -> Result<DMat4, String> {
        let mut m = DMat4::IDENTITY;
        for t in node.children().filter(|c| c.is_element()) {
            let v = |min: usize| {
                let a = numbers(t)?;
                if a.len() < min {
                    return Err(format!("<{}> needs {min} values", t.tag_name().name()));
                }
                Ok(a)
            };
            let step = match t.tag_name().name() {
                "matrix" => {
                    let a = v(16)?;
                    // COLLADA matrices are row-major.
                    DMat4::from_cols_array(&[
                        a[0], a[4], a[8], a[12], a[1], a[5], a[9], a[13], a[2], a[6], a[10], a[14], a[3],
                        a[7], a[11], a[15],
                    ])
                }
                "translate" => {
                    let a = v(3)?;
                    DMat4::from_translation(DVec3::new(a[0], a[1], a[2]))
                }
                "scale" => {
                    let a = v(3)?;
                    DMat4::from_scale(DVec3::new(a[0], a[1], a[2]))
                }
                "rotate" => {
                    let a = v(4)?;
                    let axis = DVec3::new(a[0], a[1], a[2]);
                    if axis.length_squared() == 0.0 {
                        DMat4::IDENTITY
                    } else {
                        DMat4::from_axis_angle(axis.normalize(), a[3].to_radians())
                    }
                }
                _ => continue,
            };
            m *= step;
        }
        Ok(m)
    }

    fn node(&mut self, node: XNode<'a, 'i>, parent: DMat4, depth: usize) -> Result<(), String> {
        if depth > 64 {
            return Err("node hierarchy too deep (cyclic instance_node?)".into());
        }
        let m = parent * Self::local_matrix(node)?;
        for c in node.children().filter(|c| c.is_element()) {
            match c.tag_name().name() {
                "node" => self.node(c, m, depth + 1)?,
                "instance_node" => {
                    let target = self.lookup(c.attribute("url").unwrap_or(""))?;
                    self.node(target, m, depth + 1)?;
                }
                "instance_geometry" => {
                    let g = self.lookup(c.attribute("url").unwrap_or(""))?;
                    self.geometry(g, m)?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Decode a `<source>` of positions (first three accessor params).
    fn source_positions(&mut self, source: XNode<'a, 'i>) -> Result<Vec<DVec3>, String> {
        let id = source.attribute("id").unwrap_or("");
        if let Some(p) = self.positions.get(id) {
            return Ok(p.clone());
        }
        let arr = child(source, "float_array").ok_or("position source without float_array")?;
        let data = numbers(arr)?;
        let acc = child(source, "technique_common").and_then(|t| child(t, "accessor"));
        let (count, stride, offset) = match acc {
            Some(a) => {
                let get = |k: &str, d: usize| a.attribute(k).and_then(|s| s.parse().ok()).unwrap_or(d);
                (get("count", data.len() / 3), get("stride", 3), get("offset", 0))
            }
            None => (data.len() / 3, 3, 0),
        };
        if stride < 3 {
            return Err("position accessor stride below 3".into());
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let s = offset + i * stride;
            let p = data.get(s..s + 3).ok_or("position accessor exceeds its array")?;
            out.push(DVec3::new(p[0], p[1], p[2]));
        }
        self.positions.insert(id, out.clone());
        Ok(out)
    }

    fn geometry(&mut self, geom: XNode<'a, 'i>, m: DMat4) -> Result<(), String> {
        let Some(mesh) = child(geom, "mesh") else { return Ok(()) };
        let verts = child(mesh, "vertices").ok_or("mesh without <vertices>")?;
        let pos_input = children(verts, "input")
            .find(|i| i.attribute("semantic") == Some("POSITION"))
            .ok_or("<vertices> without POSITION input")?;
        let pos_source = self.lookup(pos_input.attribute("source").unwrap_or(""))?;
        let local = self.source_positions(pos_source)?;
        let base = self.vertices.len() as u32;
        self.vertices.extend(local.iter().map(|&p| m.transform_point3(p)));
        let n = local.len() as u32;

        for prim in mesh.children().filter(|c| c.is_element()) {
            let kind = prim.tag_name().name();
            if !matches!(kind, "triangles" | "polylist" | "polygons") {
                continue;
            }
            let inputs: Vec<_> = children(prim, "input").collect();
            let offset_of =
                |i: &XNode| i.attribute("offset").and_then(|s| s.parse::<usize>().ok()).unwrap_or(0);
            let stride = inputs.iter().map(offset_of).max().map_or(1, |o| o + 1);
            let Some(vin) = inputs.iter().find(|i| i.attribute("semantic") == Some("VERTEX")) else {
                continue; // pycollada: no vertex input, nothing to load
            };
            let voff = offset_of(vin);
            let pick = |p: &[u32]| -> Result<Vec<u32>, String> {
                if p.len() % stride != 0 {
                    return Err("index list length is not a multiple of the input stride".into());
                }
                p.chunks_exact(stride)
                    .map(|c| {
                        let i = c[voff];
                        if i >= n {
                            Err(format!("vertex index {i} out of range ({n})"))
                        } else {
                            Ok(base + i)
                        }
                    })
                    .collect()
            };
            match kind {
                "triangles" => {
                    let Some(p) = child(prim, "p") else { continue };
                    let v = pick(&indices(p)?)?;
                    for t in v.chunks_exact(3) {
                        self.faces.push([t[0], t[1], t[2]]);
                    }
                }
                "polylist" => {
                    let Some(p) = child(prim, "p") else { continue };
                    let v = pick(&indices(p)?)?;
                    let counts = child(prim, "vcount").map(indices).transpose()?.unwrap_or_default();
                    let mut at = 0;
                    for c in counts {
                        let c = c as usize;
                        let poly = v.get(at..at + c).ok_or("vcount exceeds the index list")?;
                        fan(poly, &mut self.faces);
                        at += c;
                    }
                }
                _ => {
                    for p in children(prim, "p") {
                        fan(&pick(&indices(p)?)?, &mut self.faces);
                    }
                }
            }
        }
        Ok(())
    }
}

pub(super) fn parse(bytes: &[u8]) -> Result<Soup, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "COLLADA file is not UTF-8 text")?;
    let doc = Document::parse(text).map_err(|e| format!("invalid XML: {e}"))?;
    let root = doc.root_element();
    if root.tag_name().name() != "COLLADA" {
        return Err("root element is not <COLLADA>".into());
    }
    let by_id = root.descendants().filter_map(|n| n.attribute("id").map(|id| (id, n))).collect();
    let mut r = Reader { by_id, vertices: Vec::new(), faces: Vec::new(), positions: HashMap::new() };

    if let Some(asset) = child(root, "asset") {
        let unit =
            child(asset, "unit").and_then(|u| u.attribute("meter")).and_then(|m| m.parse::<f64>().ok());
        if let Some(u) = unit.filter(|u| (u - 1.0).abs() > 1e-9) {
            tracing::warn!("COLLADA unit is {u} m; coordinates are used unscaled (as trimesh does)");
        }
    }

    // The scene to load: <scene><instance_visual_scene url>, else the first one.
    let scene = match child(root, "scene").and_then(|s| child(s, "instance_visual_scene")) {
        Some(inst) => Some(r.lookup(inst.attribute("url").unwrap_or(""))?),
        None => root.descendants().find(|n| n.has_tag_name("visual_scene")),
    };
    match scene {
        Some(scene) => {
            for node in children(scene, "node") {
                r.node(node, DMat4::IDENTITY, 0)?;
            }
        }
        None => {
            // No scene graph: take every geometry untransformed.
            let geoms: Vec<_> = root.descendants().filter(|n| n.has_tag_name("geometry")).collect();
            for g in geoms {
                r.geometry(g, DMat4::IDENTITY)?;
            }
        }
    }
    if r.faces.is_empty() {
        return Err("no triangle geometry in the scene".into());
    }
    Ok((r.vertices, r.faces))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A unit cube as quads, instanced under a node that rotates it 90 degrees
    /// about Z (row-major matrix) and then translates it via a child node.
    fn cube_dae(prim: &str) -> String {
        let pos = "0 0 0 1 0 0 1 1 0 0 1 0 0 0 1 1 0 1 1 1 1 0 1 1";
        let quads = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]];
        // Index pairs: VERTEX at offset 0, NORMAL at offset 1 (dummy 0).
        let pairs = |q: &[u32]| q.iter().map(|i| format!("{i} 0")).collect::<Vec<_>>().join(" ");
        let body = match prim {
            "polylist" => format!(
                "<polylist count=\"6\"><input semantic=\"VERTEX\" source=\"#v\" offset=\"0\"/>\
                 <input semantic=\"NORMAL\" source=\"#n\" offset=\"1\"/><vcount>4 4 4 4 4 4</vcount>\
                 <p>{}</p></polylist>",
                quads.iter().map(|q| pairs(q)).collect::<Vec<_>>().join(" ")
            ),
            _ => format!(
                "<triangles count=\"12\"><input semantic=\"VERTEX\" source=\"#v\" offset=\"0\"/>\
                 <input semantic=\"NORMAL\" source=\"#n\" offset=\"1\"/><p>{}</p></triangles>",
                quads
                    .iter()
                    .map(|q| format!("{} {}", pairs(&[q[0], q[1], q[2]]), pairs(&[q[0], q[2], q[3]])))
                    .collect::<Vec<_>>()
                    .join(" ")
            ),
        };
        format!(
            r##"<?xml version="1.0"?>
<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
  <asset><unit name="meter" meter="1"/><up_axis>Z_UP</up_axis></asset>
  <library_geometries><geometry id="g"><mesh>
    <source id="p"><float_array id="pa" count="24">{pos}</float_array>
      <technique_common><accessor source="#pa" count="8" stride="3">
        <param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/>
      </accessor></technique_common></source>
    <source id="n"><float_array id="na" count="3">0 0 1</float_array></source>
    <vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
    {body}
  </mesh></geometry></library_geometries>
  <library_visual_scenes><visual_scene id="s">
    <node id="outer"><translate>10 0 0</translate>
      <node id="inner"><matrix>0 -1 0 0 1 0 0 0 0 0 1 0 0 0 0 1</matrix>
        <instance_geometry url="#g"/></node>
    </node>
  </visual_scene></library_visual_scenes>
  <scene><instance_visual_scene url="#s"/></scene>
</COLLADA>"##
        )
    }

    #[test]
    fn transformed_cube_from_polylist_and_triangles() {
        for prim in ["polylist", "triangles"] {
            let (v, f) = parse(cube_dae(prim).as_bytes()).unwrap();
            assert_eq!(f.len(), 12, "{prim}");
            let m = crate::Mesh::from_vertices(v, f, None).unwrap();
            assert!((m.volume() - 1.0).abs() < 1e-12, "{prim}");
            assert!(!m.winding_flipped(), "{prim}");
            // Rotated +90 deg about Z: x' = -y, y' = x; then +10 in x.
            let c = m.center_mass();
            assert!((c - DVec3::new(9.5, 0.5, 0.5)).length() < 1e-12, "{prim}: {c}");
        }
    }

    #[test]
    fn errors_are_reported() {
        assert!(parse(b"<nope/>").unwrap_err().contains("COLLADA"));
        assert!(parse(b"<COLLADA").unwrap_err().contains("invalid XML"));
        let broken = cube_dae("polylist").replace("url=\"#g\"", "url=\"#missing\"");
        assert!(parse(broken.as_bytes()).unwrap_err().contains("broken reference"));
        let short =
            cube_dae("polylist").replace("<translate>10 0 0</translate>", "<translate>10</translate>");
        assert!(parse(short.as_bytes()).unwrap_err().contains("<translate> needs 3 values"));
    }
}
