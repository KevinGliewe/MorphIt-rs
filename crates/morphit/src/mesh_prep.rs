//! Mesh preparation: union overlapping closed bodies (`mesh_prep.py`), and
//! optionally replace every body with its convex hull first.
//!
//! CAD exports often contain several closed solids that overlap, for example
//! a body and a fitting exported as separate parts into one file. The inside
//! test counts ray crossings over all faces, so a region inside two solids
//! has an even crossing count and reads as *outside*. That phantom hollow
//! pulls spheres out of the mesh: interior samples skip it, the internal end
//! caps act as surface for the losses and the projection, density control
//! culls the spheres there and the final prune deletes them.
//!
//! [`prepare_mesh`] detects overlapping closed bodies and replaces the mesh
//! with their boolean union (Manifold, via the pure-Rust `manifold-rust`
//! port). Meshes with a single body, or with disjoint bodies, are returned as
//! the same object so results on clean inputs do not change. Anything that
//! cannot be fixed safely falls back to the raw mesh with a warning.
//!
//! The steps follow the Python implementation and trimesh's semantics:
//! vertices merged by position rounded to 8 decimals, bodies split over
//! faces that share an edge used by exactly two faces, degenerate bodies
//! (< 4 faces or zero volume) dropped, winding made consistent and outward,
//! closed = watertight with consistent winding, overlap = any vertex of one
//! body inside the other by the exact ray-parity test.
//!
//! With [`MeshPrepOptions::convex_hull`] (not in Python, off by default) each
//! body is replaced by its convex hull before the overlap test, so hulls that
//! overlap are unioned and open bodies become closed ones. Without the union
//! the hulls are packed side by side.

use std::collections::{HashMap, VecDeque};
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::Arc;

use glam::DVec3;
use serde::{Deserialize, Serialize};

use crate::contains::{HASH_RESOLUTION, ZRayIntersector};
use crate::mesh::{Mesh, mass_properties};

/// A component with fewer faces than a tetrahedron cannot enclose volume.
const MIN_FACES: usize = 4;
/// Relative tolerance of the union-volume sanity check.
const VOLUME_TOL: f64 = 0.01;
/// trimesh merges vertex positions equal after rounding to this many decimals.
const MERGE_DIGITS: i32 = 8;

/// Which preparation steps run; see [`prepare_mesh`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct MeshPrepOptions {
    /// Union overlapping closed bodies (`model.union_overlapping_bodies`).
    pub union_overlapping_bodies: bool,
    /// Replace each body with its convex hull before the union (`model.convex_hull`).
    pub convex_hull: bool,
}

impl MeshPrepOptions {
    /// The defaults: union on, convex hull off.
    pub const DEFAULT: MeshPrepOptions =
        MeshPrepOptions { union_overlapping_bodies: true, convex_hull: false };
    /// No preparation: the mesh as loaded.
    pub const NONE: MeshPrepOptions = MeshPrepOptions { union_overlapping_bodies: false, convex_hull: false };

    /// True when neither step runs.
    pub fn is_disabled(&self) -> bool {
        !self.union_overlapping_bodies && !self.convex_hull
    }
}

impl Default for MeshPrepOptions {
    fn default() -> Self {
        Self::DEFAULT
    }
}

impl From<&crate::config::ModelConfig> for MeshPrepOptions {
    fn from(m: &crate::config::ModelConfig) -> Self {
        MeshPrepOptions { union_overlapping_bodies: m.union_overlapping_bodies, convex_hull: m.convex_hull }
    }
}

/// What [`prepare_mesh`] found and did; serialized as `mesh_prep` in the
/// result JSON with the keys of Python's `MeshPrepReport.to_dict()`, plus
/// `convex_hull` and `n_hulled`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct MeshPrepReport {
    /// `unchanged`, `unioned`, `hulled`, `skipped` or `disabled`.
    pub action: String,
    pub reason: String,
    /// Non-degenerate connected bodies.
    pub n_bodies: usize,
    pub n_closed: usize,
    pub n_open: usize,
    pub n_degenerate_dropped: usize,
    /// Whether any two bodies overlap.
    pub overlapping: bool,
    pub faces_before: usize,
    pub faces_after: usize,
    pub volume_before: f64,
    pub volume_after: f64,
    pub warnings: Vec<String>,
    /// Whether bodies were replaced by their convex hulls (`model.convex_hull`).
    #[serde(default)]
    pub convex_hull: bool,
    /// Number of convex hulls in the prepared mesh (before any union).
    #[serde(default)]
    pub n_hulled: usize,
}

impl MeshPrepReport {
    fn new(mesh: &Mesh) -> Self {
        MeshPrepReport {
            action: "unchanged".into(),
            reason: String::new(),
            n_bodies: 1,
            n_closed: 0,
            n_open: 0,
            n_degenerate_dropped: 0,
            overlapping: false,
            faces_before: mesh.faces().len(),
            faces_after: mesh.faces().len(),
            volume_before: mesh.volume(),
            volume_after: mesh.volume(),
            warnings: Vec::new(),
            convex_hull: false,
            n_hulled: 0,
        }
    }

    /// The report when both steps are off (`model.union_overlapping_bodies =
    /// false`, `model.convex_hull = false`).
    pub fn disabled(mesh: &Mesh) -> Self {
        MeshPrepReport {
            action: "disabled".into(),
            reason: "model.union_overlapping_bodies is False".into(),
            ..Self::new(mesh)
        }
    }

    /// True when the returned mesh is a union rather than the input.
    pub fn is_unioned(&self) -> bool {
        self.action == "unioned"
    }

    /// True when the returned mesh differs from the input (a union or convex hulls).
    pub fn changed(&self) -> bool {
        self.action == "unioned" || self.action == "hulled"
    }

    fn warn(&mut self, message: String) {
        tracing::warn!("mesh prep: {message}");
        self.warnings.push(message);
    }

    fn skip(&mut self, reason: String, warning: String) {
        self.action = "skipped".into();
        self.reason = reason;
        self.warn(warning);
    }
}

/// Return the mesh MorphIt should pack, plus a report of what was done.
///
/// The returned `Arc` is `mesh` itself unless overlapping closed bodies were
/// unioned or bodies were replaced by their convex hulls. With both steps off
/// it returns `mesh` with action `disabled`. Preparation never fails: any
/// unexpected problem falls back to `mesh` with action `skipped` and a warning.
pub fn prepare_mesh(mesh: &Arc<Mesh>, options: MeshPrepOptions) -> (Arc<Mesh>, MeshPrepReport) {
    if options.is_disabled() {
        return (Arc::clone(mesh), MeshPrepReport::disabled(mesh));
    }
    let mut report = MeshPrepReport::new(mesh);
    let outcome = catch_unwind(AssertUnwindSafe(|| prepare(mesh, options, &mut report)));
    let failure = match outcome {
        Ok(Ok(Some(union))) => return (Arc::new(union), report),
        Ok(Ok(None)) => return (Arc::clone(mesh), report),
        Ok(Err(e)) => e,
        Err(payload) => payload
            .downcast_ref::<&str>()
            .map(|s| s.to_string())
            .or_else(|| payload.downcast_ref::<String>().cloned())
            .unwrap_or_else(|| "internal panic".to_string()),
    };
    let mut report = MeshPrepReport::new(mesh);
    report.skip(
        format!("mesh preparation failed: {failure}"),
        format!("mesh preparation failed ({failure}); packing the raw mesh."),
    );
    (Arc::clone(mesh), report)
}

/// Body of [`prepare_mesh`] once enabled: `Ok(Some(mesh))` for a union or
/// convex hulls, `Ok(None)` to keep the input (the report says why), `Err`
/// on failure.
fn prepare(
    mesh: &Mesh,
    options: MeshPrepOptions,
    report: &mut MeshPrepReport,
) -> Result<Option<Mesh>, String> {
    #[cfg(test)]
    if tests::FAIL_SPLIT.with(|f| f.get()) {
        return Err("simulated failure".into());
    }
    let (vertices, faces) = merge_positions(mesh.vertices(), mesh.faces());
    let bodies = split_bodies(&vertices, &faces);
    let n_all = bodies.len();
    let mut kept: Vec<Body> = bodies.into_iter().filter(|b| !b.is_degenerate()).collect();
    report.n_degenerate_dropped = n_all - kept.len();
    report.n_bodies = kept.len();

    let mut hulled = false;
    if options.convex_hull && !kept.is_empty() {
        match kept.iter().map(hull).collect::<Result<Vec<Body>, UnionError>>() {
            Ok(hulls) => {
                let n = hulls.len();
                // A flat body has a flat hull, which encloses nothing.
                kept = hulls.into_iter().filter(|b| !b.is_degenerate()).collect();
                report.n_degenerate_dropped += n - kept.len();
                report.n_bodies = kept.len();
                report.convex_hull = true;
                report.n_hulled = kept.len();
                hulled = !kept.is_empty();
            }
            Err(UnionError::Unavailable(why) | UnionError::Failed(why)) => {
                report.warn(format!("convex hulls not built ({why}); preparing the mesh without them."));
            }
        }
    }
    // The hulls side by side, when no union follows.
    let hulls_only = |report: &mut MeshPrepReport, kept: &[Body], reason: String| {
        if hulled { hulled_mesh(mesh, kept, report, reason).map(Some) } else { Ok(None) }
    };

    if !options.union_overlapping_bodies {
        let reason = format!("{} replaced by convex hulls", plural(kept.len(), "body", "bodies"));
        return hulls_only(report, &kept, reason);
    }
    if kept.len() <= 1 {
        report.reason = "single body".into();
        return hulls_only(report, &kept, "single body replaced by its convex hull".into());
    }

    for body in &mut kept {
        body.fix_normals();
    }
    let closed: Vec<bool> = kept.iter().map(Body::is_closed).collect();
    report.n_closed = closed.iter().filter(|&&c| c).count();
    report.n_open = kept.len() - report.n_closed;

    report.overlapping = any_overlap(&kept);
    if !report.overlapping {
        report.reason = format!("{} disjoint bodies", kept.len());
        let reason = format!("{} disjoint bodies replaced by convex hulls", kept.len());
        return hulls_only(report, &kept, reason);
    }

    if report.n_open > 0 {
        let (n, k) = (kept.len(), report.n_open);
        report.skip(
            format!("{k} of {n} overlapping bodies are open"),
            format!(
                "mesh has {n} overlapping bodies but {k} are not closed; packing the raw mesh. \
                 The overlap region will be treated as outside. Export the part as a single \
                 solid to fix this."
            ),
        );
        return Ok(None);
    }

    let n = kept.len();
    let (uv, uf) = match union(&kept) {
        Ok(u) => u,
        Err(UnionError::Unavailable(why)) => {
            report.skip(
                why.clone(),
                format!(
                    "mesh has {n} overlapping closed bodies but {why}; packing the raw mesh. \
                     Rebuild with the `union` feature to union them automatically."
                ),
            );
            return Ok(None);
        }
        Err(UnionError::Failed(e)) => {
            report.skip(
                format!("union failed: {e}"),
                format!("union of {n} overlapping bodies failed ({e}); packing the raw mesh."),
            );
            return Ok(None);
        }
    };

    let volumes: Vec<f64> = kept.iter().map(|b| b.signed_volume().abs()).collect();
    let lower = volumes.iter().cloned().fold(0.0, f64::max) * (1.0 - VOLUME_TOL);
    let upper = volumes.iter().sum::<f64>() * (1.0 + VOLUME_TOL);
    let union_body = Body { vertices: uv, faces: uf };
    let watertight = union_body.is_watertight();
    let union_volume = if union_body.faces.is_empty() { 0.0 } else { union_body.signed_volume() };
    let built = if watertight && lower <= union_volume && union_volume <= upper {
        Mesh::from_vertices(union_body.vertices, union_body.faces, mesh.source_path().map(str::to_string))
            .ok()
    } else {
        None
    };
    let Some(union_mesh) = built else {
        let py_bool = if watertight { "True" } else { "False" };
        report.skip(
            "union result failed sanity check".into(),
            format!(
                "union result rejected (watertight={py_bool}, volume={}, expected {}..{}); \
                 packing the raw mesh.",
                py_e3(union_volume),
                py_e3(lower),
                py_e3(upper)
            ),
        );
        return Ok(None);
    };

    report.action = "unioned".into();
    report.reason = if hulled {
        format!("{n} overlapping convex hulls unioned")
    } else {
        format!("{n} overlapping closed bodies unioned")
    };
    report.faces_after = union_mesh.faces().len();
    report.volume_after = union_mesh.volume();
    tracing::info!(
        "mesh prep: unioned {n} overlapping bodies ({} degenerate pieces dropped): faces {} -> {}, \
         volume {} -> {}",
        report.n_degenerate_dropped,
        report.faces_before,
        report.faces_after,
        py_e3(report.volume_before),
        py_e3(report.volume_after)
    );
    Ok(Some(union_mesh))
}

/// The convex hulls in `bodies` as one mesh, with the report filled in.
fn hulled_mesh(
    mesh: &Mesh,
    bodies: &[Body],
    report: &mut MeshPrepReport,
    reason: String,
) -> Result<Mesh, String> {
    let mut vertices = Vec::new();
    let mut faces = Vec::new();
    for body in bodies {
        let base = vertices.len() as u32;
        vertices.extend_from_slice(&body.vertices);
        faces.extend(body.faces.iter().map(|f| f.map(|i| i + base)));
    }
    let hulls = Mesh::from_vertices(vertices, faces, mesh.source_path().map(str::to_string))
        .map_err(|e| format!("convex hull mesh: {e}"))?;
    report.action = "hulled".into();
    report.reason = reason;
    report.faces_after = hulls.faces().len();
    report.volume_after = hulls.volume();
    tracing::info!(
        "mesh prep: {}: faces {} -> {}, volume {} -> {}",
        report.reason,
        report.faces_before,
        report.faces_after,
        py_e3(report.volume_before),
        py_e3(report.volume_after)
    );
    Ok(hulls)
}

/// `1 body`, `3 bodies`.
fn plural(n: usize, one: &str, many: &str) -> String {
    format!("{n} {}", if n == 1 { one } else { many })
}

/// Python's `f"{x:.3e}"` (`1.568e+00`).
fn py_e3(x: f64) -> String {
    if !x.is_finite() {
        return format!("{x}");
    }
    let s = format!("{x:.3e}");
    match s.split_once('e') {
        Some((m, e)) => {
            let exp: i32 = e.parse().unwrap_or(0);
            format!("{m}e{}{:02}", if exp < 0 { '-' } else { '+' }, exp.abs())
        }
        None => s,
    }
}

// ---------------------------------------------------------------------------
// Topology
// ---------------------------------------------------------------------------

/// Merge vertices whose positions agree after rounding to 8 decimals (trimesh
/// `merge_vertices`, round half to even); unreferenced vertices are dropped.
/// The first occurrence of each position is kept, in order.
fn merge_positions(vertices: &[DVec3], faces: &[[u32; 3]]) -> (Vec<DVec3>, Vec<[u32; 3]>) {
    let scale = 10f64.powi(MERGE_DIGITS);
    let key = |v: DVec3| -> [i64; 3] { (v * scale).to_array().map(|c| c.round_ties_even() as i64) };
    let mut index: HashMap<[i64; 3], u32> = HashMap::new();
    let mut remap = vec![u32::MAX; vertices.len()];
    let mut merged = Vec::new();
    let mut out_faces = Vec::with_capacity(faces.len());
    for f in faces {
        out_faces.push(f.map(|i| {
            let i = i as usize;
            if remap[i] == u32::MAX {
                remap[i] = *index.entry(key(vertices[i])).or_insert_with(|| {
                    merged.push(vertices[i]);
                    (merged.len() - 1) as u32
                });
            }
            remap[i]
        }));
    }
    (merged, out_faces)
}

/// Undirected edge key.
fn edge_key(a: u32, b: u32) -> (u32, u32) {
    if a < b { (a, b) } else { (b, a) }
}

/// Directed edges of a face: (v0, v1), (v1, v2), (v2, v0).
fn face_edges(f: &[u32; 3]) -> [(u32, u32); 3] {
    [(f[0], f[1]), (f[1], f[2]), (f[2], f[0])]
}

/// Faces per undirected edge.
fn edge_faces(faces: &[[u32; 3]]) -> HashMap<(u32, u32), Vec<u32>> {
    let mut map: HashMap<(u32, u32), Vec<u32>> = HashMap::with_capacity(faces.len() * 3 / 2);
    for (fi, f) in faces.iter().enumerate() {
        for (a, b) in face_edges(f) {
            map.entry(edge_key(a, b)).or_default().push(fi as u32);
        }
    }
    map
}

/// trimesh `face_adjacency`: pairs of faces sharing an edge used by exactly two faces.
fn face_adjacency(faces: &[[u32; 3]]) -> Vec<Vec<u32>> {
    let mut adj = vec![Vec::new(); faces.len()];
    for users in edge_faces(faces).into_values() {
        if let [a, b] = users[..] {
            adj[a as usize].push(b);
            adj[b as usize].push(a);
        }
    }
    for l in &mut adj {
        l.sort_unstable();
        l.dedup();
    }
    adj
}

/// One connected body with compacted vertices.
#[derive(Clone, Debug)]
struct Body {
    vertices: Vec<DVec3>,
    faces: Vec<[u32; 3]>,
}

/// Connected components of the face adjacency, ordered by their lowest face
/// index, faces ascending, vertices renumbered in ascending original order
/// (trimesh `split(only_watertight=False)` + `submesh`).
fn split_bodies(vertices: &[DVec3], faces: &[[u32; 3]]) -> Vec<Body> {
    let adj = face_adjacency(faces);
    let mut label = vec![usize::MAX; faces.len()];
    let mut groups: Vec<Vec<usize>> = Vec::new();
    for start in 0..faces.len() {
        if label[start] != usize::MAX {
            continue;
        }
        let id = groups.len();
        let mut group = vec![start];
        label[start] = id;
        let mut queue = VecDeque::from([start]);
        while let Some(f) = queue.pop_front() {
            for &g in &adj[f] {
                let g = g as usize;
                if label[g] == usize::MAX {
                    label[g] = id;
                    group.push(g);
                    queue.push_back(g);
                }
            }
        }
        group.sort_unstable();
        groups.push(group);
    }
    groups
        .into_iter()
        .map(|group| {
            let mut used: Vec<u32> = group.iter().flat_map(|&f| faces[f]).collect();
            used.sort_unstable();
            used.dedup();
            let new_index: HashMap<u32, u32> = used.iter().enumerate().map(|(k, &v)| (v, k as u32)).collect();
            Body {
                vertices: used.iter().map(|&v| vertices[v as usize]).collect(),
                faces: group.iter().map(|&f| faces[f].map(|v| new_index[&v])).collect(),
            }
        })
        .collect()
}

impl Body {
    fn signed_volume(&self) -> f64 {
        mass_properties(&self.vertices, &self.faces).0
    }

    fn is_degenerate(&self) -> bool {
        self.faces.len() < MIN_FACES || self.signed_volume().abs() == 0.0
    }

    fn bounds(&self) -> (DVec3, DVec3) {
        self.vertices
            .iter()
            .fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
    }

    /// trimesh `is_watertight` and `is_winding_consistent`: every edge is used by
    /// exactly two faces; for edges used twice, the two uses run in opposite
    /// directions.
    fn watertight_and_winding(&self) -> (bool, bool) {
        let mut uses: HashMap<(u32, u32), Vec<(u32, u32)>> = HashMap::with_capacity(self.faces.len() * 3 / 2);
        for f in &self.faces {
            for (a, b) in face_edges(f) {
                uses.entry(edge_key(a, b)).or_default().push((a, b));
            }
        }
        let mut paired = 0;
        let mut winding = true;
        for u in uses.values() {
            if let [(_, a1), (b0, _)] = u[..] {
                paired += 2;
                winding &= a1 == b0;
            }
        }
        (paired == self.faces.len() * 3, winding)
    }

    fn is_watertight(&self) -> bool {
        !self.faces.is_empty() && self.watertight_and_winding().0
    }

    fn is_closed(&self) -> bool {
        let (watertight, winding) = self.watertight_and_winding();
        watertight && winding
    }

    /// trimesh `fix_normals(multibody=False)`: make the winding consistent by a
    /// breadth-first walk over adjacent faces, then invert a watertight body
    /// with negative volume.
    fn fix_normals(&mut self) {
        let (watertight, winding) = self.watertight_and_winding();
        if !winding {
            let adj = face_adjacency(&self.faces);
            let mut seen = vec![false; self.faces.len()];
            for start in 0..self.faces.len() {
                if seen[start] {
                    continue;
                }
                seen[start] = true;
                let mut queue = VecDeque::from([start]);
                while let Some(f) = queue.pop_front() {
                    for &g in &adj[f] {
                        let g = g as usize;
                        if seen[g] {
                            continue;
                        }
                        seen[g] = true;
                        let fe = face_edges(&self.faces[f]);
                        let same_direction = face_edges(&self.faces[g]).iter().any(|e| fe.contains(e));
                        if same_direction {
                            self.faces[g].reverse();
                        }
                        queue.push_back(g);
                    }
                }
            }
        }
        if (watertight || self.is_watertight()) && self.signed_volume() < 0.0 {
            for f in &mut self.faces {
                f.reverse();
            }
        }
    }

    fn triangles(&self) -> Vec<[DVec3; 3]> {
        self.faces.iter().map(|f| f.map(|i| self.vertices[i as usize])).collect()
    }
}

/// True when any pair of bodies overlaps: bounding boxes intersect with
/// positive extent and a vertex of one lies inside the other by the exact
/// ray-parity test built on that single body (exact for one closed shell).
/// Bodies that only touch along a face may be reported either way.
fn any_overlap(bodies: &[Body]) -> bool {
    let bounds: Vec<(DVec3, DVec3)> = bodies.iter().map(Body::bounds).collect();
    let mut intersectors: Vec<Option<ZRayIntersector>> = (0..bodies.len()).map(|_| None).collect();
    for i in 0..bodies.len() {
        for j in i + 1..bodies.len() {
            let lo = bounds[i].0.max(bounds[j].0);
            let hi = bounds[i].1.min(bounds[j].1);
            if hi.cmple(lo).any() {
                continue;
            }
            for (inner, outer) in [(i, j), (j, i)] {
                let ix = intersectors[outer]
                    .get_or_insert_with(|| ZRayIntersector::new(&bodies[outer].triangles(), HASH_RESOLUTION));
                if bodies[inner].vertices.iter().any(|&p| ix.contains(p)) {
                    return true;
                }
            }
        }
    }
    false
}

// ---------------------------------------------------------------------------
// Boolean union
// ---------------------------------------------------------------------------

enum UnionError {
    /// No union backend in this build (Python: manifold3d not installed).
    #[cfg_attr(feature = "union", allow(dead_code))]
    Unavailable(String),
    /// Manifold rejected a body or the operation failed.
    #[cfg_attr(not(feature = "union"), allow(dead_code))]
    Failed(String),
}

/// Boolean union of closed bodies, each converted to a Manifold from f32
/// positions (as Python's manifold3d `Mesh`), combined left to right. The
/// result's duplicate positions are merged like trimesh's `process=True`.
#[cfg(feature = "union")]
fn union(bodies: &[Body]) -> Result<(Vec<DVec3>, Vec<[u32; 3]>), UnionError> {
    use manifold_rust::manifold::Manifold;
    use manifold_rust::types::{Error as MfError, MeshGL};

    let mut result: Option<Manifold> = None;
    for body in bodies {
        let mesh = MeshGL {
            num_prop: 3,
            vert_properties: body
                .vertices
                .iter()
                .flat_map(|v| [v.x as f32, v.y as f32, v.z as f32])
                .collect(),
            tri_verts: body.faces.iter().flatten().copied().collect(),
            ..Default::default()
        };
        let part = Manifold::from_mesh_gl(&mesh);
        let status = part.status();
        if status != MfError::NoError {
            return Err(UnionError::Failed(format!("manifold rejected a body: Error.{status:?}")));
        }
        result = Some(match result {
            None => part,
            Some(acc) => acc + part,
        });
    }
    let result = result.ok_or_else(|| UnionError::Failed("no bodies".into()))?;
    let status = result.status();
    if status != MfError::NoError {
        return Err(UnionError::Failed(format!("union status Error.{status:?}")));
    }
    let out = result.get_mesh_gl(-1);
    let np = out.num_prop as usize;
    if np < 3 {
        return Err(UnionError::Failed(format!("union mesh has {np} properties per vertex")));
    }
    let vertices: Vec<DVec3> = out
        .vert_properties
        .chunks_exact(np)
        .map(|p| DVec3::new(p[0] as f64, p[1] as f64, p[2] as f64))
        .collect();
    let faces: Vec<[u32; 3]> = out.tri_verts.chunks_exact(3).map(|t| [t[0], t[1], t[2]]).collect();
    Ok(merge_positions(&vertices, &faces))
}

#[cfg(not(feature = "union"))]
fn union(_bodies: &[Body]) -> Result<(Vec<DVec3>, Vec<[u32; 3]>), UnionError> {
    Err(UnionError::Unavailable("mesh union support is not built in (feature `union`)".into()))
}

/// Convex hull of a body's vertices (Manifold's quickhull in f64), wound
/// outward, with duplicate positions merged.
#[cfg(feature = "union")]
fn hull(body: &Body) -> Result<Body, UnionError> {
    use manifold_rust::linalg::Vec3;
    use manifold_rust::manifold::Manifold;
    use manifold_rust::types::Error as MfError;

    let points: Vec<Vec3> = body.vertices.iter().map(|v| Vec3::new(v.x, v.y, v.z)).collect();
    let hull = Manifold::hull(&points);
    let status = hull.status();
    if status != MfError::NoError {
        return Err(UnionError::Failed(format!("convex hull status Error.{status:?}")));
    }
    let out = hull.get_mesh_gl64(-1);
    let np = out.num_prop as usize;
    if np < 3 {
        return Err(UnionError::Failed(format!("hull mesh has {np} properties per vertex")));
    }
    let vertices: Vec<DVec3> =
        out.vert_properties.chunks_exact(np).map(|p| DVec3::new(p[0], p[1], p[2])).collect();
    let faces: Vec<[u32; 3]> =
        out.tri_verts.chunks_exact(3).map(|t| [t[0] as u32, t[1] as u32, t[2] as u32]).collect();
    let (vertices, faces) = merge_positions(&vertices, &faces);
    let mut body = Body { vertices, faces };
    body.fix_normals();
    Ok(body)
}

#[cfg(not(feature = "union"))]
fn hull(_body: &Body) -> Result<Body, UnionError> {
    Err(UnionError::Unavailable("convex hull support is not built in (feature `union`)".into()))
}

#[cfg(test)]
pub(crate) mod tests {
    use std::cell::Cell;

    use super::*;
    use crate::shapes::box_mesh;

    thread_local! {
        /// Makes `prepare` fail, to exercise the fallback path.
        pub(crate) static FAIL_SPLIT: Cell<bool> = const { Cell::new(false) };
    }

    type Parts = (Vec<DVec3>, Vec<[u32; 3]>);

    fn parts(m: &Mesh) -> Parts {
        (m.vertices().to_vec(), m.faces().to_vec())
    }

    fn unit_box(offset: DVec3) -> Parts {
        parts(&box_mesh(offset, offset + DVec3::ONE))
    }

    /// Concatenate parts into one mesh (vertex indices offset, nothing merged).
    fn concat(list: &[Parts]) -> Arc<Mesh> {
        let mut v = Vec::new();
        let mut f = Vec::new();
        for (pv, pf) in list {
            let base = v.len() as u32;
            v.extend_from_slice(pv);
            f.extend(pf.iter().map(|t| t.map(|i| i + base)));
        }
        Arc::new(Mesh::from_vertices(v, f, None).unwrap())
    }

    const SHIFT: DVec3 = DVec3::new(0.4, 0.2, 0.1);

    /// Two unit boxes overlapping in a 0.6 x 0.8 x 0.9 block (union volume 1.568).
    #[cfg(feature = "union")]
    pub(crate) fn two_overlapping_boxes() -> Arc<Mesh> {
        concat(&[unit_box(DVec3::ZERO), unit_box(SHIFT)])
    }

    #[test]
    fn single_body_is_returned_unchanged() {
        let m = Arc::new(box_mesh(DVec3::ZERO, DVec3::ONE));
        let (out, r) = prepare_mesh(&m, MeshPrepOptions::DEFAULT);
        assert!(Arc::ptr_eq(&out, &m));
        assert_eq!((r.action.as_str(), r.reason.as_str(), r.n_bodies), ("unchanged", "single body", 1));
        assert_eq!((r.faces_before, r.faces_after), (12, 12));
        assert!(r.warnings.is_empty());
    }

    #[test]
    fn disabled_returns_the_input() {
        let m = concat(&[unit_box(DVec3::ZERO), unit_box(SHIFT)]);
        let (out, r) = prepare_mesh(&m, MeshPrepOptions::NONE);
        assert!(Arc::ptr_eq(&out, &m));
        assert_eq!(r.action, "disabled");
        assert_eq!(r.reason, "model.union_overlapping_bodies is False");
    }

    #[test]
    fn degenerate_sliver_is_dropped() {
        let sliver: Parts = (
            vec![DVec3::new(5.0, 0.0, 0.0), DVec3::new(5.0, 1.0, 0.0), DVec3::new(5.0, 0.0, 1.0)],
            vec![[0, 1, 2], [0, 2, 1]],
        );
        let m = concat(&[unit_box(DVec3::ZERO), sliver]);
        let (out, r) = prepare_mesh(&m, MeshPrepOptions::DEFAULT);
        assert!(Arc::ptr_eq(&out, &m));
        assert_eq!((r.action.as_str(), r.n_degenerate_dropped, r.n_bodies), ("unchanged", 1, 1));
    }

    #[test]
    fn disjoint_bodies_are_left_alone() {
        let m = concat(&[unit_box(DVec3::ZERO), unit_box(DVec3::new(3.0, 0.0, 0.0))]);
        let (out, r) = prepare_mesh(&m, MeshPrepOptions::DEFAULT);
        assert!(Arc::ptr_eq(&out, &m));
        assert_eq!((r.action.as_str(), r.reason.as_str()), ("unchanged", "2 disjoint bodies"));
        assert_eq!((r.n_bodies, r.n_closed, r.n_open, r.overlapping), (2, 2, 0, false));
    }

    #[test]
    fn overlapping_open_body_is_skipped_with_a_warning() {
        let (v, mut f) = unit_box(SHIFT);
        f.remove(0);
        let m = concat(&[unit_box(DVec3::ZERO), (v, f)]);
        let (out, r) = prepare_mesh(&m, MeshPrepOptions::DEFAULT);
        assert!(Arc::ptr_eq(&out, &m));
        assert_eq!((r.action.as_str(), r.n_open, r.n_closed), ("skipped", 1, 1));
        assert!(r.overlapping);
        assert_eq!(r.reason, "1 of 2 overlapping bodies are open");
        assert_eq!(r.warnings.len(), 1);
    }

    #[test]
    fn internal_failure_falls_back_to_the_raw_mesh() {
        let m = concat(&[unit_box(DVec3::ZERO), unit_box(SHIFT)]);
        FAIL_SPLIT.with(|f| f.set(true));
        let (out, r) = prepare_mesh(&m, MeshPrepOptions::DEFAULT);
        FAIL_SPLIT.with(|f| f.set(false));
        assert!(Arc::ptr_eq(&out, &m));
        assert_eq!(r.action, "skipped");
        assert!(r.reason.contains("simulated failure"), "{}", r.reason);
        assert_eq!(r.warnings.len(), 1);
    }

    #[test]
    fn obj_group_style_duplicated_vertices_form_one_body() {
        // Every face with its own three vertices, as an OBJ with one group per face.
        let (v, f) = unit_box(DVec3::ZERO);
        let mut dv = Vec::new();
        let mut df = Vec::new();
        for t in &f {
            let base = dv.len() as u32;
            dv.extend(t.iter().map(|&i| v[i as usize]));
            df.push([base, base + 1, base + 2]);
        }
        let m = concat(&[(dv, df)]);
        let (out, r) = prepare_mesh(&m, MeshPrepOptions::DEFAULT);
        assert!(Arc::ptr_eq(&out, &m));
        assert_eq!((r.action.as_str(), r.n_bodies), ("unchanged", 1));
    }

    #[test]
    fn topology_helpers() {
        let (v, f) = unit_box(DVec3::ZERO);
        let b = Body { vertices: v.clone(), faces: f.clone() };
        assert_eq!(b.watertight_and_winding(), (true, true));
        // One flipped face: still watertight, winding inconsistent; fix_normals repairs it.
        let mut flipped = b.clone();
        flipped.faces[3].reverse();
        assert_eq!(flipped.watertight_and_winding(), (true, false));
        flipped.fix_normals();
        assert!(flipped.is_closed());
        assert!((flipped.signed_volume() - 1.0).abs() < 1e-12);
        // Fully inverted: consistent but negative volume; fix_normals turns it outward.
        let mut inverted = b.clone();
        inverted.faces.iter_mut().for_each(|t| t.reverse());
        inverted.fix_normals();
        assert!((inverted.signed_volume() - 1.0).abs() < 1e-12);
        // Open: not watertight.
        let open = Body { vertices: v, faces: f[1..].to_vec() };
        assert!(!open.is_closed());
        // Rounding merge: positions 1e-10 apart collapse, 1e-6 apart do not.
        let pts = [DVec3::ZERO, DVec3::splat(1e-10), DVec3::X, DVec3::splat(1e-6)];
        let (mv, mf) = merge_positions(&pts, &[[0, 2, 3], [1, 2, 3]]);
        assert_eq!(mv.len(), 3);
        assert_eq!(mf, vec![[0, 1, 2], [0, 1, 2]]);
        assert_eq!(py_e3(1.568), "1.568e+00");
        assert_eq!(py_e3(-0.00012346), "-1.235e-04");
        assert_eq!(py_e3(2.5e10), "2.500e+10");
    }

    #[cfg(feature = "union")]
    #[test]
    fn overlapping_closed_bodies_are_unioned() {
        let m = concat(&[unit_box(DVec3::ZERO), unit_box(SHIFT)]);
        let (out, r) = prepare_mesh(&m, MeshPrepOptions::DEFAULT);
        assert!(!Arc::ptr_eq(&out, &m));
        let expected = 2.0 - 0.6 * 0.8 * 0.9;
        assert_eq!(r.action, "unioned", "{r:?}");
        assert_eq!(r.reason, "2 overlapping closed bodies unioned");
        assert!((out.volume() - expected).abs() < 0.01 * expected, "volume {}", out.volume());
        assert!(Body { vertices: out.vertices().to_vec(), faces: out.faces().to_vec() }.is_closed());
        assert_eq!(r.faces_after, out.faces().len());
        assert_eq!(r.volume_after, out.volume());
        assert!((r.volume_before - 2.0).abs() < 1e-12);
        // The union really is solid in the former overlap.
        let mid = DVec3::new(0.7, 0.6, 0.55);
        assert!(out.contains(mid) && !m.contains(mid));
    }

    #[cfg(feature = "union")]
    #[test]
    fn inward_wound_inner_shell_is_turned_outward_and_filled() {
        // Python semantics: every body is oriented outward before the overlap
        // test, so an inner shell wound inward (a cavity) counts as a solid
        // overlapping the outer body and the union fills it.
        let (v, mut f) = parts(&box_mesh(SHIFT, SHIFT + DVec3::splat(0.5)));
        f.iter_mut().for_each(|t| t.reverse());
        let m = concat(&[unit_box(DVec3::ZERO), (v, f)]);
        assert!((m.volume() - 0.875).abs() < 1e-12);
        let (out, r) = prepare_mesh(&m, MeshPrepOptions::DEFAULT);
        assert_eq!(r.action, "unioned", "{r:?}");
        assert!((out.volume() - 1.0).abs() < 1e-6, "volume {}", out.volume());
    }

    #[cfg(not(feature = "union"))]
    #[test]
    fn without_the_union_feature_overlaps_are_skipped() {
        let m = concat(&[unit_box(DVec3::ZERO), unit_box(SHIFT)]);
        let (out, r) = prepare_mesh(&m, MeshPrepOptions::DEFAULT);
        assert!(Arc::ptr_eq(&out, &m));
        assert_eq!(r.action, "skipped");
        assert_eq!(r.warnings.len(), 1);
    }

    const HULL: MeshPrepOptions = MeshPrepOptions { union_overlapping_bodies: true, convex_hull: true };
    const HULL_ONLY: MeshPrepOptions = MeshPrepOptions { union_overlapping_bodies: false, convex_hull: true };

    /// A concave body: the union of two overlapping unit boxes (volume 1.568).
    #[cfg(feature = "union")]
    fn concave_body() -> Parts {
        parts(&two_overlapping_boxes().prepared().0)
    }

    /// True when every point lies on the inner side of every face plane of `m`.
    fn inside_all_planes(m: &Mesh, points: &[DVec3]) -> bool {
        m.triangles().all(|[a, b, c]| {
            let n = (b - a).cross(c - a).normalize();
            points.iter().all(|&p| n.dot(p - a) <= 1e-9)
        })
    }

    #[cfg(feature = "union")]
    #[test]
    fn a_concave_body_becomes_its_convex_hull() {
        let body = concave_body();
        let m = concat(std::slice::from_ref(&body));
        let (out, r) = prepare_mesh(&m, HULL);
        assert!(!Arc::ptr_eq(&out, &m));
        assert_eq!(
            (r.action.as_str(), r.reason.as_str()),
            ("hulled", "single body replaced by its convex hull")
        );
        assert!(r.convex_hull && r.n_hulled == 1, "{r:?}");
        // Convex, holds every input vertex, larger than the body, within its box.
        assert!(inside_all_planes(&out, out.vertices()));
        assert!(inside_all_planes(&out, &body.0));
        assert!(out.volume() > 1.568 + 0.05 && out.volume() < 1.4 * 1.2 * 1.1, "volume {}", out.volume());
        assert!(Body { vertices: out.vertices().to_vec(), faces: out.faces().to_vec() }.is_closed());
        assert_eq!((r.faces_after, r.volume_after), (out.faces().len(), out.volume()));
        // A notch of the concave body is solid in the hull.
        let notch = DVec3::new(1.05, 0.12, 0.5);
        assert!(out.contains(notch) && !m.contains(notch));
    }

    #[cfg(feature = "union")]
    #[test]
    fn disjoint_bodies_get_one_hull_each() {
        let (a, (bv, bf)) = (concave_body(), concave_body());
        let far = (bv.iter().map(|&v| v + DVec3::splat(5.0)).collect(), bf);
        let m = concat(&[a, far]);
        let (out, r) = prepare_mesh(&m, HULL);
        assert_eq!((r.action.as_str(), r.n_hulled, r.n_bodies), ("hulled", 2, 2), "{r:?}");
        assert_eq!(r.reason, "2 disjoint bodies replaced by convex hulls");
        let single = prepare_mesh(&concat(&[concave_body()]), HULL).0;
        assert_eq!(out.faces().len(), 2 * single.faces().len());
        assert!((out.volume() - 2.0 * single.volume()).abs() < 1e-9);
    }

    /// The concave body and a small box in its notch: apart as loaded, but
    /// the box lies inside the body's hull.
    #[cfg(feature = "union")]
    fn body_and_box_in_its_notch() -> Arc<Mesh> {
        let notch_box = parts(&box_mesh(DVec3::new(1.02, 0.1, 0.4), DVec3::new(1.08, 0.18, 0.6)));
        concat(&[concave_body(), notch_box])
    }

    #[cfg(feature = "union")]
    #[test]
    fn overlapping_hulls_are_unioned() {
        let m = body_and_box_in_its_notch();
        let (_, raw) = prepare_mesh(&m, MeshPrepOptions::DEFAULT);
        assert_eq!((raw.action.as_str(), raw.overlapping), ("unchanged", false));

        let (out, r) = prepare_mesh(&m, HULL);
        assert_eq!(r.action, "unioned", "{r:?}");
        assert_eq!(r.reason, "2 overlapping convex hulls unioned");
        assert!(r.convex_hull && r.n_hulled == 2 && r.overlapping);
        // The box is swallowed: the union is the body's hull.
        let hull = prepare_mesh(&concat(&[concave_body()]), HULL).0;
        assert!(
            (out.volume() - hull.volume()).abs() < 1e-5 * hull.volume(),
            "{} vs {}",
            out.volume(),
            hull.volume()
        );
    }

    #[cfg(feature = "union")]
    #[test]
    fn without_the_union_hulls_are_packed_side_by_side() {
        let m = body_and_box_in_its_notch();
        let (out, r) = prepare_mesh(&m, HULL_ONLY);
        assert_eq!((r.action.as_str(), r.n_hulled), ("hulled", 2), "{r:?}");
        assert_eq!(r.reason, "2 bodies replaced by convex hulls");
        let hull = prepare_mesh(&concat(&[concave_body()]), HULL).0;
        assert_eq!(out.faces().len(), hull.faces().len() + 12);
    }

    #[cfg(feature = "union")]
    #[test]
    fn an_open_body_is_closed_by_its_hull_and_unioned() {
        let (v, mut f) = unit_box(SHIFT);
        f.remove(0);
        let m = concat(&[unit_box(DVec3::ZERO), (v, f)]);
        let (out, r) = prepare_mesh(&m, HULL);
        assert_eq!((r.action.as_str(), r.n_open, r.n_closed), ("unioned", 0, 2), "{r:?}");
        assert!((out.volume() - (2.0 - 0.6 * 0.8 * 0.9)).abs() < 0.01);
        assert!(r.warnings.is_empty());
    }

    #[cfg(feature = "union")]
    #[test]
    fn preparations_are_cached_per_option() {
        let m = two_overlapping_boxes();
        let (union, _) = m.prepared();
        let (hulled, r) = m.prepared_with(HULL);
        assert_eq!(r.action, "unioned");
        assert!(!Arc::ptr_eq(&union, &hulled));
        assert!(Arc::ptr_eq(&hulled, &m.prepared_with(HULL).0));
        assert!(Arc::ptr_eq(&union, &m.prepared_with(MeshPrepOptions::DEFAULT).0));
        assert!(Arc::ptr_eq(&m, &m.prepared_with(MeshPrepOptions::NONE).0));
        // Boxes are their own hulls: the same solid, computed separately.
        assert!((hulled.volume() - union.volume()).abs() < 1e-6);
    }

    #[cfg(not(feature = "union"))]
    #[test]
    fn without_the_union_feature_hulls_are_skipped_with_a_warning() {
        let m = concat(&[unit_box(DVec3::ZERO)]);
        let (out, r) = prepare_mesh(&m, HULL_ONLY);
        assert!(Arc::ptr_eq(&out, &m));
        assert!(!r.convex_hull);
        assert_eq!(r.warnings.len(), 1, "{r:?}");
    }
}
