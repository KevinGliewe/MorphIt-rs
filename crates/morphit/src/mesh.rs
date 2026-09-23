//! Immutable triangle mesh with the derived quantities MorphIt needs:
//! face normals and areas, bounds, scale, and trimesh-compatible mass properties.

use std::fmt;
use std::path::Path;
use std::sync::{Arc, OnceLock};

use crate::par::*;
use glam::{DMat3, DVec3};

use crate::contains::{HASH_RESOLUTION, Parity, RobustIntersector, ZRayIntersector};
use crate::error::{Error, Result};
use crate::mesh_io;
use crate::mesh_prep::{MeshPrepReport, prepare_mesh};

/// A closed triangle mesh. Immutable after construction and safe to share
/// between threads (`Send + Sync`); wrap it in an `Arc` to share it between
/// sessions.
pub struct Mesh {
    vertices: Vec<DVec3>,
    faces: Vec<[u32; 3]>,
    face_normals: Vec<DVec3>,
    face_areas: Vec<f64>,
    bounds: (DVec3, DVec3),
    scale: f64,
    volume: f64,
    center_mass: DVec3,
    moment_inertia: DMat3,
    source_path: Option<String>,
    winding_flipped: bool,
    exact: OnceLock<ZRayIntersector>,
    robust: OnceLock<RobustIntersector>,
    /// Cached [`Mesh::prepared`]; `None` means the mesh itself (storing an
    /// `Arc` to `self` here would be a reference cycle).
    prepared: OnceLock<(Option<Arc<Mesh>>, MeshPrepReport)>,
}

impl fmt::Debug for Mesh {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Mesh")
            .field("vertices", &self.vertices.len())
            .field("faces", &self.faces.len())
            .field("volume", &self.volume)
            .field("scale", &self.scale)
            .field("source_path", &self.source_path)
            .finish()
    }
}

impl Mesh {
    /// Load a mesh file, choosing the format by extension (case-insensitive):
    /// `.obj`, `.stl`, `.ply` or `.dae`. Polygons are fan-triangulated and all
    /// objects, groups or scene nodes are concatenated.
    pub fn load(path: impl AsRef<Path>) -> Result<Mesh> {
        let path = path.as_ref();
        if !path.is_file() {
            return Err(Error::Io(format!("mesh file not found: {}", path.display())));
        }
        let ext = path.extension().and_then(|e| e.to_str()).unwrap_or_default();
        let bytes =
            std::fs::read(path).map_err(|e| Error::Io(format!("cannot read {}: {e}", path.display())))?;
        Mesh::load_from_bytes(&bytes, ext, Some(path.display().to_string()))
    }

    /// Parse a mesh held in memory; `ext` names the format like a file
    /// extension (`"obj"`, `".STL"`, ...). `source_path` is recorded in results.
    pub fn load_from_bytes(bytes: &[u8], ext: &str, source_path: Option<String>) -> Result<Mesh> {
        let name = source_path.clone().unwrap_or_else(|| format!("<{} bytes>", bytes.len()));
        let (vertices, faces) = mesh_io::parse(bytes, ext).map_err(|e| match e {
            Some(msg) => Error::Mesh(format!("failed to parse {name}: {msg}")),
            None => Error::Mesh(format!(
                "unsupported mesh format `.{}` ({name}); supported: {}",
                ext.trim_start_matches('.'),
                Mesh::supported_extensions().iter().map(|e| format!(".{e}")).collect::<Vec<_>>().join(", ")
            )),
        })?;
        Mesh::from_vertices(vertices, faces, source_path)
    }

    /// Lower-case extensions (without the dot) that [`Mesh::load`] accepts.
    pub fn supported_extensions() -> &'static [&'static str] {
        mesh_io::SUPPORTED_EXTENSIONS
    }

    /// Build from vertex positions and triangle indices.
    pub fn from_arrays(vertices: &[[f64; 3]], faces: &[[u32; 3]]) -> Result<Mesh> {
        Mesh::from_vertices(vertices.iter().map(|v| DVec3::from_array(*v)).collect(), faces.to_vec(), None)
    }

    /// Build from flat buffers: `xyz` holds 3 coordinates per vertex, `tris` 3 indices per triangle.
    pub fn from_flat(xyz: &[f64], tris: &[u32]) -> Result<Mesh> {
        if xyz.len() % 3 != 0 || tris.len() % 3 != 0 {
            return Err(Error::Mesh("flat buffers must have a multiple of 3 entries".into()));
        }
        let vertices = xyz.chunks_exact(3).map(|c| DVec3::new(c[0], c[1], c[2])).collect();
        let faces = tris.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
        Mesh::from_vertices(vertices, faces, None)
    }

    /// Build from owned vertices and faces. `source_path` is recorded in results.
    ///
    /// Fails when there are no faces, an index is out of range, a coordinate is
    /// not finite, or the mesh encloses no volume. A mesh whose triangles are
    /// consistently wound inward is flipped (with a warning) so the volume is positive.
    pub fn from_vertices(
        vertices: Vec<DVec3>,
        mut faces: Vec<[u32; 3]>,
        source_path: Option<String>,
    ) -> Result<Mesh> {
        if faces.is_empty() {
            return Err(Error::Mesh("mesh has no faces".into()));
        }
        let nv = vertices.len();
        if let Some(bad) = faces.iter().flatten().find(|&&i| i as usize >= nv) {
            return Err(Error::Mesh(format!("face index {bad} out of range ({nv} vertices)")));
        }
        if vertices.iter().any(|v| !v.is_finite()) {
            return Err(Error::Mesh("mesh has non-finite vertex coordinates".into()));
        }

        // Bounds over referenced vertices only (trimesh behaviour).
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for &i in faces.iter().flatten() {
            let v = vertices[i as usize];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let extents = hi - lo;
        let scale = extents.length();
        if extents.min_element() <= 1e-12 * scale {
            return Err(Error::Mesh(format!("mesh is flat (extents {extents:?}); it must enclose a volume")));
        }

        let (mut volume, mut center_mass, mut moment_inertia) = mass_properties(&vertices, &faces);
        let bbox_volume = extents.x * extents.y * extents.z;
        if !volume.is_finite() || volume.abs() <= 1e-9 * bbox_volume {
            return Err(Error::Mesh(format!(
                "mesh encloses no volume (signed volume {volume:e}); it must be closed"
            )));
        }
        let mut winding_flipped = false;
        if volume < 0.0 {
            tracing::warn!("mesh triangles are wound inward (negative volume); flipping all faces");
            for f in &mut faces {
                f.swap(1, 2);
            }
            (volume, center_mass, moment_inertia) = mass_properties(&vertices, &faces);
            winding_flipped = true;
        }

        let mut face_normals = Vec::with_capacity(faces.len());
        let mut face_areas = Vec::with_capacity(faces.len());
        for f in &faces {
            let [a, b, c] = f.map(|i| vertices[i as usize]);
            let n = (b - a).cross(c - a);
            let len = n.length();
            face_areas.push(0.5 * len);
            // trimesh pads degenerate faces with a zero normal.
            face_normals.push(if len > 0.0 { n / len } else { DVec3::ZERO });
        }

        Ok(Mesh {
            vertices,
            faces,
            face_normals,
            face_areas,
            bounds: (lo, hi),
            scale,
            volume,
            center_mass,
            moment_inertia,
            source_path,
            winding_flipped,
            exact: OnceLock::new(),
            robust: OnceLock::new(),
            prepared: OnceLock::new(),
        })
    }

    pub fn vertices(&self) -> &[DVec3] {
        &self.vertices
    }

    pub fn faces(&self) -> &[[u32; 3]] {
        &self.faces
    }

    /// Unit outward normal per face (zero for degenerate faces).
    pub fn face_normals(&self) -> &[DVec3] {
        &self.face_normals
    }

    pub fn face_areas(&self) -> &[f64] {
        &self.face_areas
    }

    /// Total surface area.
    pub fn area(&self) -> f64 {
        self.face_areas.iter().sum()
    }

    /// Axis-aligned bounds `(min, max)` of the referenced vertices.
    pub fn bounds(&self) -> (DVec3, DVec3) {
        self.bounds
    }

    pub fn extents(&self) -> DVec3 {
        self.bounds.1 - self.bounds.0
    }

    /// Length of the bounding-box diagonal (trimesh's `scale`).
    pub fn scale(&self) -> f64 {
        self.scale
    }

    /// Enclosed volume (positive).
    pub fn volume(&self) -> f64 {
        self.volume
    }

    /// Center of mass of the solid at uniform density.
    pub fn center_mass(&self) -> DVec3 {
        self.center_mass
    }

    /// Inertia tensor about the center of mass at density 1 (trimesh's `moment_inertia`).
    pub fn moment_inertia(&self) -> DMat3 {
        self.moment_inertia
    }

    /// File the mesh was loaded from, if any.
    pub fn source_path(&self) -> Option<&str> {
        self.source_path.as_deref()
    }

    /// True if the input was wound inward and all faces were flipped on load.
    pub fn winding_flipped(&self) -> bool {
        self.winding_flipped
    }

    /// Iterate over the triangles as vertex triples.
    pub fn triangles(&self) -> impl Iterator<Item = [DVec3; 3]> + '_ {
        self.faces.iter().map(|f| f.map(|i| self.vertices[i as usize]))
    }

    fn exact_intersector(&self) -> &ZRayIntersector {
        self.exact.get_or_init(|| {
            let soup: Vec<[DVec3; 3]> = self.triangles().collect();
            ZRayIntersector::new(&soup, HASH_RESOLUTION)
        })
    }

    fn robust_intersector(&self) -> &RobustIntersector {
        self.robust.get_or_init(|| {
            let soup: Vec<[DVec3; 3]> = self.triangles().collect();
            RobustIntersector::new(&soup, HASH_RESOLUTION)
        })
    }

    /// The mesh to pack: overlapping closed bodies unioned (see
    /// [`crate::mesh_prep`]), or this mesh itself. Computed once and shared by
    /// every session on this mesh.
    pub fn prepared(self: &Arc<Self>) -> (Arc<Mesh>, MeshPrepReport) {
        let (mesh, report) = self.prepared.get_or_init(|| {
            let (mesh, report) = prepare_mesh(self, true);
            ((!Arc::ptr_eq(&mesh, self)).then_some(mesh), report)
        });
        (mesh.clone().unwrap_or_else(|| Arc::clone(self)), report.clone())
    }

    /// Python's `check_mesh_contains` for one point (axis-aligned ray parity).
    pub fn contains(&self, p: DVec3) -> bool {
        self.exact_intersector().contains(p)
    }

    /// Python's `check_mesh_contains` for many points, computed in parallel.
    pub fn contains_many(&self, points: &[DVec3]) -> Vec<bool> {
        let ix = self.exact_intersector();
        points.par_iter().map(|&p| ix.contains(p)).collect()
    }

    /// Containment with a non axis-aligned ray, robust for grid-aligned points.
    /// Falls back to the axis-aligned test when the rotated rays disagree.
    pub fn contains_robust(&self, p: DVec3) -> bool {
        match self.robust_intersector().parity(p) {
            None => false,
            Some(par) if par.consistent() => par.up,
            Some(_) => self.exact_intersector().parity(p).is_some_and(Parity::inside),
        }
    }

    /// Parallel [`Mesh::contains_robust`].
    pub fn contains_robust_many(&self, points: &[DVec3]) -> Vec<bool> {
        points.par_iter().map(|&p| self.contains_robust(p)).collect()
    }
}

/// Volume, center of mass and inertia about the center of mass (density 1),
/// following `trimesh.triangles.mass_properties` (Eberly's polyhedral integrals).
pub(crate) fn mass_properties(vertices: &[DVec3], faces: &[[u32; 3]]) -> (f64, DVec3, DMat3) {
    let mut integral = [0.0f64; 10];
    for f in faces {
        let [p0, p1, p2] = f.map(|i| vertices[i as usize]);
        // trimesh.triangles.cross: np.cross of the two consecutive edge vectors.
        let d = (p1 - p0).cross(p2 - p1);
        let a0 = p0.to_array();
        let a1 = p1.to_array();
        let a2 = p2.to_array();
        let dd = d.to_array();
        let mut f1 = [0.0; 3];
        let mut f2 = [0.0; 3];
        let mut f3 = [0.0; 3];
        let mut g0 = [0.0; 3];
        let mut g1 = [0.0; 3];
        let mut g2 = [0.0; 3];
        for k in 0..3 {
            let (x0, x1, x2) = (a0[k], a1[k], a2[k]);
            f1[k] = x0 + x1 + x2;
            f2[k] = x0 * x0 + x1 * x1 + x0 * x1 + x2 * f1[k];
            f3[k] = x0 * x0 * x0 + x0 * x0 * x1 + x0 * x1 * x1 + x1 * x1 * x1 + x2 * f2[k];
            g0[k] = f2[k] + x0 * (f1[k] + x0);
            g1[k] = f2[k] + x1 * (f1[k] + x1);
            g2[k] = f2[k] + x2 * (f1[k] + x2);
        }
        integral[0] += dd[0] * f1[0];
        for k in 0..3 {
            integral[1 + k] += dd[k] * f2[k];
            integral[4 + k] += dd[k] * f3[k];
        }
        for i in 0..3 {
            let ti = (i + 1) % 3;
            integral[7 + i] += dd[i] * (a0[ti] * g0[i] + a1[ti] * g1[i] + a2[ti] * g2[i]);
        }
    }
    let coeff = [6.0, 24.0, 24.0, 24.0, 60.0, 60.0, 60.0, 120.0, 120.0, 120.0];
    for (v, c) in integral.iter_mut().zip(coeff) {
        *v *= 1.0 / c;
    }
    let volume = integral[0];
    let cm = DVec3::new(integral[1], integral[2], integral[3]) / volume;
    let ixx = integral[5] + integral[6] - volume * (cm.y * cm.y + cm.z * cm.z);
    let iyy = integral[4] + integral[6] - volume * (cm.x * cm.x + cm.z * cm.z);
    let izz = integral[4] + integral[5] - volume * (cm.x * cm.x + cm.y * cm.y);
    let ixy = -(integral[7] - volume * (cm.x * cm.y));
    let iyz = -(integral[8] - volume * (cm.y * cm.z));
    let ixz = -(integral[9] - volume * (cm.x * cm.z));
    let inertia =
        DMat3::from_cols(DVec3::new(ixx, ixy, ixz), DVec3::new(ixy, iyy, iyz), DVec3::new(ixz, iyz, izz));
    (volume, cm, inertia)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes;

    fn close(a: f64, b: f64, tol: f64) -> bool {
        (a - b).abs() <= tol * b.abs().max(1.0)
    }

    #[test]
    fn unit_cube_properties() {
        let m = shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        assert!(close(m.volume(), 1.0, 1e-12));
        assert!((m.center_mass() - DVec3::splat(0.5)).length() < 1e-12);
        let i = m.moment_inertia();
        for r in 0..3 {
            for c in 0..3 {
                let expect = if r == c { 1.0 / 6.0 } else { 0.0 };
                assert!((i.col(c)[r] - expect).abs() < 1e-12, "I[{r}][{c}]");
            }
        }
        assert!(close(m.scale(), 3f64.sqrt(), 1e-12));
        assert!(close(m.area(), 6.0, 1e-12));
        for (n, a) in m.face_normals().iter().zip(m.face_areas()) {
            assert!(close(n.length(), 1.0, 1e-12));
            assert!(close(*a, 0.5, 1e-12));
        }
        // Outward: normal points away from the centroid.
        for (t, n) in m.triangles().zip(m.face_normals()) {
            let c = (t[0] + t[1] + t[2]) / 3.0;
            assert!((c - DVec3::splat(0.5)).dot(*n) > 0.0);
        }
    }

    #[test]
    fn translated_scaled_box() {
        let m = shapes::box_mesh(DVec3::new(1.0, 2.0, 3.0), DVec3::new(3.0, 3.0, 7.0));
        let (a, b, c) = (2.0, 1.0, 4.0);
        let vol = a * b * c;
        assert!(close(m.volume(), vol, 1e-12));
        assert!((m.center_mass() - DVec3::new(2.0, 2.5, 5.0)).length() < 1e-12);
        let i = m.moment_inertia();
        assert!(close(i.col(0).x, vol * (b * b + c * c) / 12.0, 1e-12));
        assert!(close(i.col(1).y, vol * (a * a + c * c) / 12.0, 1e-12));
        assert!(close(i.col(2).z, vol * (a * a + b * b) / 12.0, 1e-12));
        assert!(i.col(0).y.abs() < 1e-12 && i.col(0).z.abs() < 1e-12 && i.col(1).z.abs() < 1e-12);
    }

    #[test]
    fn tetrahedron_closed_form() {
        let m = shapes::corner_tetrahedron();
        assert!(close(m.volume(), 1.0 / 6.0, 1e-12));
        assert!((m.center_mass() - DVec3::splat(0.25)).length() < 1e-12);
        // Corner tetrahedron about its centroid: diag = 1/80, off-diag = +1/480.
        let i = m.moment_inertia();
        assert!(close(i.col(0).x, 1.0 / 80.0, 1e-10));
        assert!(close(i.col(0).y, 1.0 / 480.0, 1e-10));
    }

    #[test]
    fn uv_sphere_converges_to_ball() {
        let r = 0.7;
        let m = shapes::uv_sphere(DVec3::new(0.3, 0.1, -0.2), r, 96, 192);
        let vol = 4.0 / 3.0 * std::f64::consts::PI * r * r * r;
        assert!((m.volume() - vol).abs() / vol < 0.005);
        let i_expect = 0.4 * vol * r * r;
        let i = m.moment_inertia();
        for k in 0..3 {
            assert!((i.col(k)[k] - i_expect).abs() / i_expect < 0.01);
        }
        assert!((m.center_mass() - DVec3::new(0.3, 0.1, -0.2)).length() < 1e-9);
    }

    #[test]
    fn inverted_mesh_is_flipped() {
        let cube = shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        let inverted: Vec<[u32; 3]> = cube.faces().iter().map(|f| [f[0], f[2], f[1]]).collect();
        let m = Mesh::from_vertices(cube.vertices().to_vec(), inverted, None).unwrap();
        assert!(m.winding_flipped());
        assert!(close(m.volume(), 1.0, 1e-12));
        assert_eq!(m.faces(), cube.faces());
    }

    #[test]
    fn degenerate_inputs_are_rejected() {
        assert!(Mesh::from_vertices(vec![DVec3::ZERO], vec![], None).is_err());
        assert!(Mesh::from_vertices(vec![DVec3::ZERO; 3], vec![[0, 1, 5]], None).is_err());
        // Open surface: a single square has zero volume.
        let flat = Mesh::from_arrays(
            &[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]],
            &[[0, 1, 2], [0, 2, 3]],
        );
        assert!(flat.is_err());
        let nan = Mesh::from_arrays(
            &[[f64::NAN, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            &[[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]],
        );
        assert!(nan.is_err());
    }

    #[test]
    fn obj_and_stl_agree() {
        let dir = tempfile::tempdir().unwrap();
        let cube = shapes::box_mesh(DVec3::new(-1.0, 0.0, 2.0), DVec3::new(0.5, 2.0, 2.5));
        let obj = dir.path().join("c.obj");
        let mut s = String::new();
        for v in cube.vertices() {
            s += &format!("v {} {} {}\n", v.x, v.y, v.z);
        }
        for f in cube.faces() {
            s += &format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1);
        }
        std::fs::write(&obj, s).unwrap();
        let stl = dir.path().join("c.STL");
        let tris: Vec<stl_io::Triangle> = cube
            .triangles()
            .zip(cube.face_normals())
            .map(|(t, n)| stl_io::Triangle {
                normal: stl_io::Vector::new([n.x as f32, n.y as f32, n.z as f32]),
                vertices: t.map(|v| stl_io::Vector::new([v.x as f32, v.y as f32, v.z as f32])),
            })
            .collect();
        {
            let mut f = std::fs::File::create(&stl).unwrap();
            stl_io::write_stl(&mut f, tris.iter()).unwrap();
        }
        let a = Mesh::load(&obj).unwrap();
        let b = Mesh::load(&stl).unwrap();
        assert!(close(a.volume(), cube.volume(), 1e-12));
        assert!(close(b.volume(), cube.volume(), 1e-6));
        assert!((a.center_mass() - b.center_mass()).length() < 1e-6);
        assert_eq!(a.source_path(), Some(obj.display().to_string().as_str()));
    }

    #[test]
    fn obj_quads_are_triangulated() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("q.obj");
        std::fs::write(
            &p,
            "v 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n\
             f 1 4 3 2\nf 5 6 7 8\nf 1 2 6 5\nf 3 4 8 7\nf 1 5 8 4\nf 2 3 7 6\n",
        )
        .unwrap();
        let m = Mesh::load(&p).unwrap();
        assert_eq!(m.faces().len(), 12);
        assert!(close(m.volume(), 1.0, 1e-12));
    }

    #[test]
    fn link0_fixture_loads() {
        let p = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/link0.obj");
        let m = Mesh::load(p).unwrap();
        assert!(m.faces().len() > 100);
        assert!(m.volume() > 0.0);
        assert!(m.contains(m.center_mass()) || m.contains_robust(m.center_mass()));
    }

    #[test]
    fn unsupported_and_missing_files() {
        assert!(matches!(Mesh::load("nope.obj"), Err(Error::Io(_))));
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("x.ply");
        std::fs::write(&p, "ply").unwrap();
        assert!(matches!(Mesh::load(&p), Err(Error::Mesh(_))));
    }

    #[test]
    fn mesh_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<Mesh>();
    }
}
