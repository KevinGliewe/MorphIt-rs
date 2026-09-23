//! Point-in-mesh tests.
//!
//! [`ZRayIntersector`] is an exact port of `inside_mesh.py` + `triangle_hash.pyx`:
//! rescale the mesh into `[0.5, res - 0.5]^3`, bucket triangles by their xy
//! bounding box into a `res x res` grid, and count crossings of a vertical ray
//! through the query point in both directions. A point is inside when both
//! counts are odd.
//!
//! That test misclassifies points whose xy projection lies exactly on a
//! projected triangle edge, which happens for axis-aligned grid points (the
//! Python code swaps in `trimesh.contains` for its voxel-grid init for that
//! reason). [`RobustIntersector`] runs the same machinery on a rotated copy of
//! the mesh so the ray direction is not axis-aligned.

use glam::{DMat3, DQuat, DVec3};

/// Grid resolution used by Python's `check_mesh_contains`.
pub const HASH_RESOLUTION: usize = 512;

/// Direction used by trimesh's `contains` ray test; used by the robust variant.
const ROBUST_RAY: [f64; 3] = [0.4395064455, 0.617598629942, 0.652231566745];

/// Crossing parities of the upward and downward ray through a point.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Parity {
    /// Odd number of crossings in Python's `smaller_depth` set (`depth >= p_z`).
    pub up: bool,
    /// Odd number of crossings in Python's `bigger_depth` set (`depth < p_z`).
    pub down: bool,
}

impl Parity {
    /// Python's verdict: inside only when both parities are odd.
    #[inline]
    pub fn inside(self) -> bool {
        self.up && self.down
    }

    /// Both rays agree (either both odd or both even).
    #[inline]
    pub fn consistent(self) -> bool {
        self.up == self.down
    }
}

/// Exact port of Python's `MeshIntersector`.
#[derive(Debug)]
pub struct ZRayIntersector {
    res: usize,
    scale: DVec3,
    translate: DVec3,
    tris: Vec<[DVec3; 3]>,
    cell_start: Vec<u32>,
    cell_tris: Vec<u32>,
}

impl ZRayIntersector {
    /// Build the spatial hash for a triangle soup.
    pub fn new(triangles: &[[DVec3; 3]], res: usize) -> Self {
        assert!(res >= 2, "resolution must be at least 2");
        let mut lo = DVec3::splat(f64::INFINITY);
        let mut hi = DVec3::splat(f64::NEG_INFINITY);
        for t in triangles {
            for v in t {
                lo = lo.min(*v);
                hi = hi.max(*v);
            }
        }
        let scale = DVec3::splat((res - 1) as f64) / (hi - lo);
        let translate = DVec3::splat(0.5) - scale * lo;
        let tris: Vec<[DVec3; 3]> = triangles.iter().map(|t| t.map(|v| scale * v + translate)).collect();

        let cells = res * res;
        let ranges: Vec<[usize; 4]> = tris.iter().map(|t| cell_range(t, res)).collect();
        let mut counts = vec![0u32; cells + 1];
        for r in &ranges {
            for x in r[0]..=r[1] {
                for y in r[2]..=r[3] {
                    counts[res * x + y + 1] += 1;
                }
            }
        }
        for i in 1..=cells {
            counts[i] += counts[i - 1];
        }
        let cell_start = counts;
        let mut fill = cell_start.clone();
        let mut cell_tris = vec![0u32; cell_start[cells] as usize];
        for (i, r) in ranges.iter().enumerate() {
            for x in r[0]..=r[1] {
                for y in r[2]..=r[3] {
                    let c = res * x + y;
                    cell_tris[fill[c] as usize] = i as u32;
                    fill[c] += 1;
                }
            }
        }
        ZRayIntersector { res, scale, translate, tris, cell_start, cell_tris }
    }

    /// Crossing parities for one point, or `None` when it lies outside the
    /// rescaled bounding box (always outside the mesh).
    pub fn parity(&self, p: DVec3) -> Option<Parity> {
        let q = self.scale * p + self.translate;
        let r = self.res as f64;
        if !(0.0 <= q.x && q.x <= r && 0.0 <= q.y && q.y <= r && 0.0 <= q.z && q.z <= r) {
            return None;
        }
        // Cython: `x = int(points[i, 0])` (truncation).
        let x = q.x as i64;
        let y = q.y as i64;
        let res = self.res as i64;
        if !(0 <= x && x < res && 0 <= y && y < res) {
            return Some(Parity { up: false, down: false });
        }
        let c = (res * x + y) as usize;
        let (a, b) = (self.cell_start[c] as usize, self.cell_start[c + 1] as usize);
        let mut n0 = 0u32;
        let mut n1 = 0u32;
        for &ti in &self.cell_tris[a..b] {
            let t = &self.tris[ti as usize];
            if !inside_triangle_2d(q, t) {
                continue;
            }
            // compute_intersection_depth
            let t1 = t[0];
            let t2 = t[1];
            let t3 = t[2];
            let v1 = t3 - t1;
            let v2 = t2 - t1;
            let n = v1.cross(v2);
            let alpha = n.x * (t1.x - q.x) + n.y * (t1.y - q.y);
            let n_2 = n.z;
            if n_2 == 0.0 {
                continue; // NaN depth in Python: neither comparison holds.
            }
            let abs_n_2 = n_2.abs();
            let s_n_2 = if n_2 > 0.0 { 1.0 } else { -1.0 };
            let depth = t1.z * abs_n_2 + alpha * s_n_2;
            let pz = q.z * abs_n_2;
            if depth >= pz {
                n0 += 1;
            } else if depth < pz {
                n1 += 1;
            }
        }
        Some(Parity { up: n0 % 2 == 1, down: n1 % 2 == 1 })
    }

    /// Python's `check_mesh_contains` verdict for one point.
    #[inline]
    pub fn contains(&self, p: DVec3) -> bool {
        self.parity(p).is_some_and(Parity::inside)
    }
}

/// Triangle xy bounding box in grid cells, as `[x_min, x_max, y_min, y_max]`,
/// using C-style truncation and clamping like `_build_hash`.
fn cell_range(t: &[DVec3; 3], res: usize) -> [usize; 4] {
    let clamp = |v: f64| -> usize { (v as i64).clamp(0, res as i64 - 1) as usize };
    let min_x = t[0].x.min(t[1].x).min(t[2].x);
    let max_x = t[0].x.max(t[1].x).max(t[2].x);
    let min_y = t[0].y.min(t[1].y).min(t[2].y);
    let max_y = t[0].y.max(t[1].y).max(t[2].y);
    [clamp(min_x), clamp(max_x), clamp(min_y), clamp(max_y)]
}

/// `TriangleIntersector2d.check_triangles` for a single point/triangle pair.
#[inline]
fn inside_triangle_2d(p: DVec3, t: &[DVec3; 3]) -> bool {
    // A = [[t0x - t2x, t1x - t2x], [t0y - t2y, t1y - t2y]]
    let a00 = t[0].x - t[2].x;
    let a01 = t[1].x - t[2].x;
    let a10 = t[0].y - t[2].y;
    let a11 = t[1].y - t[2].y;
    let y0 = p.x - t[2].x;
    let y1 = p.y - t[2].y;
    let det = a00 * a11 - a01 * a10;
    if det == 0.0 {
        return false;
    }
    let s = if det > 0.0 { 1.0 } else { -1.0 };
    let abs_det = det.abs();
    let u = (a11 * y0 - a01 * y1) * s;
    let v = (-a10 * y0 + a00 * y1) * s;
    let sum = u + v;
    0.0 < u && u < abs_det && 0.0 < v && v < abs_det && 0.0 < sum && sum < abs_det
}

/// Same machinery as [`ZRayIntersector`], but the ray points along trimesh's
/// default (non axis-aligned) direction. Falls back to the axis-aligned test
/// when the two rotated rays disagree.
#[derive(Debug)]
pub struct RobustIntersector {
    rot: DMat3,
    inner: ZRayIntersector,
}

impl RobustIntersector {
    pub fn new(triangles: &[[DVec3; 3]], res: usize) -> Self {
        let dir = DVec3::from_array(ROBUST_RAY).normalize();
        let rot = DMat3::from_quat(DQuat::from_rotation_arc(dir, DVec3::Z));
        let rotated: Vec<[DVec3; 3]> = triangles.iter().map(|t| t.map(|v| rot * v)).collect();
        RobustIntersector { rot, inner: ZRayIntersector::new(&rotated, res) }
    }

    /// Parity of the rotated rays, or `None` outside the bounding box.
    pub fn parity(&self, p: DVec3) -> Option<Parity> {
        self.inner.parity(self.rot * p)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::shapes;

    fn tri_soup(m: &crate::Mesh) -> Vec<[DVec3; 3]> {
        m.triangles().collect()
    }

    #[test]
    fn cube_points_off_the_diagonals() {
        let cube = shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        let ix = ZRayIntersector::new(&tri_soup(&cube), HASH_RESOLUTION);
        assert!(ix.contains(DVec3::new(0.3, 0.6, 0.5)));
        assert!(ix.contains(DVec3::new(0.9, 0.2, 0.01)));
        assert!(!ix.contains(DVec3::new(1.3, 0.6, 0.5)));
        assert!(!ix.contains(DVec3::new(0.3, 0.6, -0.2)));
        assert!(!ix.contains(DVec3::new(5.0, 5.0, 5.0)));
    }

    #[test]
    fn exact_test_fails_on_projected_diagonal_but_robust_does_not() {
        // The cube's top and bottom faces are split along the (0,0)-(1,1)
        // diagonal, so the xy projection of the center lies on a triangle edge.
        let cube = shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        let soup = tri_soup(&cube);
        let exact = ZRayIntersector::new(&soup, HASH_RESOLUTION);
        let robust = RobustIntersector::new(&soup, HASH_RESOLUTION);
        let center = DVec3::splat(0.5);
        assert!(!exact.contains(center), "documents the Python limitation");
        assert!(robust.parity(center).unwrap().inside());
    }

    #[test]
    fn random_points_agree_with_analytic_sphere() {
        use rand::{Rng, SeedableRng};
        let sphere = shapes::uv_sphere(DVec3::new(0.1, -0.2, 0.3), 0.5, 48, 96);
        let soup = tri_soup(&sphere);
        let exact = ZRayIntersector::new(&soup, HASH_RESOLUTION);
        let robust = RobustIntersector::new(&soup, HASH_RESOLUTION);
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(1);
        let mut disagree = 0;
        for _ in 0..4000 {
            let p = DVec3::new(
                rng.random_range(-0.5..0.7),
                rng.random_range(-0.8..0.4),
                rng.random_range(-0.3..0.9),
            );
            let d = (p - DVec3::new(0.1, -0.2, 0.3)).length();
            // Skip the thin shell where the polygonal sphere and the analytic one differ.
            if (d - 0.5).abs() < 0.01 {
                continue;
            }
            let truth = d < 0.5;
            if exact.contains(p) != truth {
                disagree += 1;
            }
            assert_eq!(robust.parity(p).is_some_and(Parity::inside), truth, "{p:?}");
        }
        assert_eq!(disagree, 0);
    }

    #[test]
    fn concave_notch_is_outside() {
        let l = shapes::l_prism(1.0);
        let soup = tri_soup(&l);
        let exact = ZRayIntersector::new(&soup, HASH_RESOLUTION);
        let robust = RobustIntersector::new(&soup, HASH_RESOLUTION);
        for (p, inside) in [
            (DVec3::new(0.3, 1.6, 0.5), true),
            (DVec3::new(1.6, 0.3, 0.5), true),
            (DVec3::new(1.6, 1.6, 0.5), false),
            (DVec3::new(1.2, 1.9, 0.2), false),
        ] {
            assert_eq!(exact.contains(p), inside, "exact {p:?}");
            assert_eq!(robust.parity(p).is_some_and(Parity::inside), inside, "robust {p:?}");
        }
    }
}
