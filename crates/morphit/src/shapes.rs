//! Small procedural meshes, handy for tests, examples and quick experiments.

use glam::DVec3;

use crate::mesh::Mesh;

/// Axis-aligned box between `min` and `max` with outward-facing triangles.
/// Top and bottom faces are split along the `(min.x, min.y)`-`(max.x, max.y)` diagonal.
pub fn box_mesh(min: DVec3, max: DVec3) -> Mesh {
    let v = |x: bool, y: bool, z: bool| {
        DVec3::new(
            if x { max.x } else { min.x },
            if y { max.y } else { min.y },
            if z { max.z } else { min.z },
        )
    };
    let vertices = vec![
        v(false, false, false),
        v(true, false, false),
        v(true, true, false),
        v(false, true, false),
        v(false, false, true),
        v(true, false, true),
        v(true, true, true),
        v(false, true, true),
    ];
    let faces = vec![
        [0, 2, 1],
        [0, 3, 2],
        [4, 5, 6],
        [4, 6, 7],
        [0, 1, 5],
        [0, 5, 4],
        [2, 3, 7],
        [2, 7, 6],
        [0, 4, 7],
        [0, 7, 3],
        [1, 2, 6],
        [1, 6, 5],
    ];
    Mesh::from_vertices(vertices, faces, None).expect("box is a valid mesh")
}

/// Latitude/longitude sphere with `stacks` rings and `slices` segments.
pub fn uv_sphere(center: DVec3, radius: f64, stacks: u32, slices: u32) -> Mesh {
    assert!(stacks >= 2 && slices >= 3);
    let mut vertices = vec![center + DVec3::new(0.0, 0.0, radius)];
    for i in 1..stacks {
        let theta = std::f64::consts::PI * i as f64 / stacks as f64;
        for j in 0..slices {
            let phi = 2.0 * std::f64::consts::PI * j as f64 / slices as f64;
            vertices.push(
                center + radius * DVec3::new(theta.sin() * phi.cos(), theta.sin() * phi.sin(), theta.cos()),
            );
        }
    }
    let south = vertices.len() as u32;
    vertices.push(center - DVec3::new(0.0, 0.0, radius));
    let ring = |i: u32, j: u32| 1 + (i - 1) * slices + (j % slices);
    let mut faces = Vec::new();
    for j in 0..slices {
        faces.push([0, ring(1, j), ring(1, j + 1)]);
    }
    for i in 1..stacks - 1 {
        for j in 0..slices {
            let (a, b, c, d) = (ring(i, j), ring(i, j + 1), ring(i + 1, j), ring(i + 1, j + 1));
            faces.push([a, c, d]);
            faces.push([a, d, b]);
        }
    }
    for j in 0..slices {
        faces.push([south, ring(stacks - 1, j + 1), ring(stacks - 1, j)]);
    }
    Mesh::from_vertices(vertices, faces, None).expect("sphere is a valid mesh")
}

/// L-shaped prism of height `h`: the polygon (0,0),(2,0),(2,1),(1,1),(1,2),(0,2)
/// extruded along z. The notch `[1,2]x[1,2]` is outside, which makes it a
/// simple concave test shape.
pub fn l_prism(h: f64) -> Mesh {
    let poly = [(0.0, 0.0), (2.0, 0.0), (2.0, 1.0), (1.0, 1.0), (1.0, 2.0), (0.0, 2.0)];
    let n = poly.len() as u32;
    let mut vertices: Vec<DVec3> = poly.iter().map(|&(x, y)| DVec3::new(x, y, 0.0)).collect();
    vertices.extend(poly.iter().map(|&(x, y)| DVec3::new(x, y, h)));
    // Fan from the reflex vertex (1,1) = index 3 keeps every cap triangle inside the L.
    let fan = [(4, 5), (5, 0), (0, 1), (1, 2)];
    let mut faces = Vec::new();
    for &(a, b) in &fan {
        faces.push([3, b, a]); // bottom, facing -z
        faces.push([3 + n, a + n, b + n]); // top, facing +z
    }
    for i in 0..n {
        let j = (i + 1) % n;
        faces.push([i, j, j + n]);
        faces.push([i, j + n, i + n]);
    }
    Mesh::from_vertices(vertices, faces, None).expect("L prism is a valid mesh")
}

/// Regular-ish tetrahedron with vertices at the origin and the unit axes.
pub fn corner_tetrahedron() -> Mesh {
    let vertices = vec![DVec3::ZERO, DVec3::X, DVec3::Y, DVec3::Z];
    let faces = vec![[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]];
    Mesh::from_vertices(vertices, faces, None).expect("tetrahedron is a valid mesh")
}
