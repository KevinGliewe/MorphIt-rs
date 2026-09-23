//! GPU backend tests. They need an adapter; software adapters (WARP,
//! lavapipe) are accepted so the tests also run on GPU-less CI. Without any
//! adapter every test prints a notice and passes.

use std::sync::atomic::Ordering;

use glam::DVec3;
use rand::{Rng, SeedableRng};

use super::{SessionSearch, context, list_devices, select};
use crate::device::Device;
use crate::distances;
use crate::loss::Samples;
use crate::search::tests::random_points;
use crate::search::{Wants, cpu_boundary_pairs};

fn gpu(samples: &Samples) -> Option<SessionSearch> {
    let work = usize::MAX;
    match select(Device::Gpu(None), work, true) {
        Ok(Some(index)) => Some(SessionSearch::new(context(index).unwrap(), samples)),
        _ => {
            eprintln!("no GPU adapter available; skipping");
            None
        }
    }
}

fn spheres(n: usize, seed: u64, extent: f64) -> (Vec<DVec3>, Vec<f64>) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(seed);
    let centers = random_points(n, seed + 100, extent);
    let radii = (0..n).map(|_| rng.random_range(0.02..0.3) * extent).collect();
    (centers, radii)
}

fn check_equal(g: &SessionSearch, s: &Samples, centers: &[DVec3], radii: &[f64]) {
    let want = Wants { inside_min: true, surface_min: true, pairs: true };
    let r = g.queries(s, centers, radii, want);
    assert_eq!(r.inside_min, distances::row_minima(&s.inside, centers, radii), "inside minima");
    assert_eq!(r.surface_min, distances::row_minima(&s.surface, centers, radii), "surface minima");
    // The GPU list is a superset with a tolerance band; the exact set must be inside it,
    // in the same per-sphere ascending order.
    let exact = cpu_boundary_pairs(&s.surface, centers, radii);
    for j in 0..centers.len() {
        let got = r.pairs.of(j);
        assert!(got.windows(2).all(|w| w[0] < w[1]), "sphere {j}: candidates not strictly ascending");
        for i in exact.of(j) {
            assert!(got.contains(i), "sphere {j}: exact pair {i} missing from the GPU candidates");
        }
    }
    assert_eq!(
        g.nearest_sample_per_center(&s.surface, centers),
        distances::nearest_sample_per_center(centers, &s.surface)
    );
    assert_eq!(
        g.stats.cpu_calls.load(Ordering::Relaxed),
        0,
        "the GPU path was not used: {:?}",
        g.last_error()
    );
}

#[test]
fn enumeration_does_not_panic() {
    let list = list_devices();
    for d in &list {
        assert!(!d.name.is_empty());
        eprintln!("gpu:{} {} {} {}", d.index, d.name, d.backend, d.kind);
    }
    assert!(select(Device::Cpu, usize::MAX, true).unwrap().is_none());
    assert!(select(Device::Auto, 0, true).unwrap().is_none(), "auto must stay on the CPU for tiny problems");
    assert!(select(Device::Gpu(Some(list.len())), 0, true).is_err());
}

#[test]
fn matches_cpu_on_random_problems() {
    for (n, m) in [(1usize, 10usize), (5, 10), (5, 3000), (64, 5000), (300, 5000), (64, 20000)] {
        let s = Samples {
            inside: random_points(m, 1, 1.0),
            surface: random_points(m, 2, 1.0),
            ..Default::default()
        };
        let Some(g) = gpu(&s) else { return };
        let (c, r) = spheres(n, n as u64, 1.0);
        check_equal(&g, &s, &c, &r);
    }
}

#[test]
fn matches_cpu_at_odd_scales_and_offsets() {
    for (extent, shift) in [(1e-3, 0.0), (1e3, 0.0), (1.0, 500.0), (0.3, -12.5)] {
        let off = DVec3::splat(shift);
        let mut s = Samples {
            inside: random_points(2000, 3, extent),
            surface: random_points(2000, 4, extent),
            ..Default::default()
        };
        for p in s.inside.iter_mut().chain(&mut s.surface) {
            *p += off;
        }
        let Some(g) = gpu(&s) else { return };
        let (mut c, r) = spheres(40, 7, extent);
        for p in &mut c {
            *p += off;
        }
        check_equal(&g, &s, &c, &r);
    }
}

#[test]
fn exact_ties_and_near_ties_take_the_cpu_answer() {
    // Duplicate spheres (exact ties on every row), samples exactly on sphere
    // surfaces (boundary candidates with r - d == 0), and two spheres whose
    // gaps differ by far less than f32 can resolve.
    let mut s = Samples {
        inside: random_points(1000, 5, 1.0),
        surface: random_points(1000, 6, 1.0),
        ..Default::default()
    };
    let c0 = DVec3::new(0.5, 0.5, 0.5);
    for k in 0..200 {
        let dir = random_points(1, 1000 + k, 1.0)[0] - DVec3::splat(0.5);
        s.surface[k as usize] = c0 + dir.normalize() * 0.25;
    }
    let Some(g) = gpu(&s) else { return };
    let centers =
        vec![c0, c0, DVec3::new(0.2, 0.2, 0.2), DVec3::new(0.2, 0.2, 0.2 + 1e-12), DVec3::new(0.8, 0.1, 0.4)];
    let radii = vec![0.25, 0.25, 0.1, 0.1, 0.15];
    check_equal(&g, &s, &centers, &radii);
    assert!(g.stats.rescanned.load(Ordering::Relaxed) > 0, "ties must go through the rescan path");
}

#[test]
fn non_finite_input_falls_back_to_the_cpu() {
    let s = Samples {
        inside: random_points(100, 8, 1.0),
        surface: random_points(100, 9, 1.0),
        ..Default::default()
    };
    let Some(g) = gpu(&s) else { return };
    let (mut c, r) = spheres(4, 1, 1.0);
    c[2].y = f64::NAN;
    let want = Wants { inside_min: true, surface_min: true, pairs: true };
    let got = g.queries(&s, &c, &r, want);
    let cpu = distances::row_minima(&s.inside, &c, &r);
    assert_eq!(got.inside_min.len(), cpu.len());
    for (a, b) in got.inside_min.iter().zip(&cpu) {
        assert_eq!(a.j, b.j);
        assert!(a.gap.is_nan() == b.gap.is_nan() && (a.gap.is_nan() || a.gap == b.gap));
    }
    assert_eq!(g.stats.cpu_calls.load(Ordering::Relaxed), 1);
    assert!(g.gpu_ok_for_test(), "a non-finite input must not poison the session");
}

#[test]
fn pair_buffer_regrows() {
    let s = Samples { surface: random_points(3000, 10, 1.0), ..Default::default() };
    let Some(g) = gpu(&s) else { return };
    g.set_pair_capacity(1);
    let (c, r) = spheres(30, 11, 1.0);
    let got = g.boundary_pairs(&s.surface, &c, &r);
    let exact = cpu_boundary_pairs(&s.surface, &c, &r);
    assert!(exact.len() > 100, "test needs many pairs");
    for j in 0..30 {
        for i in exact.of(j) {
            assert!(got.of(j).contains(i));
        }
    }
    assert!(g.stats.pair_reruns.load(Ordering::Relaxed) >= 1);
}

#[test]
fn rescan_rate_is_small_on_a_realistic_problem() {
    let mesh = crate::shapes::box_mesh(DVec3::ZERO, DVec3::new(0.3, 0.2, 0.1));
    let mut rng = crate::sampling::MorphRng::seed_from_u64(0);
    let inside = crate::sampling::sample_inside(&mesh, 20000, &mut rng).unwrap();
    let (surface, _) = crate::sampling::sample_surface(&mesh, 20000, &mut rng);
    let s = Samples { inside, surface, ..Default::default() };
    let Some(g) = gpu(&s) else { return };
    let mut centers = random_points(256, 12, 0.1);
    for c in &mut centers {
        *c += DVec3::new(0.0, 0.0, 0.0);
    }
    let radii = vec![0.02; 256];
    check_equal(&g, &s, &centers, &radii);
    let fast = g.stats.fast.load(Ordering::Relaxed) as f64;
    let rescanned = g.stats.rescanned.load(Ordering::Relaxed) as f64;
    let rate = rescanned / (fast + rescanned);
    eprintln!("rescan rate {rate:.5}");
    assert!(rate < 0.01, "rescan rate {rate}");
}

#[test]
fn error_bound_constant_has_slack() {
    // Emulate the kernel arithmetic in f32 on the CPU and measure the actual
    // error relative to u (B + R); the kernel bound assumes < 62 u.
    let u = (f32::EPSILON / 2.0) as f64;
    let s = random_points(20000, 13, 1.0);
    let (c, r) = spheres(64, 14, 1.0);
    let b = 1.0;
    let rmax = r.iter().cloned().fold(0.0, f64::max);
    let mut worst: f64 = 0.0;
    for &p in &s {
        for (&cc, &rr) in c.iter().zip(&r) {
            let d32 = {
                let a = [p.x as f32, p.y as f32, p.z as f32];
                let q = [cc.x as f32, cc.y as f32, cc.z as f32];
                let dx = a[0] - q[0];
                let dy = a[1] - q[1];
                let dz = a[2] - q[2];
                (dx * dx + dy * dy + dz * dz).sqrt() - rr as f32
            };
            let d64 = distances::dist(p, cc) - rr;
            worst = worst.max((d32 as f64 - d64).abs() / (u * (b + rmax)));
        }
    }
    eprintln!("worst observed single-gap error: {worst:.2} u (B + R)");
    assert!(worst < 31.0);
}
