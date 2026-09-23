//! Deterministic parity against the Python implementation.
//!
//! The checked-in fixtures were dumped from the Python reference with every
//! tensor forced to float64. Each test feeds the Rust code the exact Python inputs (samples,
//! initial spheres) and compares outputs. Tests skip with a message when the
//! fixture files are absent.

use std::path::PathBuf;
use std::sync::Arc;

use glam::{DMat3, DVec3};
use morphit::config::LossId;
use morphit::density::DensityController;
use morphit::loss::{PhysicsTargets, Problem, Samples, evaluate};
use morphit::{Config, Device, LossWeights, Mesh, Session, Spheres};
use serde_json::Value;

const FIXTURES: [&str; 3] = ["py_link0_b_n16.json", "py_link0_objmass_n16.json", "py_link0_obj_n16.json"];

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests").join("fixtures")
}

fn load(name: &str) -> Option<Value> {
    let p = fixture_dir().join(name);
    match std::fs::read_to_string(&p) {
        Ok(s) => Some(serde_json::from_str(&s).expect("fixture is valid JSON")),
        Err(_) => {
            eprintln!("skipping: {} not found (generated from the Python reference)", p.display());
            None
        }
    }
}

fn f64s(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn vec3s(v: &Value) -> Vec<DVec3> {
    v.as_array().unwrap().iter().map(|r| DVec3::from_slice(&f64s(r))).collect()
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

fn spheres(state: &Value) -> Spheres {
    Spheres {
        centers: vec3s(&state["centers"]),
        raw_radii: f64s(&state["raw_radii"]),
        raw_masses: state["raw_masses"].as_array().map(|_| f64s(&state["raw_masses"])),
    }
}

fn mesh(fx: &Value) -> Arc<Mesh> {
    let p = fixture_dir().join(fx["mesh_file"].as_str().unwrap());
    Arc::new(Mesh::load(p).unwrap())
}

fn config(fx: &Value) -> Config {
    Config::from_json_str(&fx["config"].to_string()).unwrap()
}

fn samples(fx: &Value) -> Samples {
    let s = &fx["samples"];
    Samples {
        inside: vec3s(&s["inside"]),
        surface: vec3s(&s["surface"]),
        surface_normals: vec3s(&s["surface_normals"]),
    }
}

fn problem(fx: &Value, mesh: &Mesh, cfg: &Config) -> Problem {
    Problem::cpu(samples(fx), PhysicsTargets::from_mesh(mesh, cfg.model.density), cfg.model.density)
}

/// Largest absolute difference relative to the largest reference magnitude.
fn rel_err(a: &[f64], b: &[f64]) -> f64 {
    assert_eq!(a.len(), b.len());
    let scale = b.iter().fold(0.0f64, |m, x| m.max(x.abs())).max(1e-300);
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max) / scale
}

fn flat(v: &[DVec3]) -> Vec<f64> {
    v.iter().flat_map(|c| c.to_array()).collect()
}

fn assert_state(label: &str, got: &Spheres, want: &Spheres, tol: f64) {
    let ec = rel_err(&flat(&got.centers), &flat(&want.centers));
    let er = rel_err(&got.raw_radii, &want.raw_radii);
    assert!(ec < tol, "{label}: centers rel err {ec:e}");
    assert!(er < tol, "{label}: raw radii rel err {er:e}");
    if let (Some(a), Some(b)) = (&got.raw_masses, &want.raw_masses) {
        let em = rel_err(a, b);
        assert!(em < tol, "{label}: raw masses rel err {em:e}");
    }
}

fn loss_id(name: &str) -> LossId {
    LossId::ALL.into_iter().find(|id| id.name() == name).unwrap()
}

#[test]
fn mass_properties_match_trimesh() {
    for name in FIXTURES {
        let Some(fx) = load(name) else { return };
        let m = mesh(&fx);
        let want = &fx["mesh"];
        let rel = |a: f64, b: f64| (a - b).abs() / b.abs();
        assert!(rel(m.volume(), want["volume"].as_f64().unwrap()) < 1e-9);
        assert!(rel(m.scale(), want["scale"].as_f64().unwrap()) < 1e-12);
        let com = DVec3::from_slice(&f64s(&want["center_mass"]));
        assert!((m.center_mass() - com).length() / com.length().max(m.scale()) < 1e-9);
        let rows: Vec<Vec<f64>> = want["moment_inertia"].as_array().unwrap().iter().map(f64s).collect();
        let i_want = DMat3::from_cols(
            DVec3::new(rows[0][0], rows[1][0], rows[2][0]),
            DVec3::new(rows[0][1], rows[1][1], rows[2][1]),
            DVec3::new(rows[0][2], rows[1][2], rows[2][2]),
        );
        let diff = m.moment_inertia() - i_want;
        let norm = |x: DMat3| {
            (x.x_axis.length_squared() + x.y_axis.length_squared() + x.z_axis.length_squared()).sqrt()
        };
        assert!(norm(diff) / norm(i_want) < 1e-9, "inertia rel err {:e}", norm(diff) / norm(i_want));
    }
}

#[test]
fn containment_matches_python() {
    let Some(fx) = load(FIXTURES[0]) else { return };
    let m = mesh(&fx);
    let c = &fx["contains"];
    let pts = vec3s(&c["points"]);
    let want = bools(&c["exact"]);
    let got = m.contains_many(&pts);
    let mismatch = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert!(mismatch * 1000 <= pts.len(), "exact containment: {mismatch} of {} differ", pts.len());

    let grid = vec3s(&c["grid"]);
    let want = bools(&c["trimesh"]);
    let got = m.contains_robust_many(&grid);
    let mismatch = got.iter().zip(&want).filter(|(a, b)| a != b).count();
    assert!(mismatch * 100 <= grid.len(), "robust vs trimesh: {mismatch} of {} differ", grid.len());
    eprintln!("robust vs trimesh.contains on the voxel grid: {mismatch} of {} differ", grid.len());
}

#[test]
fn every_loss_and_gradient_matches() {
    for name in FIXTURES {
        let Some(fx) = load(name) else { return };
        let m = mesh(&fx);
        let cfg = config(&fx);
        let p = problem(&fx, &m, &cfg);
        let sp = spheres(&fx["initial"]);
        for (lname, want) in fx["losses"].as_object().unwrap() {
            let id = loss_id(lname);
            let mut w = LossWeights::default();
            w[id] = 1.0;
            let e = evaluate(&p, &sp, &w);
            let v = e.raw[id.index()];
            let wv = want["value"].as_f64().unwrap();
            let tol = 1e-9 * wv.abs().max(1e-12);
            assert!((v - wv).abs() <= tol, "{name} {lname}: value {v:e} vs {wv:e}");
            // Errors are relative to the largest reference entry, with a floor so
            // that gradients which are zero up to rounding (e.g. the mass loss when
            // the initial masses already sum to the mesh mass) compare absolutely.
            let check = |what: &str, got: &[f64], want: &[f64]| {
                let scale = want.iter().fold(1e-8f64, |m, x| m.max(x.abs()));
                let err = got.iter().zip(want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max) / scale;
                assert!(err < 1e-8, "{name} {lname}: {what} grad rel err {err:e}");
            };
            check("center", &flat(&e.grads.centers), &flat(&vec3s(&want["centers"])));
            check("radius", &e.grads.raw_radii, &f64s(&want["raw_radii"]));
            if let Some(gm) = &e.grads.raw_masses {
                check("mass", gm, &f64s(&want["raw_masses"]));
            }
        }
    }
}

#[test]
fn five_step_trajectory_matches() {
    for name in FIXTURES {
        let Some(fx) = load(name) else { return };
        let m = mesh(&fx);
        let cfg = config(&fx);
        assert!(!cfg.training.density_control_enabled);
        let mut s = Session::from_parts(cfg, m, samples(&fx), spheres(&fx["initial"])).unwrap();
        for (k, want) in fx["trajectory"].as_array().unwrap().iter().enumerate() {
            let info = s.step().unwrap();
            let wl = want["total_loss"].as_f64().unwrap();
            assert!(
                (info.total_loss - wl).abs() <= 1e-9 * wl.abs(),
                "{name} step {k}: loss {} vs {wl}",
                info.total_loss
            );
            assert_state(&format!("{name} step {k}"), s.spheres(), &spheres(want), 1e-8);
        }
    }
}

#[test]
fn density_control_pass_matches() {
    for name in FIXTURES {
        let Some(fx) = load(name) else { return };
        let m = mesh(&fx);
        let mut cfg = config(&fx);
        let p = problem(&fx, &m, &cfg);
        let dc_fx = &fx["density_control"];
        let weights = cfg.training.loss_weights();

        // Scoring, culling, survivor ordering and farthest-point reseeding.
        cfg.training.density_control_warmup_steps = 0;
        let mut sp = spheres(&dc_fx["pre"]);
        let mut dc = DensityController::new(cfg.model.num_spheres, &cfg.training);
        assert_eq!(dc.temperature(), dc_fx["temperature_before"].as_f64().unwrap());
        let out = dc.repack(&p, &m, &mut sp, &cfg.training, &weights);
        assert_eq!(out.added as u64, dc_fx["added"].as_u64().unwrap(), "{name}");
        assert_eq!(out.removed as u64, dc_fx["removed"].as_u64().unwrap(), "{name}");
        assert!((dc.temperature() - dc_fx["temperature_after"].as_f64().unwrap()).abs() < 1e-15);
        assert_state(
            &format!("{name} density control (no warmup)"),
            &sp,
            &spheres(&dc_fx["post_no_warmup"]),
            1e-12,
        );

        // With warmup. Reseeding sizes a new sphere to exactly touch its nearest
        // survivor, so their overlap is zero up to one ulp; last-bit differences
        // between PyTorch's and Rust's exp/log1p decide whether that overlap term
        // is active, and Adam carries the difference through the warmup. The
        // result therefore agrees to ~1e-5 rather than to rounding.
        cfg.training.density_control_warmup_steps = dc_fx["warmup_steps"].as_u64().unwrap() as usize;
        let mut sp = spheres(&dc_fx["pre"]);
        let mut dc = DensityController::new(cfg.model.num_spheres, &cfg.training);
        dc.repack(&p, &m, &mut sp, &cfg.training, &weights);
        assert_state(&format!("{name} density control (warmup)"), &sp, &spheres(&dc_fx["post"]), 1e-4);
    }
}

/// The fixture problem bound to a GPU, or `None` (with a notice) when no GPU is usable.
fn gpu_problem(fx: &Value, mesh: &Mesh, cfg: &Config) -> Option<Problem> {
    let targets = PhysicsTargets::from_mesh(mesh, cfg.model.density);
    match Problem::new(samples(fx), targets, cfg.model.density, Device::Gpu(None), cfg.model.num_spheres) {
        Ok(p) => Some(p),
        Err(e) => {
            eprintln!("skipping GPU parity: {e}");
            None
        }
    }
}

/// On a GPU every loss, gradient, training step and density-control pass is
/// bit-identical to the CPU (and therefore meets the Python tolerances above).
#[test]
fn gpu_backend_is_bit_identical_to_cpu() {
    for name in FIXTURES {
        let Some(fx) = load(name) else { return };
        let m = mesh(&fx);
        let mut cfg = config(&fx);
        let Some(gpu) = gpu_problem(&fx, &m, &cfg) else { return };
        let cpu = problem(&fx, &m, &cfg);
        assert!(matches!(gpu.search.device(), morphit::ResolvedDevice::Gpu { .. }));

        // Every loss alone and the preset's full weight set.
        let sp = spheres(&fx["initial"]);
        let mut sets: Vec<LossWeights> = LossId::ALL
            .into_iter()
            .filter(|id| *id != LossId::Flatness)
            .map(|id| {
                let mut w = LossWeights::default();
                w[id] = 1.0;
                w
            })
            .collect();
        sets.push(cfg.training.loss_weights());
        for w in &sets {
            assert_eq!(evaluate(&gpu, &sp, w), evaluate(&cpu, &sp, w), "{name}");
        }

        // Five training steps.
        let mut on_cpu =
            Session::from_parts(cfg.clone(), m.clone(), samples(&fx), spheres(&fx["initial"])).unwrap();
        cfg.model.device = "gpu".to_string();
        let mut on_gpu =
            Session::from_parts(cfg.clone(), m.clone(), samples(&fx), spheres(&fx["initial"])).unwrap();
        for k in 0..5 {
            let a = on_gpu.step().unwrap();
            let b = on_cpu.step().unwrap();
            assert_eq!(a.total_loss.to_bits(), b.total_loss.to_bits(), "{name} step {k}");
            assert_eq!(on_gpu.spheres(), on_cpu.spheres(), "{name} step {k}");
        }

        // One density-control pass with warmup.
        let dc_fx = &fx["density_control"];
        cfg.training.density_control_warmup_steps = dc_fx["warmup_steps"].as_u64().unwrap() as usize;
        let weights = cfg.training.loss_weights();
        let mut a = spheres(&dc_fx["pre"]);
        let mut b = spheres(&dc_fx["pre"]);
        DensityController::new(cfg.model.num_spheres, &cfg.training).repack(
            &gpu,
            &m,
            &mut a,
            &cfg.training,
            &weights,
        );
        DensityController::new(cfg.model.num_spheres, &cfg.training).repack(
            &cpu,
            &m,
            &mut b,
            &cfg.training,
            &weights,
        );
        assert_eq!(a, b, "{name} density control");
    }
}
