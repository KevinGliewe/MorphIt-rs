use std::path::{Path, PathBuf};

use assert_cmd::Command;
use morphit::glam::DVec3;
use morphit::{PackResult, shapes};

/// Two unit boxes overlapping in a 0.6 x 0.8 x 0.9 block, as one OBJ with two objects.
fn write_overlapping_boxes(dir: &Path) -> PathBuf {
    let mut s = String::new();
    for (k, o) in [DVec3::ZERO, DVec3::new(0.4, 0.2, 0.1)].into_iter().enumerate() {
        let b = shapes::box_mesh(o, o + DVec3::ONE);
        s += &format!("o box{k}\n");
        for v in b.vertices() {
            s += &format!("v {} {} {}\n", v.x, v.y, v.z);
        }
        for f in b.faces() {
            s += &format!(
                "f {} {} {}\n",
                f[0] + 1 + 8 * k as u32,
                f[1] + 1 + 8 * k as u32,
                f[2] + 1 + 8 * k as u32
            );
        }
    }
    let p = dir.join("boxes.obj");
    std::fs::write(&p, s).unwrap();
    p
}

fn write_cube(dir: &Path) -> PathBuf {
    let cube = shapes::box_mesh(DVec3::ZERO, DVec3::new(1.0, 0.7, 0.5));
    let mut s = String::new();
    for v in cube.vertices() {
        s += &format!("v {} {} {}\n", v.x, v.y, v.z);
    }
    for f in cube.faces() {
        s += &format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1);
    }
    let p = dir.join("cube.obj");
    std::fs::write(&p, s).unwrap();
    p
}

fn morphit() -> Command {
    Command::cargo_bin("morphit").unwrap()
}

#[test]
fn pack_to_file_then_score() {
    let dir = tempfile::tempdir().unwrap();
    let cube = write_cube(dir.path());
    let out = dir.path().join("sub").join("out.json");
    let assert = morphit()
        .args(["pack", cube.to_str().unwrap(), "-n", "6", "-i", "20", "-s", "1", "-q", "-o"])
        .arg(&out)
        .args(["--set", "training.center_lr=0.001"])
        .assert()
        .success();
    assert!(assert.get_output().stdout.is_empty(), "nothing on stdout when writing to a file");
    let r = PackResult::load(&out).unwrap();
    assert!(r.num_spheres >= 1 && r.num_spheres <= 6);
    assert_eq!(r.config.training.center_lr, 0.001);
    assert_eq!(r.config.random_seed, Some(1));

    let m = morphit().args(["metrics", cube.to_str().unwrap(), out.to_str().unwrap()]).assert().success();
    let text = String::from_utf8(m.get_output().stdout.clone()).unwrap();
    assert!(text.contains("r_in") && text.contains("d_avg_mm"));

    let j = morphit()
        .args([
            "metrics",
            cube.to_str().unwrap(),
            out.to_str().unwrap(),
            "--json",
            "--volume-samples",
            "2000",
        ])
        .assert()
        .success();
    let v: serde_json::Value = serde_json::from_slice(&j.get_output().stdout).unwrap();
    assert_eq!(v["actual_n"].as_u64().unwrap() as usize, r.num_spheres);
}

#[test]
fn pack_to_stdout_is_pure_json() {
    let dir = tempfile::tempdir().unwrap();
    let cube = write_cube(dir.path());
    let a =
        morphit().args(["pack", cube.to_str().unwrap(), "-n", "4", "-i", "5", "-s", "2"]).assert().success();
    let r: PackResult = serde_json::from_slice(&a.get_output().stdout).unwrap();
    assert!(r.num_spheres <= 4);
    let stderr = String::from_utf8(a.get_output().stderr.clone()).unwrap();
    assert!(stderr.contains("iter"), "progress goes to stderr");
}

#[test]
fn bad_inputs_fail_with_a_message() {
    let dir = tempfile::tempdir().unwrap();
    let cube = write_cube(dir.path());
    let a = morphit().args(["pack", cube.to_str().unwrap(), "-p", "MorphIt-X"]).assert().failure();
    assert!(String::from_utf8_lossy(&a.get_output().stderr).contains("unknown preset"));
    let a = morphit().args(["pack", cube.to_str().unwrap(), "--set", "bogus.key=1"]).assert().failure();
    assert!(String::from_utf8_lossy(&a.get_output().stderr).contains("unknown section"));
    let a = morphit().args(["pack", "missing.obj"]).assert().failure();
    assert!(String::from_utf8_lossy(&a.get_output().stderr).contains("not found"));
    assert!(a.get_output().stdout.is_empty());
}

#[test]
fn export_writes_urdf_and_mjcf() {
    let dir = tempfile::tempdir().unwrap();
    let cube = write_cube(dir.path());
    let result = dir.path().join("r.json");
    morphit()
        .args(["pack", cube.to_str().unwrap(), "-n", "4", "-i", "5", "-s", "1", "-q", "-o"])
        .arg(&result)
        .assert()
        .success();
    let r = PackResult::load(&result).unwrap();

    // MJCF chosen by the .xml extension; name from the mesh file.
    let xml = dir.path().join("cube.xml");
    morphit().args(["export", result.to_str().unwrap(), "-o"]).arg(&xml).assert().success();
    let text = std::fs::read_to_string(&xml).unwrap();
    assert!(text.starts_with("<?xml version=\"1.0\"?>\n<mujoco model=\"cube\">"), "{text}");
    assert!(text.contains("<freejoint name=\"base_free\"/>"));
    assert_eq!(text.matches("type=\"sphere\"").count(), r.num_spheres);

    // URDF on stdout, anchored, custom name and color.
    let a = morphit()
        .args(["export", result.to_str().unwrap(), "--anchored", "--name", "box", "--color", "#ff0000"])
        .assert()
        .success();
    let urdf = String::from_utf8(a.get_output().stdout.clone()).unwrap();
    assert!(urdf.starts_with("<?xml version=\"1.0\"?>\n<robot name=\"box\">"));
    assert!(urdf.contains("<joint name=\"world_to_base\" type=\"fixed\">"));
    assert!(urdf.contains("rgba=\"1.000000 0.000000 0.000000 1.000000\""));
    assert_eq!(urdf.matches("_fixed\"").count(), r.num_spheres);

    let a = morphit().args(["export", result.to_str().unwrap(), "--color", "blue"]).assert().failure();
    assert!(String::from_utf8_lossy(&a.get_output().stderr).contains("--color must be #rrggbb"));
}

#[test]
fn presets_and_info() {
    let a = morphit().arg("presets").assert().success();
    let text = String::from_utf8(a.get_output().stdout.clone()).unwrap();
    for p in ["MorphIt-V", "MorphIt-S", "MorphIt-B", "MorphIt-Obj", "MorphIt-Obj-mass"] {
        assert!(text.contains(p));
    }
    let dir = tempfile::tempdir().unwrap();
    let cube = write_cube(dir.path());
    let a = morphit().args(["info", cube.to_str().unwrap()]).assert().success();
    let text = String::from_utf8(a.get_output().stdout.clone()).unwrap();
    assert!(text.contains("faces         12"));
    assert!(text.contains("volume        3.5"));
}

#[test]
fn devices_lists_adapters_and_device_flag_is_validated() {
    morphit().args(["devices"]).assert().success();
    let dir = tempfile::tempdir().unwrap();
    let cube = write_cube(dir.path());
    let a = morphit().args(["pack", cube.to_str().unwrap(), "--device", "tpu"]).assert().failure();
    assert!(String::from_utf8_lossy(&a.get_output().stderr).contains("unknown device"));
}

#[test]
fn gpu_and_cpu_results_are_identical() {
    let dir = tempfile::tempdir().unwrap();
    let cube = write_cube(dir.path());
    let run = |device: &str| {
        let out = dir.path().join(format!("{device}.json"));
        let a = morphit()
            .args([
                "pack",
                cube.to_str().unwrap(),
                "-n",
                "12",
                "-i",
                "40",
                "-s",
                "7",
                "-q",
                "--device",
                device,
                "-o",
            ])
            .arg(&out)
            .assert();
        let output = a.get_output().clone();
        if !output.status.success() {
            let err = String::from_utf8_lossy(&output.stderr);
            assert!(device == "gpu" && err.contains("GPU"), "unexpected failure: {err}");
            eprintln!("no GPU; skipping ({})", err.trim());
            return None;
        }
        let mut r = PackResult::load(&out).unwrap();
        r.config.model.device = String::new();
        Some(r)
    };
    let Some(gpu) = run("gpu") else { return };
    let cpu = run("cpu").unwrap();
    assert_eq!(serde_json::to_string(&gpu).unwrap(), serde_json::to_string(&cpu).unwrap());
}

#[test]
fn overlapping_bodies_are_unioned_unless_disabled() {
    // Without the `union` feature the step reports `skipped` instead.
    let union = cfg!(feature = "union");
    let dir = tempfile::tempdir().unwrap();
    let boxes = write_overlapping_boxes(dir.path());
    let out = dir.path().join("u.json");
    let a = morphit()
        .args(["pack", boxes.to_str().unwrap(), "-n", "8", "-i", "20", "-s", "3", "-o"])
        .arg(&out)
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&a.get_output().stderr).into_owned();
    assert_eq!(stderr.contains("mesh prep: 2 overlapping closed bodies unioned"), union, "{stderr}");
    let r = PackResult::load(&out).unwrap();
    assert_eq!(r.mesh_prep.as_ref().unwrap().action, if union { "unioned" } else { "skipped" });
    assert!(r.config.model.union_overlapping_bodies);
    if !union {
        return;
    }

    let off = dir.path().join("raw.json");
    morphit()
        .args(["pack", boxes.to_str().unwrap(), "-n", "8", "-i", "5", "-s", "3", "-q"])
        .args(["--set", "model.union_overlapping_bodies=false", "-o"])
        .arg(&off)
        .assert()
        .success();
    assert_eq!(PackResult::load(&off).unwrap().mesh_prep.unwrap().action, "disabled");

    let info = morphit().args(["info", boxes.to_str().unwrap()]).assert().success();
    let text = String::from_utf8(info.get_output().stdout.clone()).unwrap();
    assert!(text.contains("mesh prep     unioned") && text.contains("overlapping: true"), "{text}");

    let m = morphit().args(["metrics", boxes.to_str().unwrap(), out.to_str().unwrap()]).assert().success();
    let stderr = String::from_utf8_lossy(&m.get_output().stderr).to_string();
    assert!(stderr.contains("scoring the prepared mesh: 2 overlapping closed bodies unioned"), "{stderr}");
    morphit()
        .args(["metrics", boxes.to_str().unwrap(), out.to_str().unwrap(), "--raw-mesh"])
        .assert()
        .success();
}

#[test]
fn convex_hulls_on_request() {
    if !cfg!(feature = "union") {
        return;
    }
    let dir = tempfile::tempdir().unwrap();
    let boxes = write_overlapping_boxes(dir.path());
    let out = dir.path().join("h.json");
    let a = morphit()
        .args(["pack", boxes.to_str().unwrap(), "-n", "8", "-i", "5", "-s", "3"])
        .args(["--set", "model.convex_hull=true", "--set", "model.union_overlapping_bodies=false", "-o"])
        .arg(&out)
        .assert()
        .success();
    let stderr = String::from_utf8_lossy(&a.get_output().stderr).into_owned();
    assert!(stderr.contains("mesh prep: 2 bodies replaced by convex hulls"), "{stderr}");
    let r = PackResult::load(&out).unwrap();
    let prep = r.mesh_prep.as_ref().unwrap();
    assert_eq!((prep.action.as_str(), prep.n_hulled, prep.convex_hull), ("hulled", 2, true));
    assert!(r.config.model.convex_hull);

    let info = morphit().args(["info", boxes.to_str().unwrap(), "--convex-hull"]).assert().success();
    let text = String::from_utf8(info.get_output().stdout.clone()).unwrap();
    assert!(text.contains("mesh prep     unioned (2 overlapping convex hulls unioned)"), "{text}");

    // Metrics score the mesh prepared as the result was packed.
    let m = morphit().args(["metrics", boxes.to_str().unwrap(), out.to_str().unwrap()]).assert().success();
    let stderr = String::from_utf8_lossy(&m.get_output().stderr).to_string();
    assert!(stderr.contains("scoring the prepared mesh: 2 bodies replaced by convex hulls"), "{stderr}");
}

#[test]
fn prepare_writes_the_prepared_mesh() {
    let dir = tempfile::tempdir().unwrap();
    let boxes = write_overlapping_boxes(dir.path());
    let union = cfg!(feature = "union");
    let out = dir.path().join("prepared.obj");
    let a = morphit().args(["prepare", boxes.to_str().unwrap(), "-o"]).arg(&out).assert().success();
    let stderr = String::from_utf8_lossy(&a.get_output().stderr).into_owned();
    let raw = morphit::Mesh::load(&boxes).unwrap();
    let got = morphit::Mesh::load(&out).unwrap();
    if union {
        assert!(stderr.contains("mesh prep: unioned"), "{stderr}");
        assert!((got.volume() - (2.0 - 0.6 * 0.8 * 0.9)).abs() < 0.01, "{}", got.volume());
    } else {
        assert!((got.volume() - raw.volume()).abs() < 1e-9);
    }

    // Raw, as STL; and OBJ on stdout.
    let stl = dir.path().join("raw.stl");
    morphit().args(["prepare", boxes.to_str().unwrap(), "--no-union", "-o"]).arg(&stl).assert().success();
    assert!((morphit::Mesh::load(&stl).unwrap().volume() - raw.volume()).abs() < 1e-6);
    let a = morphit().args(["prepare", boxes.to_str().unwrap(), "--no-union"]).assert().success();
    assert!(String::from_utf8_lossy(&a.get_output().stdout).starts_with("# MorphIt mesh"));
    morphit().args(["prepare", boxes.to_str().unwrap(), "--format", "stl"]).assert().failure();

    if union {
        let hull = dir.path().join("hull.obj");
        morphit()
            .args(["prepare", boxes.to_str().unwrap(), "--convex-hull", "--no-union", "-o"])
            .arg(&hull)
            .assert()
            .success();
        assert_eq!(morphit::Mesh::load(&hull).unwrap().faces().len(), raw.faces().len());
    }
}
