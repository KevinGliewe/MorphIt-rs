//! The JS API under Node.js (CPU):
//! `cargo test -p morphit-wasm --target wasm32-unknown-unknown`.
#![cfg(target_arch = "wasm32")]

use morphit_wasm::{JsConfig, JsMesh, JsRobotPackage, JsSession};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use wasm_bindgen_test::*;

const LINK0: &[u8] = include_bytes!("../../morphit/tests/fixtures/link0.obj");

fn get(v: &JsValue, key: &str) -> JsValue {
    js_sys::Reflect::get(v, &key.into()).unwrap()
}

fn config(n: usize, iterations: usize) -> JsConfig {
    let mut c = JsConfig::from_preset("MorphIt-B").unwrap();
    c.set("model.num_spheres", n.into()).unwrap();
    c.set("model.num_inside_samples", 600.into()).unwrap();
    c.set("model.num_surface_samples", 600.into()).unwrap();
    c.set("training.iterations", iterations.into()).unwrap();
    c.set("random_seed", 5.into()).unwrap();
    c.set("model.device", "cpu".into()).unwrap();
    c
}

/// A plain JS object (not a `Map`) from JSON.
fn js(v: serde_json::Value) -> Result<JsValue, serde_wasm_bindgen::Error> {
    use serde::Serialize as _;
    v.serialize(&serde_wasm_bindgen::Serializer::json_compatible())
}

fn msg(e: JsValue) -> String {
    js_sys::Error::from(e).message().into()
}

#[wasm_bindgen_test]
fn mesh_and_config() {
    let m = JsMesh::from_bytes(LINK0, "obj", Some("link0.obj".into())).unwrap();
    let info = m.info().unwrap();
    assert!(get(&info, "volume").as_f64().unwrap() > 0.0);
    assert_eq!(get(&info, "sourcePath").as_string().as_deref(), Some("link0.obj"));
    assert_eq!(m.faces().len() % 3, 0);
    assert!(JsMesh::supported_extensions().contains(&"stl".to_string()));
    assert!(JsMesh::from_bytes(b"nonsense", "xyz", None).is_err());

    // Mesh preparation options: link0 is one body; its hull is larger.
    let report = m.prep_report(JsValue::UNDEFINED).unwrap();
    assert_eq!(get(&report, "action").as_string().as_deref(), Some("unchanged"));
    let hull_opts = || js(serde_json::json!({ "convexHull": true })).unwrap();
    let report = m.prep_report(hull_opts()).unwrap();
    assert_eq!(get(&report, "action").as_string().as_deref(), Some("hulled"));
    assert_eq!(get(&report, "n_hulled").as_f64(), Some(1.0));
    let hull = m.prepared(hull_opts()).unwrap();
    let volume = |m: &JsMesh| get(&m.info().unwrap(), "volume").as_f64().unwrap();
    assert!(volume(&hull) > volume(&m));

    // The prepared mesh exports and loads back.
    let obj = JsMesh::from_bytes(hull.to_obj().as_bytes(), "obj", None).unwrap();
    assert_eq!(volume(&obj), volume(&hull));
    let stl = JsMesh::from_bytes(&hull.to_stl(), "stl", None).unwrap();
    assert!((volume(&stl) - volume(&hull)).abs() < 1e-6 * volume(&hull));

    let mut c = config(7, 3);
    assert_eq!(c.get("model.num_spheres").unwrap().as_f64(), Some(7.0));
    assert!(JsConfig::presets().contains(&"MorphIt-V".to_string()));
    let e = c.set("model.num_spheres", "many".into()).map_err(JsValue::from).unwrap_err();
    assert!(msg(e).contains("num_spheres"));
    assert!(c.to_json().unwrap().contains("\"num_spheres\":7"));
}

#[wasm_bindgen_test]
async fn async_and_sync_steps_agree() {
    let mesh = JsMesh::from_bytes(LINK0, "obj", Some("link0.obj".into())).unwrap();
    let a = JsSession::new(&config(8, 6), &mesh).unwrap();
    let b = JsSession::new(&config(8, 6), &mesh).unwrap();
    assert_eq!(a.device().unwrap(), "cpu");
    assert!(!a.needs_async().unwrap());
    while !a.is_done() {
        let x = JsFuture::from(a.step()).await.unwrap();
        let y = b.step_sync().unwrap();
        assert_eq!(get(&x, "totalLoss").as_f64(), get(&y, "totalLoss").as_f64());
    }
    assert!(b.is_done());
    // Nothing left to run: resolves to undefined.
    assert!(JsFuture::from(a.step()).await.unwrap().is_undefined());
    assert_eq!(a.centers(), b.centers());
    assert_eq!(a.radii(), b.radii());
    a.finalize().unwrap();
    b.finalize().unwrap();
    assert_eq!(a.state(), "finalized");
    let r = a.result_json().unwrap();
    assert_eq!(r, b.result_json().unwrap());
    let parsed = morphit::PackResult::from_json_str(&r).unwrap();
    assert_eq!(parsed.mesh_path, "link0.obj");
    assert!(a.history_json().unwrap().contains("total_loss"));
}

#[wasm_bindgen_test]
async fn step_many_honours_limits() {
    let mesh = JsMesh::from_bytes(LINK0, "obj", None).unwrap();
    let s = JsSession::new(&config(5, 10), &mesh).unwrap();
    let opts = js_sys::Object::new();
    js_sys::Reflect::set(&opts, &"maxSteps".into(), &4.into()).unwrap();
    let last = JsFuture::from(s.step_many(opts.into())).await.unwrap();
    assert_eq!(get(&last, "iteration").as_f64(), Some(3.0));
    assert_eq!(s.iteration(), 4);
    JsFuture::from(s.step_many(JsValue::UNDEFINED)).await.unwrap();
    assert!(s.is_done());
    assert_eq!(s.iteration(), 10);
}

fn cube_stl() -> Vec<u8> {
    let m = morphit::shapes::box_mesh(morphit::glam::DVec3::ZERO, morphit::glam::DVec3::new(0.1, 0.06, 0.04));
    let mut s = String::from("solid c\n");
    for t in m.triangles() {
        s += "facet normal 0 0 0\nouter loop\n";
        for v in t {
            s += &format!("vertex {} {} {}\n", v.x, v.y, v.z);
        }
        s += "endloop\nendfacet\n";
    }
    (s + "endsolid c\n").into_bytes()
}

const URDF: &str = r#"<?xml version="1.0"?>
<robot name="r">
  <link name="base"><collision><geometry><mesh filename="package://r/meshes/cube.stl"/></geometry></collision></link>
  <link name="arm">
    <collision><origin xyz="0 0 0.2"/><geometry><mesh filename="../meshes/cube.stl"/></geometry></collision>
    <collision><geometry><box size="1 1 1"/></geometry></collision>
  </link>
  <joint name="j" type="revolute"><parent link="base"/><child link="arm"/><origin xyz="0 0 0.5"/></joint>
</robot>"#;

#[wasm_bindgen_test]
async fn robot_package_pipeline() {
    let files = js_sys::Map::new();
    files.set(&"r/urdf/r.urdf".into(), &js_sys::Uint8Array::from(URDF.as_bytes()));
    files.set(&"r/meshes/cube.stl".into(), &js_sys::Uint8Array::from(&cube_stl()[..]));
    let pkg = JsRobotPackage::from_files(files.into()).unwrap();
    assert_eq!(pkg.urdfs(), ["r/urdf/r.urdf"]);
    let report = pkg.inspect(None).unwrap();
    let collisions: js_sys::Array = get(&report, "collisions").into();
    assert_eq!(collisions.length(), 3);
    let poses = pkg.link_poses(None).unwrap();
    let arm: js_sys::Array = get(&get(&poses, "arm"), "translation").into();
    assert_eq!(arm.get(2).as_f64(), Some(0.5));

    let params = js(serde_json::json!({
        "numSpheres": 4, "iterations": 5, "seed": 1, "advanced": { "coverage_weight": 2.0 }
    }))
    .unwrap();
    for i in 0..2 {
        let item = collisions.get(i);
        assert_eq!(get(&item, "action").as_string().as_deref(), Some("pack"));
        let s = pkg.pack_link(item.clone(), params.clone(), Some("cpu".into())).unwrap();
        JsFuture::from(s.step_many(JsValue::UNDEFINED)).await.unwrap();
        s.finalize().unwrap();
        let link = get(&item, "link_name").as_string().unwrap();
        pkg.set_link_result(&link, 0, &s.result_json().unwrap()).unwrap();
        if i == 0 {
            let partial = pkg.assemble(None, None, None).unwrap();
            assert_eq!(
                get(&partial, "skipped").as_string().as_deref(),
                Some("arm[0]: missing JSON: arm_0.json")
            );
        }
    }
    let out = pkg.assemble(None, Some("#ff0000".into()), Some(0.5)).unwrap();
    assert_eq!(get(&out, "skipped").as_string().as_deref(), Some(""));
    assert_eq!(get(&out, "primitiveCollisionsRemoved").as_f64(), Some(1.0));
    let urdf = get(&out, "urdf").as_string().unwrap();
    assert!(urdf.contains("<link name=\"arm_sphere1\">"));
    assert!(urdf.contains("<joint name=\"base_to_base_sphere1\" type=\"fixed\">"));

    let raw =
        js(serde_json::json!({ "numSpheres": 3, "iterations": 1, "unionOverlappingBodies": false })).unwrap();
    let s = pkg.pack_link(collisions.get(0), raw, Some("cpu".into())).unwrap();
    assert!(s.config_json().unwrap().contains("\"union_overlapping_bodies\":false"));
    assert!(s.mesh_prep_json().unwrap().contains("\"action\":\"disabled\""));

    let hull = js(serde_json::json!({ "numSpheres": 3, "iterations": 1, "convexHull": true })).unwrap();
    let s = pkg.pack_link(collisions.get(0), hull, Some("cpu".into())).unwrap();
    assert!(s.config_json().unwrap().contains("\"convex_hull\":true"));
    assert!(s.mesh_prep_json().unwrap().contains("\"convex_hull\":true"));

    let bad = js(serde_json::json!({ "numSpheres": 0 })).unwrap();
    let e = pkg.pack_link(collisions.get(0), bad, None).map_err(JsValue::from).err().unwrap();
    assert_eq!(msg(e), "num_spheres must be in [1, 200]");
    assert!(pkg.pack_link(collisions.get(2), params, None).is_err(), "a box is not packed");
}

#[wasm_bindgen_test]
fn object_models_and_quality() {
    let centers = [0.0, 0.0, 0.0, 0.1, 0.0, 0.0];
    let radii = [0.05, 0.04];
    let urdf = morphit_wasm::object_urdf(&centers, &radii, JsValue::UNDEFINED).unwrap();
    let text = get(&urdf, "text").as_string().unwrap();
    assert!(text.contains("<robot name=\"object\">"));
    let back = morphit_wasm::spheres_from_urdf(&text).unwrap();
    assert_eq!(js_sys::Array::from(&get(&back, "radii")).length(), 2);
    let opts = js(serde_json::json!({ "name": "o", "anchored": true })).unwrap();
    let mjcf = morphit_wasm::object_mjcf(&centers, &radii, opts).unwrap();
    assert!(get(&mjcf, "text").as_string().unwrap().contains("<mujoco model=\"o\">"));
    let bad = js(serde_json::json!({ "color": "blue" })).unwrap();
    assert!(morphit_wasm::object_urdf(&centers, &radii, bad).is_err());

    let mesh = JsMesh::from_bytes(&cube_stl(), "stl", None).unwrap();
    let q = morphit_wasm::link_quality(&mesh, "l", 0, &[0.0; 3], &[0.03]).unwrap();
    assert_eq!(get(&q, "num_spheres").as_f64(), Some(1.0));
    let overall = morphit_wasm::aggregate(js_sys::Array::of1(&q).into()).unwrap();
    assert_eq!(get(&overall, "num_spheres").as_f64(), Some(1.0));
}
