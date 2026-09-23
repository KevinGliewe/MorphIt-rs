//! Object URDFs and MJCF models must equal the Python generator's output byte
//! for byte, floating and anchored. The checked-in fixture was generated from
//! the reference code.

use morphit_robot::object_model::{
    ObjectModelOptions, spheres_from_object_urdf, write_object_mjcf, write_object_urdf,
};
use serde_json::Value;

#[test]
fn models_match_python() {
    let path = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/py_object_model.json");
    let Ok(text) = std::fs::read_to_string(path) else {
        eprintln!("skipping: {path} not found (generated from the Python reference)");
        return;
    };
    let cases: Value = serde_json::from_str(&text).unwrap();
    let cases = cases.as_object().unwrap();
    assert!(cases.len() >= 3);
    for (name, c) in cases {
        let f = |v: &Value| v.as_f64().unwrap();
        let centers: Vec<[f64; 3]> =
            c["centers"].as_array().unwrap().iter().map(|p| [f(&p[0]), f(&p[1]), f(&p[2])]).collect();
        let radii: Vec<f64> = c["radii"].as_array().unwrap().iter().map(f).collect();
        let rgba = c["rgba"].as_array().unwrap();
        let want_centroid: Vec<f64> = c["centroid"].as_array().unwrap().iter().map(f).collect();
        for anchored in [false, true] {
            let opts = ObjectModelOptions {
                robot_name: c["robot_name"].as_str().unwrap().into(),
                color_rgba: [f(&rgba[0]), f(&rgba[1]), f(&rgba[2]), f(&rgba[3])],
                anchored,
                ..Default::default()
            };
            let key = if anchored { "_anchored" } else { "" };
            let u = write_object_urdf(&centers, &radii, &opts).unwrap();
            assert_eq!(u.text, c[format!("urdf{key}")].as_str().unwrap(), "{name}: urdf{key}");
            assert_eq!(u.centroid.to_vec(), want_centroid, "{name}");
            let m = write_object_mjcf(&centers, &radii, &opts).unwrap();
            assert_eq!(m.text, c[format!("mjcf{key}")].as_str().unwrap(), "{name}: mjcf{key}");
            assert_eq!(m.centroid, u.centroid);

            let (back, r) = spheres_from_object_urdf(&u.text).unwrap();
            assert_eq!(back.len(), centers.len(), "{name}");
            assert_eq!(r.len(), centers.len(), "{name}");
        }
    }
}
