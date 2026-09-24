//! End-to-end tests of the HTTP API: the router is served on a local port and
//! driven over HTTP and a websocket, with the bundled `web/` directory and
//! the CPU device (tiny iteration counts keep packs fast).

use std::path::{Path, PathBuf};
use std::time::Duration;

use futures_util::StreamExt;
use morphit_robot::object_model::{extract_centroid, spheres_from_object_urdf};
use morphit_server::{AppState, ServerConfig, router};
use reqwest::StatusCode;
use reqwest::multipart::{Form, Part};
use serde_json::Value;

struct Server {
    base: String,
    _sessions: tempfile::TempDir,
    http: reqwest::Client,
}

fn web_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web")
}

async fn start(ttl: Duration) -> Server {
    let sessions = tempfile::tempdir().unwrap();
    let mut config = ServerConfig::new(web_dir());
    config.device = "cpu".into();
    config.session_root = sessions.path().to_path_buf();
    config.session_ttl = ttl;
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let app = router(AppState::new(config));
    tokio::spawn(async move { axum::serve(listener, app).await.unwrap() });
    Server { base: format!("http://{addr}"), _sessions: sessions, http: reqwest::Client::new() }
}

impl Server {
    fn url(&self, path: &str) -> String {
        format!("{}{path}", self.base)
    }

    async fn get(&self, path: &str) -> reqwest::Response {
        self.http.get(self.url(path)).send().await.unwrap()
    }

    async fn post(&self, path: &str, form: Form) -> reqwest::Response {
        self.http.post(self.url(path)).multipart(form).send().await.unwrap()
    }

    async fn json(&self, path: &str) -> Value {
        let r = self.get(path).await;
        assert_eq!(r.status(), StatusCode::OK, "{path}");
        r.json().await.unwrap()
    }
}

fn header(r: &reqwest::Response, name: &str) -> String {
    r.headers().get(name).map(|v| v.to_str().unwrap().to_string()).unwrap_or_default()
}

async fn detail(r: reqwest::Response) -> Value {
    r.json::<Value>().await.unwrap()["detail"].clone()
}

fn link0() -> Vec<u8> {
    std::fs::read(web_dir().join("examples/link0.obj")).unwrap()
}

fn mesh_part(data: Vec<u8>, name: &str) -> Part {
    Part::bytes(data).file_name(name.to_string())
}

#[tokio::test(flavor = "multi_thread")]
async fn static_routes_and_example_library() {
    let s = start(Duration::from_secs(3600)).await;
    assert_eq!(s.get("/healthz").await.text().await.unwrap(), r#"{"ok":true}"#);
    let r = s.get("/").await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(header(&r, "cache-control"), "no-store");
    assert!(header(&r, "content-type").starts_with("text/html"));

    let list = s.json("/api/examples").await;
    let list = list.as_array().unwrap();
    assert!(list.len() >= 15);
    assert_eq!(list[0]["name"], "bunny");
    assert_eq!(list[0]["label"], "bunny.obj — Stanford Bunny");
    assert_eq!(list[0]["default"], true);
    let keys: Vec<&String> = list[0].as_object().unwrap().keys().collect();
    assert_eq!(keys.len(), 6);
    let link0 = list.iter().find(|e| e["name"] == "link0").unwrap();
    assert_eq!((link0["has_packed"].as_bool(), link0["has_thumbnail"].as_bool()), (Some(false), Some(true)));

    let r = s.get("/api/example/bunny").await;
    assert_eq!(header(&r, "content-type"), "model/obj");
    assert_eq!(header(&r, "content-disposition"), r#"attachment; filename="bunny.obj""#);
    assert!(r.bytes().await.unwrap().len() > 1000);

    let r = s.get("/api/example/bunny/packed").await;
    assert_eq!(header(&r, "content-type"), "application/xml");
    let centroid = header(&r, "x-morphit-centroid");
    let text = r.text().await.unwrap();
    let c = extract_centroid(&text).unwrap();
    assert_eq!(centroid, format!("{:.6},{:.6},{:.6}", c[0], c[1], c[2]));

    let r = s.get("/api/example/bunny/thumbnail").await;
    assert_eq!(header(&r, "content-type"), "image/png");
    assert_eq!(header(&r, "cache-control"), "public, max-age=86400");

    let r = s.get("/api/example/nope").await;
    assert_eq!(r.status(), StatusCode::NOT_FOUND);
    assert_eq!(detail(r).await, "example 'nope' not found");
    let r = s.get("/api/example/link0/packed").await;
    assert_eq!(detail(r).await, "example 'link0' has no pre-baked packing");

    let robots = s.json("/api/robot/examples").await;
    let robots = robots.as_array().unwrap();
    assert_eq!(robots.len(), 10);
    assert_eq!(robots[0]["name"], "kinova");
    assert_eq!(robots[0]["default"], true);
    assert!(robots.iter().all(|r| r["has_spherical"] == true));
    let r = s.get("/api/robot/example/kinova/spherical").await;
    assert_eq!(header(&r, "content-type"), "application/xml");
    let r = s.get("/api/robot/example/nope/spherical").await;
    assert_eq!(detail(r).await, "robot 'nope' not in library");
}

#[tokio::test(flavor = "multi_thread")]
async fn object_mode_pack_and_analyze() {
    let s = start(Duration::from_secs(3600)).await;
    let form = || {
        Form::new()
            .part("mesh", mesh_part(link0(), "link0.obj"))
            .text("num_spheres", "8")
            .text("iterations", "5")
            .text("seed", "1")
            .text("advanced", r#"{"coverage_weight": 1200, "unknown_key": 1}"#)
            .text("base_color", "#ff0000")
    };
    let r = s.post("/api/morph", form()).await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(header(&r, "content-type"), "application/xml");
    let centroid = header(&r, "x-morphit-centroid");
    let loss: Value = serde_json::from_str(&header(&r, "x-morphit-loss")).unwrap();
    let prep: Value = serde_json::from_str(&header(&r, "x-morphit-mesh-prep")).unwrap();
    let urdf = r.text().await.unwrap();
    assert_eq!(loss.as_array().unwrap().len(), 5);
    assert_eq!(loss[0][0], 0);
    assert_eq!(prep["action"], "unchanged");
    assert!(urdf.starts_with("<?xml version=\"1.0\"?>\n<robot name=\"link0\">\n  <!-- morphit:centroid "));
    assert!(urdf.contains(r#"<color rgba="1.000000 0.000000 0.000000 1.000000"/>"#));
    let c = extract_centroid(&urdf).unwrap();
    assert_eq!(centroid, format!("{:.6},{:.6},{:.6}", c[0], c[1], c[2]));
    let (centers, _) = spheres_from_object_urdf(&urdf).unwrap();
    assert!((1..=8).contains(&centers.len()));

    // Same seed, same device: same URDF.
    let again = s.post("/api/morph", form()).await.text().await.unwrap();
    assert_eq!(again, urdf);

    let r = s
        .post(
            "/api/morph/analyze",
            Form::new().part("mesh", mesh_part(link0(), "link0.obj")).text("urdf", urdf.clone()),
        )
        .await;
    assert_eq!(r.status(), StatusCode::OK);
    let a: Value = r.json().await.unwrap();
    let row = &a["links"][0];
    assert_eq!(row["link_name"], "link0");
    assert_eq!(row["collision_index"], 0);
    assert_eq!(row["num_spheres"].as_u64().unwrap() as usize, centers.len());
    let cov = row["coverage"].as_f64().unwrap();
    assert!(cov > 0.0 && cov <= 1.0);
    assert!(row["d_max"].as_f64().unwrap() >= row["d_mean"].as_f64().unwrap());
    assert_eq!(row["spheres"]["radii"].as_array().unwrap().len(), centers.len());
    assert_eq!(a["overall"].as_object().unwrap().len(), 5);
    assert_eq!(a["overall"]["coverage"], row["coverage"]);

    let r = s
        .post(
            "/api/morph/analyze",
            Form::new().part("mesh", mesh_part(link0(), "link0.obj")).text("urdf", "<robot/>"),
        )
        .await;
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    assert_eq!(detail(r).await, "URDF has no morphit:centroid comment; not a MorphIt object URDF");
}

/// `union_overlapping_bodies` switches mesh preparation off (default on).
#[tokio::test(flavor = "multi_thread")]
async fn mesh_prep_can_be_disabled() {
    let s = start(Duration::from_secs(3600)).await;
    let form = |union: Option<&'static str>| {
        let f = Form::new()
            .part("mesh", mesh_part(link0(), "link0.obj"))
            .text("num_spheres", "6")
            .text("iterations", "3")
            .text("seed", "1");
        match union {
            Some(v) => f.text("union_overlapping_bodies", v),
            None => f,
        }
    };
    let prep =
        |r: &reqwest::Response| -> Value { serde_json::from_str(&header(r, "x-morphit-mesh-prep")).unwrap() };
    let r = s.post("/api/morph", form(None)).await;
    assert_eq!(prep(&r)["action"], "unchanged");
    let r = s.post("/api/morph", form(Some("true"))).await;
    assert_eq!(prep(&r)["action"], "unchanged");
    let r = s.post("/api/morph", form(Some("false"))).await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(prep(&r)["action"], "disabled");
    let urdf = r.text().await.unwrap();

    let r = s.post("/api/morph", form(Some("maybe"))).await;
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
    let d = detail(r).await;
    assert_eq!(d[0]["loc"], serde_json::json!(["body", "union_overlapping_bodies"]));
    assert_eq!(d[0]["type"], "bool_parsing");

    for union in ["0", "1"] {
        let analyze = Form::new()
            .part("mesh", mesh_part(link0(), "link0.obj"))
            .text("urdf", urdf.clone())
            .text("union_overlapping_bodies", union);
        let r = s.post("/api/morph/analyze", analyze).await;
        assert_eq!(r.status(), StatusCode::OK, "{union}");
    }

    // Robot links remember the choice; analyze scores the same mesh.
    let report = load_example(&s, "kinova").await;
    let sid = report["session_id"].as_str().unwrap().to_string();
    let r = s
        .post(
            "/api/robot/pack-link",
            pack_form(&sid, "m1n4s200_link_base", 0, 4, 2).text("union_overlapping_bodies", "false"),
        )
        .await;
    assert_eq!(r.status(), StatusCode::OK);
    let out: Value = r.json().await.unwrap();
    let saved = morphit::PackResult::load(out["json_path"].as_str().unwrap()).unwrap();
    assert!(!saved.config.model.union_overlapping_bodies);
    assert_eq!(saved.mesh_prep.unwrap().action, "disabled");
    let r = s.post("/api/robot/analyze", Form::new().text("session_id", sid)).await;
    assert_eq!(r.status(), StatusCode::OK);
}

/// `convex_hull` replaces each body with its convex hull (default off).
#[tokio::test(flavor = "multi_thread")]
async fn convex_hull_can_be_enabled() {
    let s = start(Duration::from_secs(3600)).await;
    let form = |hull: &'static str| {
        Form::new()
            .part("mesh", mesh_part(link0(), "link0.obj"))
            .text("num_spheres", "6")
            .text("iterations", "3")
            .text("seed", "1")
            .text("convex_hull", hull)
    };
    let prep =
        |r: &reqwest::Response| -> Value { serde_json::from_str(&header(r, "x-morphit-mesh-prep")).unwrap() };
    let r = s.post("/api/morph", form("false")).await;
    assert_eq!(prep(&r)["action"], "unchanged");
    assert_eq!(prep(&r)["convex_hull"], false);
    let r = s.post("/api/morph", form("true")).await;
    assert_eq!(r.status(), StatusCode::OK);
    let p = prep(&r);
    assert_eq!((p["action"].as_str(), p["convex_hull"].as_bool()), (Some("hulled"), Some(true)), "{p}");
    assert!(p["volume_after"].as_f64().unwrap() > p["volume_before"].as_f64().unwrap());
    let urdf = r.text().await.unwrap();

    let r = s.post("/api/morph", form("sometimes")).await;
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(detail(r).await[0]["loc"], serde_json::json!(["body", "convex_hull"]));

    let analyze = Form::new()
        .part("mesh", mesh_part(link0(), "link0.obj"))
        .text("urdf", urdf)
        .text("convex_hull", "true");
    assert_eq!(s.post("/api/morph/analyze", analyze).await.status(), StatusCode::OK);

    // Robot links remember the choice.
    let report = load_example(&s, "kinova").await;
    let sid = report["session_id"].as_str().unwrap().to_string();
    let r = s
        .post(
            "/api/robot/pack-link",
            pack_form(&sid, "m1n4s200_link_base", 0, 4, 2).text("convex_hull", "on"),
        )
        .await;
    assert_eq!(r.status(), StatusCode::OK);
    let out: Value = r.json().await.unwrap();
    let saved = morphit::PackResult::load(out["json_path"].as_str().unwrap()).unwrap();
    assert!(saved.config.model.convex_hull);
    assert!(saved.mesh_prep.unwrap().convex_hull);
    let r = s.post("/api/robot/analyze", Form::new().text("session_id", sid)).await;
    assert_eq!(r.status(), StatusCode::OK);
}

#[tokio::test(flavor = "multi_thread")]
async fn object_mode_rejects_bad_input() {
    let s = start(Duration::from_secs(3600)).await;
    let mesh = || Form::new().part("mesh", mesh_part(link0(), "link0.obj"));
    let check = async |form: Form, status: StatusCode, want: Value| {
        let r = s.post("/api/morph", form).await;
        assert_eq!(r.status(), status);
        let d = detail(r).await;
        if want.is_string() {
            assert_eq!(d, want);
        } else {
            assert_eq!(d[0]["type"], want["type"]);
            assert_eq!(d[0]["loc"], want["loc"]);
        }
    };
    let bad = StatusCode::BAD_REQUEST;
    check(
        mesh().text("variant", "MorphIt-Obj"),
        bad,
        "variant must be one of ('MorphIt-V', 'MorphIt-S', 'MorphIt-B')".into(),
    )
    .await;
    check(mesh().text("num_spheres", "0"), bad, "num_spheres must be in [1, 200]".into()).await;
    check(mesh().text("iterations", "1001"), bad, "iterations must be in [1, 1000]".into()).await;
    check(
        Form::new().part("mesh", mesh_part(link0(), "link0.dae")),
        bad,
        "mesh extension must be one of ('.obj', '.stl', '.ply')".into(),
    )
    .await;
    check(mesh().text("advanced", "[1]"), bad, "advanced must be a JSON object".into()).await;
    check(mesh().text("seed", "-1"), bad, "seed must be a non-negative integer, got -1".into()).await;
    let unprocessable = StatusCode::UNPROCESSABLE_ENTITY;
    check(
        mesh().text("num_spheres", "abc"),
        unprocessable,
        serde_json::json!({"type": "int_parsing", "loc": ["body", "num_spheres"]}),
    )
    .await;
    check(
        Form::new().text("num_spheres", "3"),
        unprocessable,
        serde_json::json!({"type": "missing", "loc": ["body", "mesh"]}),
    )
    .await;
    let r = s.post("/api/morph", Form::new().part("mesh", mesh_part(b"not a mesh".to_vec(), "x.obj"))).await;
    assert_eq!(r.status(), bad);

    let huge = vec![b' '; 100 * 1024 * 1024 + 1];
    let r = s.post("/api/morph", Form::new().part("mesh", mesh_part(huge, "big.obj"))).await;
    assert_eq!(r.status(), StatusCode::PAYLOAD_TOO_LARGE);
    let d = detail(r).await;
    assert!(
        d.as_str().unwrap().starts_with("Mesh file too large: 100.0 MB exceeds the 100 MB limit."),
        "{d}"
    );
}

async fn load_example(s: &Server, name: &str) -> Value {
    let r = s.http.post(s.url(&format!("/api/robot/example/{name}"))).send().await.unwrap();
    assert_eq!(r.status(), StatusCode::OK);
    r.json().await.unwrap()
}

fn pack_form(sid: &str, link: &str, idx: u64, spheres: u32, iterations: u32) -> Form {
    Form::new()
        .text("session_id", sid.to_string())
        .text("link_name", link.to_string())
        .text("collision_index", idx.to_string())
        .text("num_spheres", spheres.to_string())
        .text("iterations", iterations.to_string())
        .text("seed", "0")
}

#[tokio::test(flavor = "multi_thread")]
async fn robot_mode_full_pipeline() {
    let s = start(Duration::from_secs(3600)).await;
    let report = load_example(&s, "kinova").await;
    let sid = report["session_id"].as_str().unwrap().to_string();
    assert_eq!(sid.len(), 32);
    let packs: Vec<(String, u64)> = report["collisions"]
        .as_array()
        .unwrap()
        .iter()
        .filter(|c| c["action"] == "pack")
        .map(|c| (c["link_name"].as_str().unwrap().to_string(), c["collision_index"].as_u64().unwrap()))
        .collect();
    assert_eq!(packs.len(), 9);

    // Files of the session: package URIs, relative paths, sandbox.
    let q = |p: &str| format!("/api/robot/file?session_id={sid}&path={p}");
    let r = s.get(&q("package://kinova_description/meshes/base.stl")).await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(header(&r, "content-type"), "model/stl");
    let urdf_rel = "kinova_description/urdf/m1n4s200_standalone.urdf";
    assert_eq!(s.get(&q(urdf_rel)).await.status(), StatusCode::OK);
    assert_eq!(detail(s.get(&q("../x")).await).await, "Bad upload path: '../x'");
    assert_eq!(s.get(&q("package://nope/x.stl")).await.status(), StatusCode::NOT_FOUND);
    let outside = std::fs::canonicalize(web_dir().join("index.html")).unwrap();
    let r = s.get(&q(&outside.display().to_string())).await;
    assert_eq!(r.status(), StatusCode::FORBIDDEN);
    assert_eq!(detail(r).await, "path escapes session sandbox");
    let r = s.get("/api/robot/file?session_id=nope&path=x").await;
    assert_eq!(detail(r).await, "session 'nope' not found or expired");
    let r = s.get("/api/robot/file?path=x").await;
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);

    // Mesh statistics are computed once and cached.
    let stats = s.json(&format!("/api/robot/mesh-stats?session_id={sid}")).await;
    let items = stats["items"].as_array().unwrap();
    assert_eq!(items.len(), 9);
    assert!(items.iter().all(|i| i["area"].as_f64().unwrap() > 0.0 && i["volume"].as_f64().unwrap() > 0.0));
    assert_eq!(s.json(&format!("/api/robot/mesh-stats?session_id={sid}")).await, stats);

    // Pack one link; assembling now reports the others as missing.
    let (link, idx) = &packs[0];
    let r = s.post("/api/robot/pack-link", pack_form(&sid, link, *idx, 3, 4)).await;
    assert_eq!(r.status(), StatusCode::OK);
    let out: Value = r.json().await.unwrap();
    assert_eq!(out["link_name"], link.as_str());
    assert_eq!(out["num_spheres"], 3);
    assert_eq!(out["loss_history"].as_array().unwrap().len(), 4);
    assert!(Path::new(out["json_path"].as_str().unwrap()).is_file());
    let r = s.post("/api/robot/assemble", Form::new().text("session_id", sid.clone())).await;
    assert_eq!(r.status(), StatusCode::BAD_REQUEST);
    let d = detail(r).await;
    assert!(
        d.as_str()
            .unwrap()
            .starts_with(&format!("Some collisions weren't packed: {}[0]: missing JSON: ", packs[1].0))
    );
    // Analyze works on whatever is packed.
    let r = s.post("/api/robot/analyze", Form::new().text("session_id", sid.clone())).await;
    assert_eq!(r.json::<Value>().await.unwrap()["links"].as_array().unwrap().len(), 1);

    // Errors of pack-link.
    let r = s.post("/api/robot/pack-link", pack_form(&sid, "nope", 0, 3, 2)).await;
    assert_eq!(detail(r).await, "no collision item: nope[0]");
    let r = s.post("/api/robot/pack-link", pack_form(&sid, link, *idx, 300, 2)).await;
    assert_eq!(detail(r).await, "num_spheres must be in [1, 200]");

    for (link, idx) in &packs[1..] {
        let r = s.post("/api/robot/pack-link", pack_form(&sid, link, *idx, 3, 2)).await;
        assert_eq!(r.status(), StatusCode::OK, "{link}");
    }
    let r = s
        .post(
            "/api/robot/assemble",
            Form::new()
                .text("session_id", sid.clone())
                .text("base_color", "#00ff00")
                .text("color_variation", "0.5"),
        )
        .await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(header(&r, "content-type"), "application/xml");
    let urdf = r.text().await.unwrap();
    assert!(urdf.contains(&format!("<link name=\"{}_sphere1\">", packs[0].0)));
    let (mesh_collisions, sphere_links) = count_collisions(&urdf);
    assert_eq!(mesh_collisions, 0, "no mesh collisions remain");
    assert!(sphere_links >= 9);
    let r = s
        .post("/api/robot/assemble", Form::new().text("session_id", sid.clone()).text("color_variation", "2"))
        .await;
    assert_eq!(detail(r).await, "color_variation must be in [0, 1]");

    let r = s.post("/api/robot/analyze", Form::new().text("session_id", sid.clone())).await;
    let a: Value = r.json().await.unwrap();
    assert_eq!(a["links"].as_array().unwrap().len(), 9);
    assert_eq!(
        a["overall"]["num_spheres"].as_u64().unwrap(),
        a["links"].as_array().unwrap().iter().map(|l| l["num_spheres"].as_u64().unwrap()).sum::<u64>()
    );
}

/// (mesh collisions, sphere links) in a URDF.
fn count_collisions(urdf: &str) -> (usize, usize) {
    let mut mesh_collisions = 0;
    for part in urdf.split("<collision").skip(1) {
        let body = part.split("</collision>").next().unwrap_or("");
        mesh_collisions += usize::from(body.contains("<mesh"));
    }
    (mesh_collisions, urdf.matches("_sphere1\">").count())
}

#[tokio::test(flavor = "multi_thread")]
async fn robot_upload_inspection() {
    let s = start(Duration::from_secs(3600)).await;
    let root = web_dir().join("examples/ur5");
    let mut form = Form::new().text("urdf", "ur5_gripper.urdf");
    for f in morphit_robot::paths::walk_files(&root) {
        let rel = f.strip_prefix(root.parent().unwrap()).unwrap().to_string_lossy().replace('\\', "/");
        form = form.part("files", mesh_part(std::fs::read(&f).unwrap(), &rel));
    }
    let r = s.post("/api/robot/inspect", form).await;
    assert_eq!(r.status(), StatusCode::OK);
    let report: Value = r.json().await.unwrap();
    let count =
        |a: &str| report["collisions"].as_array().unwrap().iter().filter(|c| c["action"] == a).count();
    assert_eq!((count("pack"), count("remove-primitive"), count("error")), (7, 4, 0));
    assert!(report["urdf_path"].as_str().unwrap().ends_with("ur5_gripper.urdf"));

    let r = s
        .post("/api/robot/inspect", Form::new().part("files", mesh_part(b"x".to_vec(), "../evil.urdf")))
        .await;
    assert_eq!(detail(r).await, "Bad upload path: '../evil.urdf'");
    let r = s
        .post("/api/robot/inspect", Form::new().part("files", mesh_part(b"x".to_vec(), "a/readme.txt")))
        .await;
    assert_eq!(detail(r).await, "No .urdf files found in folder.");
    let r = s.post("/api/robot/inspect", Form::new().text("urdf", "x.urdf")).await;
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test(flavor = "multi_thread")]
async fn live_frames_on_the_websocket() {
    let s = start(Duration::from_secs(3600)).await;
    let report = load_example(&s, "kinova").await;
    let sid = report["session_id"].as_str().unwrap().to_string();
    let ws_url = format!("{}/api/robot/pack-live?session_id={sid}", s.base.replace("http://", "ws://"));
    let (mut ws, _) = tokio_tungstenite::connect_async(&ws_url).await.unwrap();

    let link = report["collisions"][0]["link_name"].as_str().unwrap().to_string();
    let pack = s.post("/api/robot/pack-link", pack_form(&sid, &link, 0, 4, 40));
    let (packed, frame) = tokio::join!(pack, async {
        tokio::time::timeout(Duration::from_secs(60), ws.next())
            .await
            .expect("a live frame")
            .unwrap()
            .unwrap()
    });
    assert_eq!(packed.status(), StatusCode::OK);
    let frame: Value = serde_json::from_str(frame.to_text().unwrap()).unwrap();
    let keys: Vec<&str> = frame.as_object().unwrap().keys().map(String::as_str).collect();
    let mut want =
        ["centers", "collision_index", "iteration", "link_name", "loss", "radii", "total_iterations"];
    want.sort();
    assert_eq!(keys, want);
    assert_eq!(frame["link_name"], link.as_str());
    assert_eq!(frame["total_iterations"], 40);
    assert_eq!(frame["centers"].as_array().unwrap().len(), frame["radii"].as_array().unwrap().len());

    // Unknown sessions are accepted and closed right away.
    let bad = format!("{}/api/robot/pack-live?session_id=nope", s.base.replace("http://", "ws://"));
    let (mut ws, _) = tokio_tungstenite::connect_async(&bad).await.unwrap();
    let next = tokio::time::timeout(Duration::from_secs(5), ws.next()).await.expect("closes promptly");
    assert!(matches!(
        next,
        None | Some(Ok(tokio_tungstenite::tungstenite::Message::Close(_))) | Some(Err(_))
    ));
}

#[tokio::test(flavor = "multi_thread")]
async fn idle_sessions_expire() {
    let s = start(Duration::ZERO).await;
    let report = load_example(&s, "ur5").await;
    let sid = report["session_id"].as_str().unwrap();
    tokio::time::sleep(Duration::from_millis(20)).await;
    let r = s.get(&format!("/api/robot/mesh-stats?session_id={sid}")).await;
    assert_eq!(r.status(), StatusCode::NOT_FOUND);
    assert_eq!(detail(r).await, format!("session '{sid}' not found or expired"));
}

/// `POST /api/mesh/prepare` and `GET /api/robot/prepared-mesh` return the
/// prepared mesh as OBJ or STL, with the report in `X-Morphit-Mesh-Prep`.
#[tokio::test(flavor = "multi_thread")]
async fn prepared_mesh_downloads() {
    let s = start(Duration::from_secs(3600)).await;
    let raw = morphit::Mesh::load_from_bytes(&link0(), "obj", None).unwrap();
    let form = |fields: &[(&'static str, &'static str)]| {
        fields
            .iter()
            .fold(Form::new().part("mesh", mesh_part(link0(), "link0.obj")), |f, (k, v)| f.text(*k, *v))
    };

    let r = s.post("/api/mesh/prepare", form(&[("convex_hull", "true")])).await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(header(&r, "content-type"), "model/obj");
    assert!(header(&r, "content-disposition").contains("filename=\"link0_prepared.obj\""));
    let prep: Value = serde_json::from_str(&header(&r, "x-morphit-mesh-prep")).unwrap();
    assert_eq!(prep["action"], "hulled");
    let hull = morphit::Mesh::load_from_bytes(&r.bytes().await.unwrap(), "obj", None).unwrap();
    assert!(hull.volume() > raw.volume());
    assert_eq!(hull.volume(), prep["volume_after"].as_f64().unwrap());

    let r = s.post("/api/mesh/prepare", form(&[("format", "STL")])).await;
    assert_eq!(r.status(), StatusCode::OK);
    assert_eq!(header(&r, "content-type"), "model/stl");
    let bytes = r.bytes().await.unwrap();
    assert_eq!(bytes.len(), 84 + 50 * raw.faces().len());

    let r = s.post("/api/mesh/prepare", form(&[("format", "ply")])).await;
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(detail(r).await[0]["loc"], serde_json::json!(["body", "format"]));
    let r = s.post("/api/mesh/prepare", Form::new().text("format", "obj")).await;
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);

    // A collision mesh of a robot session.
    let report = load_example(&s, "kinova").await;
    let sid = report["session_id"].as_str().unwrap().to_string();
    let url = |extra: &str| {
        format!(
            "/api/robot/prepared-mesh?session_id={sid}&link_name=m1n4s200_link_base&collision_index=0{extra}"
        )
    };
    let r = s.get(&url("")).await;
    assert_eq!(r.status(), StatusCode::OK);
    assert!(header(&r, "content-disposition").contains("m1n4s200_link_base_0_prepared.obj"));
    let base = morphit::Mesh::load_from_bytes(&r.bytes().await.unwrap(), "obj", None).unwrap();
    let r = s.get(&url("&convex_hull=1&format=stl")).await;
    assert_eq!(r.status(), StatusCode::OK);
    let prep: Value = serde_json::from_str(&header(&r, "x-morphit-mesh-prep")).unwrap();
    assert!(prep["convex_hull"].as_bool().unwrap());
    let hull = morphit::Mesh::load_from_bytes(&r.bytes().await.unwrap(), "stl", None).unwrap();
    assert!(hull.volume() >= base.volume() * (1.0 - 1e-6));

    let r = s.get(&url("&convex_hull=maybe")).await;
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
    assert_eq!(detail(r).await[0]["loc"], serde_json::json!(["query", "convex_hull"]));
    let r =
        s.get(&format!("/api/robot/prepared-mesh?session_id={sid}&link_name=nope&collision_index=0")).await;
    assert_eq!(r.status(), StatusCode::NOT_FOUND);
    let r = s.get(&format!("/api/robot/prepared-mesh?session_id={sid}&link_name=x&collision_index=a")).await;
    assert_eq!(r.status(), StatusCode::UNPROCESSABLE_ENTITY);
}
