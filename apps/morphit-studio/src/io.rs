//! Opening meshes and robot packages (file dialogs, dropped files, the
//! bundled examples) and saving results. Natively files come from the disk;
//! in the browser from `<input type=file>` pickers and `fetch`, and saves
//! become downloads.

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use morphit::Mesh;
use morphit_robot::examples::{EXAMPLE_OBJECTS, EXAMPLE_ROBOTS, ExampleObject, ExampleRobot};
use morphit_robot::inspect::{Action, InspectionReport, inspect_urdf_in, select_urdf_in};
use morphit_robot::kinematics::{Pose, zero_config_link_poses};
use morphit_robot::vfs::MemPackage;

use crate::task::{Slot, Yielder, spawn};

include!(concat!(env!("OUT_DIR"), "/examples_manifest.rs"));

/// A robot package ready to show and pack.
pub struct RobotDoc {
    pub name: String,
    pub pkg: Arc<MemPackage>,
    pub urdfs: Vec<String>,
    pub urdf: String,
    pub report: InspectionReport,
    pub poses: BTreeMap<String, Pose>,
    /// Per collision: the loaded mesh of `pack` items.
    pub meshes: Vec<Option<Arc<Mesh>>>,
    /// Problems loading meshes.
    pub mesh_errors: Vec<String>,
}

/// What a load produced.
pub enum Loaded {
    Object {
        name: String,
        mesh: Box<Mesh>,
    },
    Robot(Box<RobotDoc>),
    Error(String),
    /// The user closed the dialog.
    Nothing,
}

fn ext_of(name: &str) -> String {
    Path::new(name).extension().map(|e| e.to_string_lossy().to_lowercase()).unwrap_or_default()
}

fn stem_of(name: &str) -> String {
    Path::new(name).file_stem().map(|s| s.to_string_lossy().into_owned()).unwrap_or_else(|| "object".into())
}

/// Parse a mesh file.
pub fn object_from_bytes(name: &str, bytes: &[u8]) -> Loaded {
    match Mesh::load_from_bytes(bytes, &ext_of(name), Some(name.to_string())) {
        Ok(mesh) => Loaded::Object { name: stem_of(name), mesh: Box::new(mesh) },
        Err(e) => Loaded::Error(format!("{name}: {e}")),
    }
}

/// Inspect a package and load its pack meshes. `urdf` picks the URDF by
/// basename (otherwise the only one, or the first of several).
pub async fn robot_doc(name: String, pkg: MemPackage, urdf: Option<&str>, y: &mut Yielder) -> Loaded {
    let urdfs = pkg.urdfs();
    let chosen = match select_urdf_in(&pkg, urdf) {
        Ok(u) => u,
        Err(_) if !urdfs.is_empty() && urdf.is_none() => urdfs[0].clone(),
        Err(e) => return Loaded::Error(e.to_string()),
    };
    let report = match inspect_urdf_in(&pkg, &chosen) {
        Ok(r) => r,
        Err(e) => return Loaded::Error(e.to_string()),
    };
    if let Some(e) = report.errors.first() {
        return Loaded::Error(format!("{chosen}: {e}"));
    }
    let text = String::from_utf8_lossy(pkg.read(&chosen).unwrap_or_default()).into_owned();
    let poses = zero_config_link_poses(&text).unwrap_or_default();
    let mut meshes = Vec::with_capacity(report.collisions.len());
    let mut mesh_errors = Vec::new();
    for c in &report.collisions {
        let mesh = match (c.action, c.mesh_path.as_deref()) {
            (Action::Pack, Some(path)) => match pkg.load_mesh(path) {
                Ok(m) => Some(Arc::new(m)),
                Err(e) => {
                    mesh_errors.push(format!("{}[{}]: {e}", c.link_name, c.collision_index));
                    None
                }
            },
            _ => None,
        };
        meshes.push(mesh);
        y.maybe_yield().await;
    }
    Loaded::Robot(Box::new(RobotDoc {
        name,
        pkg: Arc::new(pkg),
        urdfs,
        urdf: chosen,
        report,
        poses,
        meshes,
        mesh_errors,
    }))
}

/// Re-inspect an open package with another of its URDFs.
pub fn switch_urdf(doc: &RobotDoc, urdf: &str) -> Slot<Loaded> {
    let (name, pkg, urdf) = (doc.name.clone(), (*doc.pkg).clone(), urdf.to_string());
    spawn(async move { robot_doc(name, pkg, Some(&urdf), &mut Yielder::default()).await })
}

/// A package from a `.zip` archive's bytes.
pub async fn robot_from_zip(name: &str, bytes: &[u8]) -> Loaded {
    match MemPackage::from_zip(bytes) {
        Ok(pkg) => robot_doc(stem_of(name), pkg, None, &mut Yielder::default()).await,
        Err(e) => Loaded::Error(format!("{name}: {e}")),
    }
}

// ---------------------------------------------------------------------------
// Examples
// ---------------------------------------------------------------------------

/// The bundled example objects whose files exist (natively) or are listed.
pub fn example_objects() -> &'static [ExampleObject] {
    EXAMPLE_OBJECTS
}

/// The bundled robots that the manifest has files for.
pub fn example_robots() -> Vec<&'static ExampleRobot> {
    EXAMPLE_ROBOTS.iter().filter(|r| ROBOT_FILES.iter().any(|(f, _)| *f == r.folder)).collect()
}

pub fn load_object_example(ex: &'static ExampleObject) -> Slot<Loaded> {
    spawn(async move {
        match read_example(ex.filename).await {
            Ok(bytes) => object_from_bytes(ex.filename, &bytes),
            Err(e) => Loaded::Error(e),
        }
    })
}

pub fn load_robot_example(ex: &'static ExampleRobot) -> Slot<Loaded> {
    spawn(async move {
        let mut y = Yielder::default();
        let Some((_, files)) = ROBOT_FILES.iter().find(|(f, _)| *f == ex.folder) else {
            return Loaded::Error(format!("{}: not bundled", ex.folder));
        };
        let mut pkg = MemPackage::new();
        for f in *files {
            match read_example(&format!("{}/{f}", ex.folder)).await {
                Ok(bytes) => {
                    if let Err(e) = pkg.insert(f, bytes) {
                        return Loaded::Error(e.to_string());
                    }
                }
                Err(e) => return Loaded::Error(e),
            }
            y.maybe_yield().await;
        }
        robot_doc(ex.label.to_string(), pkg, Some(ex.urdf), &mut y).await
    })
}

/// Directory of the bundled examples: `MORPHIT_EXAMPLES_DIR`, `examples`
/// next to the executable, or the repository's `web/examples`.
#[cfg(not(target_arch = "wasm32"))]
pub fn examples_dir() -> std::path::PathBuf {
    if let Ok(d) = std::env::var("MORPHIT_EXAMPLES_DIR") {
        return d.into();
    }
    // (Cargo's own `target/<profile>/examples` is not it.)
    if let Some(d) = std::env::current_exe().ok().and_then(|e| e.parent().map(|p| p.join("examples")))
        && d.join(EXAMPLE_OBJECTS[0].filename).is_file()
    {
        return d;
    }
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/examples")
}

#[cfg(not(target_arch = "wasm32"))]
async fn read_example(rel: &str) -> Result<Vec<u8>, String> {
    let p = examples_dir().join(rel);
    std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))
}

#[cfg(target_arch = "wasm32")]
async fn read_example(rel: &str) -> Result<Vec<u8>, String> {
    web::fetch_bytes(&format!("examples/{rel}")).await
}

// ---------------------------------------------------------------------------
// Dialogs
// ---------------------------------------------------------------------------

const MESH_EXTENSIONS: [&str; 4] = ["obj", "stl", "ply", "dae"];

/// Pick one mesh file.
pub fn pick_mesh() -> Slot<Loaded> {
    spawn(async {
        let Some(f) = rfd::AsyncFileDialog::new().add_filter("Mesh", &MESH_EXTENSIONS).pick_file().await
        else {
            return Loaded::Nothing;
        };
        object_from_bytes(&f.file_name(), &f.read().await)
    })
}

/// Pick a `.zip` of a robot package.
pub fn pick_robot_zip() -> Slot<Loaded> {
    spawn(async {
        let Some(f) = rfd::AsyncFileDialog::new().add_filter("Zip archive", &["zip"]).pick_file().await
        else {
            return Loaded::Nothing;
        };
        robot_from_zip(&f.file_name(), &f.read().await).await
    })
}

/// Pick a robot package folder.
#[cfg(not(target_arch = "wasm32"))]
pub fn pick_robot_folder() -> Slot<Loaded> {
    spawn(async {
        let Some(d) = rfd::AsyncFileDialog::new().pick_folder().await else {
            return Loaded::Nothing;
        };
        robot_from_dir(d.path()).await
    })
}

#[cfg(target_arch = "wasm32")]
pub fn pick_robot_folder() -> Slot<Loaded> {
    spawn(async {
        match web::pick_folder().await {
            Ok(Some((name, pkg))) => robot_doc(name, pkg, None, &mut Yielder::default()).await,
            Ok(None) => Loaded::Nothing,
            Err(e) => Loaded::Error(e),
        }
    })
}

/// A package folder on disk.
#[cfg(not(target_arch = "wasm32"))]
pub async fn robot_from_dir(dir: &Path) -> Loaded {
    let name = dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_else(|| "robot".into());
    match MemPackage::from_dir(dir) {
        Ok(pkg) => robot_doc(name, pkg, None, &mut Yielder::default()).await,
        Err(e) => Loaded::Error(e.to_string()),
    }
}

/// A file or folder dropped on the window: a mesh, a `.zip`, a `.urdf`
/// (its folder is opened) or a package folder.
#[cfg(not(target_arch = "wasm32"))]
pub fn open_dropped(path: std::path::PathBuf) -> Slot<Loaded> {
    spawn(async move {
        let name = path.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
        let ext = ext_of(&name);
        if path.is_dir() {
            return robot_from_dir(&path).await;
        }
        let bytes = match std::fs::read(&path) {
            Ok(b) => b,
            Err(e) => return Loaded::Error(format!("{}: {e}", path.display())),
        };
        match ext.as_str() {
            "zip" => robot_from_zip(&name, &bytes).await,
            "urdf" => match path.parent() {
                Some(dir) => {
                    let dir_name =
                        dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
                    match MemPackage::from_dir(dir) {
                        Ok(pkg) => robot_doc(dir_name, pkg, Some(&name), &mut Yielder::default()).await,
                        Err(e) => Loaded::Error(e.to_string()),
                    }
                }
                None => Loaded::Error(format!("{name}: no folder")),
            },
            _ => object_from_bytes(&name, &bytes),
        }
    })
}

/// Save `bytes` as `name`: a save dialog natively, a download in the browser.
#[cfg(not(target_arch = "wasm32"))]
pub fn save(name: String, bytes: Vec<u8>) -> Slot<Result<Option<String>, String>> {
    spawn(async move {
        let Some(f) = rfd::AsyncFileDialog::new().set_file_name(&name).save_file().await else {
            return Ok(None);
        };
        std::fs::write(f.path(), bytes).map_err(|e| format!("{}: {e}", f.path().display()))?;
        Ok(Some(f.path().display().to_string()))
    })
}

#[cfg(target_arch = "wasm32")]
pub fn save(name: String, bytes: Vec<u8>) -> Slot<Result<Option<String>, String>> {
    let slot = Slot::default();
    slot.put(web::download(&name, &bytes).map(|()| Some(name)));
    slot
}

#[cfg(target_arch = "wasm32")]
mod web {
    use morphit_robot::vfs::MemPackage;
    use wasm_bindgen::JsCast;
    use wasm_bindgen::prelude::*;
    use wasm_bindgen_futures::JsFuture;

    fn js_err(e: JsValue) -> String {
        e.as_string().unwrap_or_else(|| format!("{e:?}"))
    }

    pub async fn fetch_bytes(url: &str) -> Result<Vec<u8>, String> {
        let w = web_sys::window().ok_or("no window")?;
        let resp: web_sys::Response =
            JsFuture::from(w.fetch_with_str(url)).await.map_err(js_err)?.dyn_into().map_err(js_err)?;
        if !resp.ok() {
            return Err(format!("{url}: HTTP {}", resp.status()));
        }
        let buf = JsFuture::from(resp.array_buffer().map_err(js_err)?).await.map_err(js_err)?;
        Ok(js_sys::Uint8Array::new(&buf).to_vec())
    }

    /// `<input type=file webkitdirectory>`: every file with its path
    /// relative to the picked folder's parent. `None` when cancelled.
    pub async fn pick_folder() -> Result<Option<(String, MemPackage)>, String> {
        let doc = web_sys::window().and_then(|w| w.document()).ok_or("no document")?;
        let input: web_sys::HtmlInputElement =
            doc.create_element("input").map_err(js_err)?.dyn_into().map_err(|_| "not an input element")?;
        input.set_type("file");
        input.set_multiple(true);
        input.set_attribute("webkitdirectory", "").map_err(js_err)?;
        let done = js_sys::Promise::new(&mut |resolve, _| {
            let on_change = resolve.clone();
            let cb = Closure::once_into_js(move || on_change.call0(&JsValue::NULL));
            input.set_onchange(Some(cb.unchecked_ref()));
            let cb = Closure::once_into_js(move || resolve.call0(&JsValue::NULL));
            let _ = input.add_event_listener_with_callback("cancel", cb.unchecked_ref());
        });
        input.click();
        JsFuture::from(done).await.map_err(js_err)?;
        let Some(files) = input.files() else { return Ok(None) };
        if files.length() == 0 {
            return Ok(None);
        }
        let mut pkg = MemPackage::new();
        let mut name = String::from("robot");
        for i in 0..files.length() {
            let Some(f) = files.get(i) else { continue };
            // (web-sys has no binding for this non-standard property.)
            let rel = js_sys::Reflect::get(&f, &"webkitRelativePath".into())
                .ok()
                .and_then(|v| v.as_string())
                .filter(|r| !r.is_empty())
                .unwrap_or_else(|| f.name());
            if i == 0 {
                name = rel.split('/').next().unwrap_or("robot").to_string();
            }
            let buf = JsFuture::from(f.array_buffer()).await.map_err(js_err)?;
            pkg.insert(&rel, js_sys::Uint8Array::new(&buf).to_vec()).map_err(|e| e.to_string())?;
        }
        Ok(Some((name, pkg)))
    }

    pub fn download(name: &str, bytes: &[u8]) -> Result<(), String> {
        let doc = web_sys::window().and_then(|w| w.document()).ok_or("no document")?;
        let parts = js_sys::Array::of1(&js_sys::Uint8Array::from(bytes));
        let blob = web_sys::Blob::new_with_u8_array_sequence(&parts).map_err(js_err)?;
        let url = web_sys::Url::create_object_url_with_blob(&blob).map_err(js_err)?;
        let a: web_sys::HtmlAnchorElement =
            doc.create_element("a").map_err(js_err)?.dyn_into().map_err(|_| "not an anchor element")?;
        a.set_href(&url);
        a.set_download(name);
        a.click();
        let _ = web_sys::Url::revoke_object_url(&url);
        Ok(())
    }
}
