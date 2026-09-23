//! Every mesh bundled with the web examples (`web/examples`) must load:
//! object meshes (.obj) and the robot packages' collision and visual meshes
//! (.stl/.STL/.dae). Skips with a message when the directory is absent.

use std::path::{Path, PathBuf};

use morphit::Mesh;

fn examples_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/examples")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for e in std::fs::read_dir(dir).unwrap() {
        let p = e.unwrap().path();
        if p.is_dir() {
            walk(&p, out);
        } else if let Some(ext) = p.extension().and_then(|e| e.to_str())
            && Mesh::supported_extensions().contains(&ext.to_ascii_lowercase().as_str())
        {
            out.push(p);
        }
    }
}

#[test]
fn all_bundled_meshes_load() {
    let dir = examples_dir();
    if !dir.is_dir() {
        eprintln!("skipping: {} not found", dir.display());
        return;
    }
    let mut files = Vec::new();
    walk(&dir, &mut files);
    assert!(files.len() > 100, "found only {} meshes", files.len());
    let mut dae = 0;
    let mut failures = Vec::new();
    for f in &files {
        match Mesh::load(f) {
            Ok(m) => {
                assert!(m.volume() > 0.0 && m.area() > 0.0, "{}", f.display());
                dae += usize::from(f.extension().unwrap().eq_ignore_ascii_case("dae"));
            }
            Err(e) => failures.push(format!("{}: {e}", f.display())),
        }
    }
    assert!(failures.is_empty(), "{} of {} failed:\n{}", failures.len(), files.len(), failures.join("\n"));
    assert!(dae >= 50, "only {dae} DAE files loaded");
}
