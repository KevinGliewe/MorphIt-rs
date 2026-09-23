//! Lists the files of every robot package under `web/examples`, so the
//! browser build can fetch a package file by file (a static server cannot
//! list directories).

use std::path::{Path, PathBuf};

fn walk(dir: &Path, root: &Path, out: &mut Vec<String>) {
    let Ok(entries) = std::fs::read_dir(dir) else { return };
    let mut entries: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        if p.is_dir() {
            walk(&p, root, out);
        } else if let Ok(rel) = p.strip_prefix(root) {
            let parts: Vec<String> =
                rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            out.push(parts.join("/"));
        }
    }
}

fn main() {
    let examples = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/examples");
    println!("cargo::rerun-if-changed={}", examples.display());
    let mut code =
        String::from("/// `(package folder, files relative to it)` of the bundled robot packages.\n");
    code += "pub const ROBOT_FILES: &[(&str, &[&str])] = &[\n";
    if let Ok(entries) = std::fs::read_dir(&examples) {
        let mut dirs: Vec<PathBuf> = entries.flatten().map(|e| e.path()).filter(|p| p.is_dir()).collect();
        dirs.sort();
        for d in dirs {
            let name = d.file_name().unwrap().to_string_lossy().into_owned();
            if name.ends_with(".spheres") {
                continue;
            }
            let mut files = Vec::new();
            walk(&d, &d, &mut files);
            code += &format!("    ({name:?}, &[\n");
            for f in files {
                code += &format!("        {f:?},\n");
            }
            code += "    ]),\n";
        }
    }
    code += "];\n";
    let out = PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("examples_manifest.rs");
    std::fs::write(out, code).unwrap();
}
