//! Path helpers for uploaded robot packages.

use std::path::{Component, Path, PathBuf};

use crate::{Error, Result, py_repr};

/// The components of an upload's relative path (the browser's
/// `webkitRelativePath`): backslashes become slashes, leading slashes, empty
/// and `.` components are dropped, and anything that could escape the
/// package (`..`, drive letters, root or prefix components) is refused with
/// the API's messages.
pub fn normalize_rel(rel: &str) -> Result<Vec<String>> {
    let norm = rel.replace('\\', "/");
    let norm = norm.trim_start_matches('/');
    if norm.is_empty() {
        return Err(Error::Invalid("Empty upload filename.".into()));
    }
    let bad = || Error::Invalid(format!("Bad upload path: {}", py_repr(norm)));
    // pathlib drops empty and `.` components.
    let parts: Vec<&str> = norm.split('/').filter(|p| !p.is_empty() && *p != ".").collect();
    let drive = parts.first().is_some_and(|f| f.len() >= 2 && f.as_bytes()[1] == b':');
    if parts.is_empty() || drive || parts.contains(&"..") {
        return Err(bad());
    }
    for p in &parts {
        let mut c = Path::new(p).components();
        if !matches!((c.next(), c.next()), (Some(Component::Normal(_)), None)) {
            return Err(bad());
        }
    }
    Ok(parts.into_iter().map(str::to_string).collect())
}

/// Join an upload's relative path onto `base`, refusing anything that could
/// land outside it. Port of the API's `_safe_join`, with the same messages.
/// Drive letters are rejected on every OS (Python on Linux would accept
/// `C:/x` as a relative directory name).
pub fn safe_join(base: &Path, rel: &str) -> Result<PathBuf> {
    let mut out = base.to_path_buf();
    out.extend(normalize_rel(rel)?);
    if !out.starts_with(base) {
        let norm = rel.replace('\\', "/");
        return Err(Error::Invalid(format!(
            "Upload path escapes work dir: {}",
            py_repr(norm.trim_start_matches('/'))
        )));
    }
    Ok(out)
}

/// Resolve `rel` against the directory `dir` of a slash-separated relative
/// path, lexically: `.` is dropped and `..` removes a component. `None` if
/// the result would leave the root.
pub fn join_lexical(dir: &str, rel: &str) -> Option<String> {
    let mut out: Vec<&str> = Vec::new();
    for part in dir.split('/').chain(rel.split('/')) {
        match part {
            "" | "." => {}
            ".." => {
                out.pop()?;
            }
            p => out.push(p),
        }
    }
    Some(out.join("/"))
}

/// The directory part of a slash-separated relative path (`""` at the root).
pub fn parent_rel(rel: &str) -> &str {
    rel.rfind('/').map_or("", |i| &rel[..i])
}

/// `path.resolve()`: canonical absolute path as a display string, without the
/// Windows `\\?\` verbatim prefix. Falls back to the input when it does not
/// exist.
pub fn canonical(path: &Path) -> PathBuf {
    match std::fs::canonicalize(path) {
        Ok(p) => strip_verbatim(p),
        Err(_) => path.to_path_buf(),
    }
}

fn strip_verbatim(p: PathBuf) -> PathBuf {
    let s = p.to_string_lossy();
    if let Some(rest) = s.strip_prefix(r"\\?\UNC\") {
        PathBuf::from(format!(r"\\{rest}"))
    } else if let Some(rest) = s.strip_prefix(r"\\?\") {
        PathBuf::from(rest)
    } else {
        p
    }
}

/// Whether `path` (resolved) lies inside `root` (resolved).
pub fn is_within(path: &Path, root: &Path) -> bool {
    match (std::fs::canonicalize(path), std::fs::canonicalize(root)) {
        (Ok(p), Ok(r)) => p.starts_with(r),
        _ => false,
    }
}

/// Every file below `dir`, recursively, in sorted order.
pub fn walk_files(dir: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![dir.to_path_buf()];
    while let Some(d) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&d) else { continue };
        for e in entries.flatten() {
            let p = e.path();
            match e.file_type() {
                Ok(t) if t.is_dir() => stack.push(p),
                Ok(_) => out.push(p),
                Err(_) => {}
            }
        }
    }
    out.sort();
    out
}

/// Copy the directory tree `src` to `dst` (created).
pub fn copy_tree(src: &Path, dst: &Path) -> std::io::Result<()> {
    std::fs::create_dir_all(dst)?;
    for e in std::fs::read_dir(src)? {
        let e = e?;
        let to = dst.join(e.file_name());
        if e.file_type()?.is_dir() {
            copy_tree(&e.path(), &to)?;
        } else {
            std::fs::copy(e.path(), to)?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safe_join_accepts_relative_paths() {
        let base = Path::new("work");
        assert_eq!(safe_join(base, "meshes/a.stl").unwrap(), base.join("meshes").join("a.stl"));
        assert_eq!(
            safe_join(base, "pkg\\urdf\\r.urdf").unwrap(),
            base.join("pkg").join("urdf").join("r.urdf")
        );
        assert_eq!(safe_join(base, "/lead/slash.obj").unwrap(), base.join("lead").join("slash.obj"));
        assert_eq!(safe_join(base, "./x.obj").unwrap(), base.join("x.obj"));
        assert_eq!(safe_join(base, "a//b").unwrap(), base.join("a").join("b"));
    }

    #[test]
    fn safe_join_rejects_escapes() {
        let base = Path::new("work");
        let msg = |r: &str| safe_join(base, r).unwrap_err().to_string();
        assert_eq!(msg(""), "Empty upload filename.");
        assert_eq!(msg("/"), "Empty upload filename.");
        assert_eq!(msg("../x"), "Bad upload path: '../x'");
        assert_eq!(msg("a/../../b"), "Bad upload path: 'a/../../b'");
        assert_eq!(msg("C:\\x"), "Bad upload path: 'C:/x'");
        assert_eq!(msg("..\\x"), "Bad upload path: '../x'");
        assert_eq!(msg("c:/x"), "Bad upload path: 'c:/x'");
        assert!(safe_join(base, "\\\\srv\\share\\x").is_ok_and(|p| p.starts_with(base)));
        assert!(safe_join(base, "/etc/passwd").is_ok_and(|p| p == base.join("etc").join("passwd")));
    }

    #[test]
    fn lexical_joins() {
        assert_eq!(join_lexical("pkg/urdf", "../meshes/a.stl").as_deref(), Some("pkg/meshes/a.stl"));
        assert_eq!(join_lexical("", "./a//b.stl").as_deref(), Some("a/b.stl"));
        assert_eq!(join_lexical("pkg", "../../x"), None);
        assert_eq!(parent_rel("pkg/urdf/r.urdf"), "pkg/urdf");
        assert_eq!(parent_rel("r.urdf"), "");
        assert_eq!(normalize_rel("\\a\\.\\b").unwrap(), ["a", "b"]);
    }

    #[test]
    fn tree_helpers() {
        let t = tempfile::tempdir().unwrap();
        let src = t.path().join("src");
        std::fs::create_dir_all(src.join("a/b")).unwrap();
        std::fs::write(src.join("a/b/f.txt"), "x").unwrap();
        std::fs::write(src.join("g.txt"), "y").unwrap();
        copy_tree(&src, &t.path().join("dst")).unwrap();
        let files = walk_files(&t.path().join("dst"));
        assert_eq!(files.len(), 2);
        assert!(is_within(&files[0], t.path()));
        assert!(!is_within(t.path(), &src));
        assert!(!canonical(&files[0]).to_string_lossy().starts_with(r"\\?\"));
    }
}
