//! Robot packages held in memory: what the browser build and the desktop app
//! work on instead of a directory. Keys are the files' relative paths with
//! `/` separators, normalized by [`normalize_rel`] (so an upload can never
//! name something outside the package).

use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use morphit::Mesh;

use crate::paths::{normalize_rel, walk_files};
use crate::{Error, Result};

/// A robot description package as a map from relative path to file bytes.
#[derive(Clone, Debug, Default)]
pub struct MemPackage {
    files: BTreeMap<String, Arc<[u8]>>,
}

impl MemPackage {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add (or replace) a file; returns its normalized key.
    pub fn insert(&mut self, rel: &str, bytes: impl Into<Arc<[u8]>>) -> Result<String> {
        let key = normalize_rel(rel)?.join("/");
        self.files.insert(key.clone(), bytes.into());
        Ok(key)
    }

    /// A package from `(relative path, bytes)` pairs.
    pub fn from_files<S: AsRef<str>, B: Into<Arc<[u8]>>>(
        files: impl IntoIterator<Item = (S, B)>,
    ) -> Result<Self> {
        let mut pkg = Self::new();
        for (rel, bytes) in files {
            pkg.insert(rel.as_ref(), bytes)?;
        }
        Ok(pkg)
    }

    /// Read every file below `root` (keys relative to `root`).
    pub fn from_dir(root: &Path) -> Result<Self> {
        let mut pkg = Self::new();
        for path in walk_files(root) {
            let rel = path.strip_prefix(root).unwrap_or(&path);
            let rel: Vec<String> =
                rel.components().map(|c| c.as_os_str().to_string_lossy().into_owned()).collect();
            let bytes = std::fs::read(&path)
                .map_err(|e| Error::Io(format!("cannot read {}: {e}", path.display())))?;
            pkg.insert(&rel.join("/"), bytes)?;
        }
        Ok(pkg)
    }

    /// Unpack a `.zip` archive. Directory entries and macOS resource forks
    /// (`__MACOSX/`) are skipped; entries that would escape the package are
    /// refused.
    #[cfg(feature = "zip")]
    pub fn from_zip(bytes: &[u8]) -> Result<Self> {
        use std::io::Read as _;
        let bad = |e: zip::result::ZipError| Error::Invalid(format!("not a valid zip archive: {e}"));
        let mut archive = zip::ZipArchive::new(std::io::Cursor::new(bytes)).map_err(bad)?;
        let mut pkg = Self::new();
        for i in 0..archive.len() {
            let mut entry = archive.by_index(i).map_err(bad)?;
            let name = entry.name().to_string();
            if entry.is_dir() || name.starts_with("__MACOSX/") {
                continue;
            }
            let mut data = Vec::with_capacity(entry.size() as usize);
            entry.read_to_end(&mut data).map_err(|e| Error::Invalid(format!("{name}: {e}")))?;
            pkg.insert(&name, data)?;
        }
        Ok(pkg)
    }

    pub fn len(&self) -> usize {
        self.files.len()
    }

    pub fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    /// Total size of all files in bytes.
    pub fn total_bytes(&self) -> usize {
        self.files.values().map(|b| b.len()).sum()
    }

    /// Every key, sorted.
    pub fn files(&self) -> impl Iterator<Item = &str> {
        self.files.keys().map(String::as_str)
    }

    fn key(rel: &str) -> Option<String> {
        normalize_rel(rel).ok().map(|p| p.join("/"))
    }

    /// The bytes of `rel`, if it is a file of the package.
    pub fn read(&self, rel: &str) -> Option<&[u8]> {
        self.files.get(&Self::key(rel)?).map(|b| &b[..])
    }

    pub fn is_file(&self, rel: &str) -> bool {
        Self::key(rel).is_some_and(|k| self.files.contains_key(&k))
    }

    /// Whether some file lies below `rel`.
    pub fn is_dir(&self, rel: &str) -> bool {
        let Some(k) = Self::key(rel) else { return false };
        let prefix = format!("{k}/");
        self.files.range(prefix.clone()..).next().is_some_and(|(f, _)| f.starts_with(&prefix))
    }

    pub fn exists(&self, rel: &str) -> bool {
        self.is_file(rel) || self.is_dir(rel)
    }

    /// Directory keys (every proper prefix of a file key), sorted.
    pub fn dirs(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for f in self.files.keys() {
            let mut end = 0;
            while let Some(i) = f[end..].find('/') {
                end += i;
                out.push(f[..end].to_string());
                end += 1;
            }
        }
        out.sort();
        out.dedup();
        out
    }

    /// Every `.urdf` file, sorted.
    pub fn urdfs(&self) -> Vec<String> {
        self.files().filter(|f| f.ends_with(".urdf")).map(str::to_string).collect()
    }

    /// Load the mesh at `rel`; the key becomes the mesh's source path.
    pub fn load_mesh(&self, rel: &str) -> Result<Mesh> {
        let bytes =
            self.read(rel).ok_or_else(|| Error::Invalid(format!("{rel}: no such file in the package")))?;
        let ext = Path::new(rel).extension().map(|e| e.to_string_lossy().into_owned()).unwrap_or_default();
        Ok(Mesh::load_from_bytes(bytes, &ext, Some(rel.to_string()))?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keys_are_normalized_and_confined() {
        let mut p = MemPackage::new();
        assert_eq!(p.insert("\\pkg\\urdf\\.\\r.urdf", b"x".to_vec()).unwrap(), "pkg/urdf/r.urdf");
        p.insert("pkg/meshes/a.stl", b"y".to_vec()).unwrap();
        assert!(p.insert("../evil", b"z".to_vec()).is_err());
        assert!(p.insert("C:/evil", b"z".to_vec()).is_err());
        assert_eq!(p.read("/pkg/urdf/r.urdf"), Some(&b"x"[..]));
        assert!(p.is_dir("pkg") && p.is_dir("pkg/meshes") && !p.is_dir("pkg/meshes/a.stl"));
        assert!(!p.is_dir("pk"));
        assert!(p.exists("pkg/urdf") && p.is_file("pkg/meshes/a.stl"));
        assert_eq!(p.dirs(), ["pkg", "pkg/meshes", "pkg/urdf"]);
        assert_eq!(p.urdfs(), ["pkg/urdf/r.urdf"]);
        assert_eq!(p.len(), 2);
        assert_eq!(p.total_bytes(), 2);
    }

    #[test]
    fn from_dir_matches_the_tree() {
        let t = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(t.path().join("a/b")).unwrap();
        std::fs::write(t.path().join("a/b/f.urdf"), "u").unwrap();
        std::fs::write(t.path().join("g.stl"), "s").unwrap();
        let p = MemPackage::from_dir(t.path()).unwrap();
        assert_eq!(p.files().collect::<Vec<_>>(), ["a/b/f.urdf", "g.stl"]);
    }

    #[cfg(feature = "zip")]
    #[test]
    fn zip_round_trip() {
        use std::io::Write as _;
        let mut buf = std::io::Cursor::new(Vec::new());
        {
            let mut w = zip::ZipWriter::new(&mut buf);
            let o = zip::write::SimpleFileOptions::default();
            w.add_directory("pkg/", o).unwrap();
            w.start_file("pkg/r.urdf", o).unwrap();
            w.write_all(b"<robot/>").unwrap();
            w.start_file("__MACOSX/pkg/._r.urdf", o).unwrap();
            w.write_all(b"junk").unwrap();
            w.finish().unwrap();
        }
        let p = MemPackage::from_zip(buf.get_ref()).unwrap();
        assert_eq!(p.files().collect::<Vec<_>>(), ["pkg/r.urdf"]);
        assert_eq!(p.read("pkg/r.urdf"), Some(&b"<robot/>"[..]));
        assert!(MemPackage::from_zip(b"not a zip").is_err());
    }
}
