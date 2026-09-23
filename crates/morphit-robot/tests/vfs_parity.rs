//! The in-memory pipeline (browser and desktop app) must agree with the
//! directory-based one the HTTP API uses: the same inspection report (paths
//! mapped to package keys) and a byte-identical assembled URDF, for every
//! bundled example robot.

use std::path::Path;

use morphit_robot::assemble::{MemSpheres, rewrite_urdf, rewrite_urdf_text};
use morphit_robot::color::DEFAULT_SPHERE_RGBA;
use morphit_robot::examples::EXAMPLE_ROBOTS;
use morphit_robot::inspect::{
    InspectionReport, find_urdfs, inspect_urdf, inspect_urdf_in, select_urdf, select_urdf_in,
};
use morphit_robot::pack::json_filename;
use morphit_robot::paths::canonical;
use morphit_robot::vfs::MemPackage;

/// The disk report with every path turned into a key relative to `root`.
fn to_keys(mut r: InspectionReport, root: &Path) -> InspectionReport {
    let prefix = canonical(root).display().to_string();
    let key =
        |p: &str| p.strip_prefix(&prefix).unwrap_or(p).trim_start_matches(['/', '\\']).replace('\\', "/");
    r.robot_dir = String::new();
    r.urdf_path = key(&r.urdf_path);
    r.urdfs_in_folder = r.urdfs_in_folder.iter().map(|p| key(p)).collect();
    for c in &mut r.collisions {
        c.mesh_path = c.mesh_path.as_deref().map(key);
    }
    r
}

#[test]
fn every_example_robot_matches() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/examples");
    if !dir.is_dir() {
        eprintln!("skipping: {} not found", dir.display());
        return;
    }
    let tmp = tempfile::tempdir().unwrap();
    for robot in EXAMPLE_ROBOTS {
        let root = dir.join(robot.folder);
        let disk_urdf = select_urdf(&find_urdfs(&root), Some(robot.urdf)).unwrap();
        let disk = inspect_urdf(&root, &disk_urdf).unwrap();

        let pkg = MemPackage::from_dir(&root).unwrap();
        let urdf_key = select_urdf_in(&pkg, Some(robot.urdf)).unwrap();
        let mem = inspect_urdf_in(&pkg, &urdf_key).unwrap();
        assert_eq!(mem, to_keys(disk.clone(), &root), "{}", robot.name);
        assert!(mem.to_pack().count() > 0, "{}", robot.name);

        // Synthetic spheres for every other pack item; the rest stay unpacked.
        let spheres_dir = tmp.path().join(robot.name);
        std::fs::create_dir_all(&spheres_dir).unwrap();
        let mut spheres = MemSpheres::default();
        for (k, item) in mem.to_pack().enumerate().filter(|(k, _)| k % 2 == 0) {
            let centers = vec![[0.01 * k as f64, -0.02, 0.1 / 3.0], [0.0, 0.05, 0.0]];
            let radii = vec![0.02 + k as f64 * 1e-3, 1.0 / 7.0];
            let json = serde_json::json!({ "centers": centers, "radii": radii });
            std::fs::write(
                spheres_dir.join(json_filename(&item.link_name, item.collision_index)),
                json.to_string(),
            )
            .unwrap();
            spheres.0.insert((item.link_name.clone(), item.collision_index), (centers, radii));
        }
        let out = tmp.path().join(format!("{}.urdf", robot.name));
        let (from_disk, disk_stats) =
            rewrite_urdf(&disk, &spheres_dir, &out, DEFAULT_SPHERE_RGBA, 1.0).unwrap();
        let text = String::from_utf8(pkg.read(&urdf_key).unwrap().to_vec()).unwrap();
        let (from_mem, mem_stats) =
            rewrite_urdf_text(&text, &mem, &spheres, DEFAULT_SPHERE_RGBA, 1.0).unwrap();
        assert_eq!(from_mem, from_disk, "{}", robot.name);
        assert_eq!(mem_stats.sphere_children_added, disk_stats.sphere_children_added, "{}", robot.name);
        assert_eq!(mem_stats.skipped_pack_items.len(), disk_stats.skipped_pack_items.len(), "{}", robot.name);
    }
}
