//! Stage 2 of the robot pipeline, ported from `pack_robot_meshes.py`: pack
//! one collision mesh and write its sphere JSON to
//! `<output_dir>/spheres/<link>_<index>.json` (the `PackResult` layout the
//! Python `save_results` writes).

use std::path::{Path, PathBuf};
use std::sync::Arc;
use web_time::Instant;

use morphit::{Mesh, MeshPrepOptions, PackResult, Session, StepInfo};
use serde::{Deserialize, Serialize};

use crate::config::PackParams;
use crate::inspect::{Action, CollisionItem};
use crate::{Error, Result};

/// `<output_dir>/spheres`.
pub fn spheres_dir(output_dir: &Path) -> PathBuf {
    output_dir.join("spheres")
}

/// The mesh preparation the result JSON `text` (a saved `PackResult`) was
/// packed with: `config.model.union_overlapping_bodies` (true when absent)
/// and `config.model.convex_hull` (false when absent). Quality metrics must
/// use the same mesh ([`morphit::Mesh::prepared_with`]).
pub fn result_mesh_prep(text: &str) -> MeshPrepOptions {
    let v = serde_json::from_str::<serde_json::Value>(text).unwrap_or_default();
    let flag = |key: &str, default: bool| {
        v.pointer(&format!("/config/model/{key}")).and_then(|b| b.as_bool()).unwrap_or(default)
    };
    MeshPrepOptions {
        union_overlapping_bodies: flag("union_overlapping_bodies", true),
        convex_hull: flag("convex_hull", false),
    }
}

/// `<link>_<index>.json`.
pub fn json_filename(link_name: &str, collision_index: usize) -> String {
    format!("{link_name}_{collision_index}.json")
}

/// Run a session to completion, calling `on_step` after every iteration.
/// Returns the finalized session and the `(iteration, total_loss)` history.
pub fn run_session(
    config: morphit::Config,
    mesh: Arc<Mesh>,
    mut on_step: impl FnMut(&Session, &StepInfo),
) -> Result<(Session, Vec<(usize, f64)>)> {
    let mut session = Session::new(config, mesh)?;
    let mut history = Vec::with_capacity(session.total_iterations());
    // Step manually (not `Session::run`) so the observer can read the
    // current spheres.
    while !session.is_done() {
        let info = session.step()?;
        history.push((info.iteration, info.total_loss));
        on_step(&session, &info);
    }
    session.finalize();
    Ok((session, history))
}

/// A packed link: what `pack-link` reports, the full result, and the
/// `(iteration, total_loss)` history.
pub type PackedLink = (PackLinkOutcome, PackResult, Vec<(usize, f64)>);

/// What `pack-link` reports (key order as in Python's `PackResult`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct PackLinkOutcome {
    pub link_name: String,
    pub collision_index: usize,
    /// The requested sphere count.
    pub num_spheres: usize,
    pub json_path: String,
    pub elapsed_seconds: f64,
}

/// The mesh path of a `pack` item; errors for any other item. Warns when
/// the URDF scales the mesh (packing uses unscaled coordinates, as Python).
pub fn pack_mesh_path(item: &CollisionItem) -> Result<&str> {
    if item.action != Action::Pack {
        return Err(Error::Invalid(format!(
            "pack_one_link called on non-PACK item: {}[{}]",
            item.link_name, item.collision_index
        )));
    }
    let Some(mesh_path) = item.mesh_path.as_deref() else {
        return Err(Error::Invalid(format!(
            "pack_one_link called on item with no resolved mesh_path: {}[{}]",
            item.link_name, item.collision_index
        )));
    };
    if item.mesh_scale != [1.0; 3] {
        tracing::warn!(
            "{}[{}] uses <mesh scale={:?}>; packing in unscaled mesh coordinates",
            item.link_name,
            item.collision_index,
            item.mesh_scale
        );
    }
    Ok(mesh_path)
}

/// The session config for packing `item`, whose result is saved as
/// `<results_dir>/<link>_<index>.json`. Returns the config and that file name.
pub fn link_config(
    item: &CollisionItem,
    params: &PackParams,
    device: &str,
    results_dir: &str,
) -> Result<(morphit::Config, String)> {
    let json_name = json_filename(&item.link_name, item.collision_index);
    let config = params.config(device, results_dir, &json_name)?;
    Ok((config, json_name))
}

/// Pack one `pack` item: load its mesh, optimize, and save the sphere JSON.
/// `on_step` sees the live session after each iteration.
pub fn pack_one_link(
    item: &CollisionItem,
    params: &PackParams,
    device: &str,
    output_dir: &Path,
    on_step: impl FnMut(&Session, &StepInfo),
) -> Result<PackedLink> {
    let mesh_path = pack_mesh_path(item)?;
    let dir = spheres_dir(output_dir);
    std::fs::create_dir_all(&dir).map_err(|e| Error::Io(format!("cannot create {}: {e}", dir.display())))?;
    let (config, json_name) = link_config(item, params, device, &dir.display().to_string())?;

    let t0 = Instant::now();
    let mesh = Arc::new(Mesh::load(mesh_path)?);
    let (session, history) = run_session(config, mesh, on_step)?;
    let result = session.result();
    let json_path = dir.join(&json_name);
    result.save(&json_path)?;
    let outcome = PackLinkOutcome {
        link_name: item.link_name.clone(),
        collision_index: item.collision_index,
        num_spheres: params.num_spheres,
        json_path: json_path.display().to_string(),
        elapsed_seconds: t0.elapsed().as_secs_f64(),
    };
    Ok((outcome, result, history))
}

#[cfg(test)]
mod tests {
    use super::*;
    use morphit::glam::DVec3;

    fn write_cube_stl(path: &Path) {
        let m = morphit::shapes::box_mesh(DVec3::ZERO, DVec3::new(0.1, 0.05, 0.04));
        let mut s = String::from("solid c\n");
        for t in m.triangles() {
            s += "facet normal 0 0 0\nouter loop\n";
            for v in t {
                s += &format!("vertex {} {} {}\n", v.x, v.y, v.z);
            }
            s += "endloop\nendfacet\n";
        }
        s += "endsolid c\n";
        std::fs::write(path, s).unwrap();
    }

    #[test]
    fn packs_a_link_and_writes_its_json() {
        let t = tempfile::tempdir().unwrap();
        let mesh = t.path().join("link.stl");
        write_cube_stl(&mesh);
        let item = CollisionItem {
            link_name: "l1".into(),
            collision_index: 2,
            geometry_type: "mesh".into(),
            action: Action::Pack,
            mesh_path: Some(mesh.display().to_string()),
            mesh_filename: Some("link.stl".into()),
            mesh_scale: [1.0; 3],
            origin_xyz: [0.0; 3],
            origin_rpy: [0.0; 3],
            warning: None,
        };
        let params = PackParams {
            variant: "MorphIt-B".into(),
            num_spheres: 5,
            iterations: 6,
            seed: Some(3),
            advanced: vec![],
            union_overlapping_bodies: true,
            convex_hull: false,
        };
        let mut steps = 0;
        let (out, result, history) = pack_one_link(&item, &params, "cpu", t.path(), |s, info| {
            assert_eq!(info.iteration, steps);
            assert!(!s.spheres().is_empty());
            steps += 1;
        })
        .unwrap();
        assert_eq!(steps, 6);
        assert_eq!(history.len(), 6);
        assert_eq!(out.num_spheres, 5);
        assert!(out.json_path.ends_with("l1_2.json"));
        let saved = PackResult::load(&out.json_path).unwrap();
        assert_eq!(saved, result);
        assert_eq!(saved.config.random_seed, Some(3));
        assert_eq!(saved.config.output_filename, "l1_2.json");
        let text = std::fs::read_to_string(&out.json_path).unwrap();
        assert_eq!(result_mesh_prep(&text), MeshPrepOptions::DEFAULT);
        let mut off = saved.clone();
        off.config.model.union_overlapping_bodies = false;
        off.config.model.convex_hull = true;
        let want = MeshPrepOptions { union_overlapping_bodies: false, convex_hull: true };
        assert_eq!(result_mesh_prep(&off.to_json_string()), want);
        assert_eq!(result_mesh_prep(r#"{"centers": [], "radii": []}"#), MeshPrepOptions::DEFAULT);

        let mut bad = item.clone();
        bad.action = Action::Error;
        assert!(pack_one_link(&bad, &params, "cpu", t.path(), |_, _| {}).is_err());
    }
}
