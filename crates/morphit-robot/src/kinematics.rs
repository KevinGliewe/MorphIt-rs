//! Link poses of a URDF at the zero joint configuration, for drawing a robot
//! and the spheres packed into its links.

use std::collections::{BTreeMap, VecDeque};

use morphit::glam::{DMat3, DVec3};
use serde::Serialize;

use crate::assemble::rpy_to_matrix;
use crate::inspect::{CollisionItem, parse_origin};
use crate::xml::{child, children};
use crate::{Error, Result};

/// A rigid transform: `p -> rotation * p + translation`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Pose {
    pub rotation: DMat3,
    pub translation: DVec3,
}

impl Pose {
    pub const IDENTITY: Pose = Pose { rotation: DMat3::IDENTITY, translation: DVec3::ZERO };

    /// A URDF `<origin xyz rpy>`.
    pub fn from_origin(xyz: [f64; 3], rpy: [f64; 3]) -> Pose {
        Pose { rotation: rpy_to_matrix(rpy), translation: DVec3::from_array(xyz) }
    }

    /// `self` after `other` (`other` expressed in `self`'s frame).
    pub fn mul(&self, other: &Pose) -> Pose {
        Pose {
            rotation: self.rotation * other.rotation,
            translation: self.rotation * other.translation + self.translation,
        }
    }

    pub fn transform_point(&self, p: DVec3) -> DVec3 {
        self.rotation * p + self.translation
    }
}

/// A `<joint>` of the robot.
#[derive(Clone, Debug, PartialEq, Serialize)]
pub struct Joint {
    pub name: String,
    /// `type` attribute (`fixed`, `revolute`, ...).
    pub kind: String,
    pub parent: String,
    pub child: String,
    pub xyz: [f64; 3],
    pub rpy: [f64; 3],
}

fn parse(urdf: &str) -> Result<roxmltree::Document<'_>> {
    let opts = roxmltree::ParsingOptions { allow_dtd: true, ..Default::default() };
    roxmltree::Document::parse_with_options(urdf, opts)
        .map_err(|e| Error::Invalid(format!("URDF is not valid XML: {e}")))
}

/// The joints of `urdf` (direct children of `<robot>`; the `<joint>` inside
/// a `<transmission>` is not one), in document order.
pub fn parse_joints(urdf: &str) -> Result<Vec<Joint>> {
    let doc = parse(urdf)?;
    let joints = children(doc.root_element(), "joint")
        .filter_map(|j| {
            let parent = child(j, "parent")?.attribute("link")?;
            let child_link = child(j, "child")?.attribute("link")?;
            let (xyz, rpy) = parse_origin(child(j, "origin"));
            Some(Joint {
                name: j.attribute("name").unwrap_or_default().to_string(),
                kind: j.attribute("type").unwrap_or_default().to_string(),
                parent: parent.to_string(),
                child: child_link.to_string(),
                xyz,
                rpy,
            })
        })
        .collect();
    Ok(joints)
}

/// World pose of every link with all joints at zero: breadth-first from the
/// root links (links that are no joint's child), composing joint origins.
/// Links a joint cycle or a missing parent leaves unreached get the identity.
pub fn zero_config_link_poses(urdf: &str) -> Result<BTreeMap<String, Pose>> {
    let joints = parse_joints(urdf)?;
    let doc = parse(urdf)?;
    let links: Vec<&str> = doc
        .root_element()
        .descendants()
        .filter(|n| n.is_element() && n.has_tag_name("link"))
        .filter_map(|n| n.attribute("name"))
        .collect();
    let mut poses: BTreeMap<String, Pose> = BTreeMap::new();
    let mut queue: VecDeque<&str> = VecDeque::new();
    for l in &links {
        if !joints.iter().any(|j| j.child == *l) {
            poses.insert(l.to_string(), Pose::IDENTITY);
            queue.push_back(l);
        }
    }
    while let Some(parent) = queue.pop_front() {
        let base = poses[parent];
        for j in joints.iter().filter(|j| j.parent == parent) {
            if poses.contains_key(&j.child) {
                continue;
            }
            poses.insert(j.child.clone(), base.mul(&Pose::from_origin(j.xyz, j.rpy)));
            queue.push_back(&j.child);
        }
    }
    for l in links {
        poses.entry(l.to_string()).or_insert(Pose::IDENTITY);
    }
    Ok(poses)
}

/// World pose of a collision's mesh frame, given its link's world pose.
pub fn collision_pose(link: &Pose, item: &CollisionItem) -> Pose {
    link.mul(&Pose::from_origin(item.origin_xyz, item.origin_rpy))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::assemble::apply_collision_transform;

    const CHAIN: &str = r#"<robot name="r">
  <link name="base"/><link name="a"/><link name="b"/><link name="loose"/>
  <joint name="j1" type="revolute"><parent link="base"/><child link="a"/>
    <origin xyz="0 0 1" rpy="0 0 1.5707963267948966"/></joint>
  <joint name="j2" type="fixed"><parent link="a"/><child link="b"/><origin xyz="1 0 0"/></joint>
  <transmission name="t"><joint name="j1"><hardwareInterface>x</hardwareInterface></joint></transmission>
</robot>"#;

    fn close(a: DVec3, b: DVec3) -> bool {
        (a - b).length() < 1e-12
    }

    #[test]
    fn chain_composes_joint_origins() {
        let joints = parse_joints(CHAIN).unwrap();
        assert_eq!(joints.len(), 2, "the transmission's joint is not a joint");
        assert_eq!((joints[0].kind.as_str(), joints[1].xyz), ("revolute", [1.0, 0.0, 0.0]));
        let p = zero_config_link_poses(CHAIN).unwrap();
        assert_eq!(p["base"], Pose::IDENTITY);
        assert_eq!(p["loose"], Pose::IDENTITY);
        assert!(close(p["a"].translation, DVec3::new(0.0, 0.0, 1.0)));
        // a is turned 90 degrees about z, so b's +x offset points along +y.
        assert!(close(p["b"].translation, DVec3::new(0.0, 1.0, 1.0)));
        assert!(close(p["b"].transform_point(DVec3::X), DVec3::new(0.0, 2.0, 1.0)));
    }

    #[test]
    fn collision_pose_matches_the_assembly_transform() {
        let item = CollisionItem {
            link_name: "l".into(),
            collision_index: 0,
            geometry_type: "mesh".into(),
            action: crate::inspect::Action::Pack,
            mesh_path: None,
            mesh_filename: None,
            mesh_scale: [1.0; 3],
            origin_xyz: [0.5, 0.0, 0.2],
            origin_rpy: [0.1, 0.2, 0.3],
            warning: None,
        };
        let c = [1.0, 2.0, 3.0];
        let want = DVec3::from_array(apply_collision_transform(c, item.origin_xyz, item.origin_rpy));
        let got = collision_pose(&Pose::IDENTITY, &item).transform_point(DVec3::from_array(c));
        assert!(close(got, want), "{got} vs {want}");
    }

    #[test]
    fn bundled_spherical_urdfs() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../web/examples");
        let Ok(text) = std::fs::read_to_string(dir.join("ur5.spherical.urdf")) else {
            eprintln!("skipping: examples not found");
            return;
        };
        let poses = zero_config_link_poses(&text).unwrap();
        let joints = parse_joints(&text).unwrap();
        // Every sphere link sits at its parent's pose moved by the joint offset.
        let mut checked = 0;
        for j in joints.iter().filter(|j| j.child.contains("_sphere")) {
            let want = poses[&j.parent].transform_point(DVec3::from_array(j.xyz));
            assert!(close(poses[&j.child].translation, want), "{}", j.name);
            checked += 1;
        }
        assert!(checked > 10);
    }
}
