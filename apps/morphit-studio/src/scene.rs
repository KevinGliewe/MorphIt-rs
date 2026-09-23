//! The 3D view: meshes, spheres and the camera framing. Everything lives
//! under one root that turns the Z-up coordinates of meshes and URDFs into
//! Bevy's Y-up.

use std::f32::consts::FRAC_PI_2;
use std::sync::Arc;

use bevy::asset::RenderAssetUsages;
use bevy::mesh::{Indices, PrimitiveTopology};
use bevy::prelude::*;
use bevy_editor_cam::prelude::EditorCam;
use morphit::glam::{DMat3, DVec3};
use morphit_robot::kinematics::{Pose, collision_pose};

use crate::state::{Mode, Studio};

/// Root of the current document's entities (Z-up).
#[derive(Component)]
pub struct DocRoot;

/// A set of spheres drawn together: the object's, or one robot collision's.
#[derive(Component)]
pub struct SphereGroup {
    /// `None` for the object, `(link, collision index)` for a robot item.
    pub key: Option<(String, usize)>,
    pub material: Handle<StandardMaterial>,
    pub pool: Vec<Entity>,
    /// Identifies what is shown, to skip unchanged updates.
    pub shown: u64,
}

/// A packed mesh.
#[derive(Component)]
pub struct PackedMesh;

/// Shared sphere mesh.
#[derive(Resource)]
pub struct SphereMesh(pub Handle<Mesh>);

pub fn setup(mut commands: Commands, mut meshes: ResMut<Assets<Mesh>>) {
    commands.insert_resource(SphereMesh(meshes.add(Sphere::new(1.0).mesh().ico(3).unwrap())));
    // One directional light: WebGL2 supports no more.
    commands.insert_resource(GlobalAmbientLight { brightness: 800.0, ..default() });
    commands.spawn((
        DirectionalLight { illuminance: 6000.0, ..default() },
        Transform::from_xyz(3.0, 6.0, 4.0).looking_at(Vec3::ZERO, Vec3::Y),
    ));
}

/// A flat-shaded Bevy mesh of a MorphIt mesh.
pub fn to_bevy_mesh(m: &morphit::Mesh) -> Mesh {
    let mut pos = Vec::with_capacity(m.faces().len() * 3);
    let mut nrm = Vec::with_capacity(m.faces().len() * 3);
    for (tri, n) in m.triangles().zip(m.face_normals()) {
        let n = n.as_vec3().to_array();
        for v in tri {
            pos.push(v.as_vec3().to_array());
            nrm.push(n);
        }
    }
    let count = pos.len() as u32;
    Mesh::new(PrimitiveTopology::TriangleList, RenderAssetUsages::default())
        .with_inserted_attribute(Mesh::ATTRIBUTE_POSITION, pos)
        .with_inserted_attribute(Mesh::ATTRIBUTE_NORMAL, nrm)
        .with_inserted_indices(Indices::U32((0..count).collect()))
}

/// MorphIt's glam vector as Bevy's (the two use different glam versions).
fn v3(d: DVec3) -> Vec3 {
    Vec3::new(d.x as f32, d.y as f32, d.z as f32)
}

fn to_transform(p: &Pose) -> Transform {
    let r = p.rotation;
    let q = Quat::from_mat3(&Mat3::from_cols(v3(r.x_axis), v3(r.y_axis), v3(r.z_axis)));
    Transform { translation: v3(p.translation), rotation: q.normalize(), scale: Vec3::ONE }
}

fn srgb(c: [f64; 4]) -> Color {
    Color::srgba(c[0] as f32, c[1] as f32, c[2] as f32, c[3] as f32)
}

/// Rebuild the scene of the current document.
pub fn rebuild(
    mut commands: Commands,
    mut studio: ResMut<Studio>,
    roots: Query<Entity, With<DocRoot>>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut camera: Query<(&mut Transform, &mut EditorCam, &Projection)>,
) {
    if !studio.scene_dirty {
        return;
    }
    studio.scene_dirty = false;
    for e in &roots {
        commands.entity(e).despawn();
    }
    let root = commands
        .spawn((DocRoot, Transform::from_rotation(Quat::from_rotation_x(-FRAC_PI_2)), Visibility::default()))
        .id();
    let mesh_material = materials.add(StandardMaterial {
        base_color: Color::srgba(0.8, 0.82, 0.86, studio.view.mesh_alpha),
        alpha_mode: AlphaMode::Blend,
        cull_mode: None,
        double_sided: true,
        perceptual_roughness: 0.8,
        ..default()
    });
    let mut bounds: Option<(DVec3, DVec3)> = None;
    let mut grow = |lo: DVec3, hi: DVec3| {
        bounds = Some(match bounds {
            Some((a, b)) => (a.min(lo), b.max(hi)),
            None => (lo, hi),
        });
    };
    let colors = studio.group_colors();
    let group =
        |commands: &mut Commands, materials: &mut Assets<StandardMaterial>, key, color: [f64; 4], parent| {
            let material = materials.add(StandardMaterial {
                base_color: srgb(color),
                perceptual_roughness: 0.5,
                ..default()
            });
            commands.spawn((
                SphereGroup { key, material, pool: Vec::new(), shown: u64::MAX },
                Transform::default(),
                Visibility::default(),
                ChildOf(parent),
            ));
        };
    match studio.mode {
        Mode::Object => {
            if let Some(doc) = &studio.object {
                commands.spawn((
                    PackedMesh,
                    Mesh3d(meshes.add(to_bevy_mesh(&doc.mesh))),
                    MeshMaterial3d(mesh_material.clone()),
                    Transform::default(),
                    ChildOf(root),
                ));
                let (lo, hi) = doc.mesh.bounds();
                grow(lo, hi);
                group(&mut commands, &mut materials, None, colors[0].1, root);
            }
        }
        Mode::Robot => {
            if let Some(doc) = studio.robot.as_ref().map(|r| &r.doc) {
                for (item, mesh) in doc.report.collisions.iter().zip(&doc.meshes) {
                    let Some(mesh) = mesh else { continue };
                    let link = doc.poses.get(&item.link_name).copied().unwrap_or(Pose::IDENTITY);
                    let pose = collision_pose(&link, item);
                    let at = commands.spawn((to_transform(&pose), Visibility::default(), ChildOf(root))).id();
                    commands.spawn((
                        PackedMesh,
                        Mesh3d(meshes.add(to_bevy_mesh(mesh))),
                        MeshMaterial3d(mesh_material.clone()),
                        Transform::default(),
                        ChildOf(at),
                    ));
                    let key = (item.link_name.clone(), item.collision_index);
                    let color =
                        colors.iter().find(|(k, _)| k.as_ref() == Some(&key)).map_or(colors[0].1, |c| c.1);
                    group(&mut commands, &mut materials, Some(key), color, at);
                    let (lo, hi) = world_bounds(mesh, &pose);
                    grow(lo, hi);
                }
            }
        }
    }
    studio.spheres_dirty = true;
    if let (Some((lo, hi)), Ok((mut t, mut cam, proj))) = (bounds, camera.single_mut()) {
        frame(&mut t, &mut cam, proj, lo, hi);
    }
}

/// Bounding box of `mesh` placed at `pose`.
fn world_bounds(mesh: &Arc<morphit::Mesh>, pose: &Pose) -> (DVec3, DVec3) {
    let (lo, hi) = mesh.bounds();
    let mut a = DVec3::INFINITY;
    let mut b = DVec3::NEG_INFINITY;
    for i in 0..8 {
        let c = DVec3::new(
            if i & 1 == 0 { lo.x } else { hi.x },
            if i & 2 == 0 { lo.y } else { hi.y },
            if i & 4 == 0 { lo.z } else { hi.z },
        );
        let p = pose.transform_point(c);
        a = a.min(p);
        b = b.max(p);
    }
    (a, b)
}

/// Point the camera at the Z-up box `lo..hi`.
fn frame(t: &mut Transform, cam: &mut EditorCam, proj: &Projection, lo: DVec3, hi: DVec3) {
    let to_y_up = DMat3::from_rotation_x(-std::f64::consts::FRAC_PI_2);
    let center = v3(to_y_up * ((lo + hi) * 0.5));
    let radius = ((hi - lo).length() * 0.5).max(1e-3) as f32;
    let fov = match proj {
        Projection::Perspective(p) => p.fov,
        _ => 0.8,
    };
    let dist = radius / (fov * 0.5).tan() * 1.15;
    let dir = Vec3::new(1.0, 0.7, 1.3).normalize();
    *t = Transform::from_translation(center + dir * dist).looking_at(center, Vec3::Y);
    cam.last_anchor_depth = -(dist as f64);
}

/// Place the spheres of every group: live ones from the running job, else
/// the finished results.
pub fn update_spheres(
    mut commands: Commands,
    mut studio: ResMut<Studio>,
    sphere_mesh: Res<SphereMesh>,
    mut groups: Query<(Entity, &mut SphereGroup)>,
    mut transforms: Query<(&mut Transform, &mut Visibility), Without<SphereGroup>>,
) {
    if !studio.spheres_dirty {
        return;
    }
    studio.spheres_dirty = false;
    for (entity, mut g) in &mut groups {
        let Some((sig, centers, radii)) = studio.spheres_for(g.key.as_ref()) else {
            hide_from(&mut g, 0, &mut transforms);
            g.shown = 0;
            continue;
        };
        if sig == g.shown {
            continue;
        }
        g.shown = sig;
        while g.pool.len() < radii.len() {
            let material = g.material.clone();
            let e = commands
                .spawn((
                    Mesh3d(sphere_mesh.0.clone()),
                    MeshMaterial3d(material),
                    Transform::from_scale(Vec3::ZERO),
                    Visibility::Hidden,
                    ChildOf(entity),
                ))
                .id();
            g.pool.push(e);
        }
        let mut deferred = Vec::new();
        for (i, (c, r)) in centers.iter().zip(&radii).enumerate() {
            let t = Transform::from_translation(Vec3::new(c[0] as f32, c[1] as f32, c[2] as f32))
                .with_scale(Vec3::splat(*r as f32));
            match transforms.get_mut(g.pool[i]) {
                Ok((mut tr, mut vis)) => {
                    *tr = t;
                    *vis = Visibility::Inherited;
                }
                // Spawned this frame: set through commands.
                Err(_) => deferred.push((g.pool[i], t)),
            }
        }
        for (e, t) in deferred {
            commands.entity(e).insert((t, Visibility::Inherited));
        }
        hide_from(&mut g, radii.len(), &mut transforms);
    }
}

fn hide_from(
    g: &mut SphereGroup,
    from: usize,
    transforms: &mut Query<(&mut Transform, &mut Visibility), Without<SphereGroup>>,
) {
    for &e in g.pool.iter().skip(from) {
        if let Ok((_, mut vis)) = transforms.get_mut(e) {
            *vis = Visibility::Hidden;
        }
    }
}

/// Mesh opacity and visibility from the view settings.
pub fn update_mesh_look(
    studio: Res<Studio>,
    mut applied: Local<Option<(bool, f32, usize)>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut meshes: Query<(&MeshMaterial3d<StandardMaterial>, &mut Visibility), With<PackedMesh>>,
) {
    let want = (studio.view.show_mesh, studio.view.mesh_alpha, meshes.iter().len());
    if *applied == Some(want) {
        return;
    }
    *applied = Some(want);
    for (m, mut vis) in &mut meshes {
        *vis = if studio.view.show_mesh { Visibility::Inherited } else { Visibility::Hidden };
        if let Some(mat) = materials.get_mut(&m.0) {
            mat.base_color.set_alpha(studio.view.mesh_alpha);
        }
    }
}
