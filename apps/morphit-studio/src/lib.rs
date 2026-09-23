//! MorphIt Studio: pack meshes and robot collision meshes with spheres,
//! interactively. One code base for the desktop and the browser.

pub mod io;
pub mod job;
mod scene;
pub mod state;
pub mod task;
mod ui;

use bevy::camera::CameraOutputMode;
use bevy::camera::visibility::RenderLayers;
use bevy::picking::PickingSystems;
use bevy::picking::mesh_picking::MeshPickingPlugin;
use bevy::prelude::*;
use bevy::render::render_resource::BlendState;
use bevy_editor_cam::input::{
    CameraPointerMap, DefaultInputPlugin, EditorCamInputMessage, default_camera_inputs,
};
use bevy_editor_cam::prelude::*;
use bevy_egui::input::egui_wants_any_pointer_input;
use bevy_egui::{EguiGlobalSettings, EguiPlugin, EguiPrimaryContextPass, PrimaryEguiContext};

use crate::state::Studio;

/// Build and run the app.
pub fn run() {
    #[cfg(target_arch = "wasm32")]
    console_error_panic_hook::set_once();
    App::new()
        .insert_resource(ClearColor(Color::srgb(0.12, 0.13, 0.16)))
        .add_plugins(
            DefaultPlugins
                .set(bevy::log::LogPlugin {
                    // The Vulkan loader reports unrelated system layers as errors.
                    filter: "info,wgpu=error,wgpu_hal=off,naga=warn".into(),
                    ..default()
                })
                .set(WindowPlugin {
                    primary_window: Some(Window {
                        title: "MorphIt Studio".into(),
                        canvas: Some("#morphit-canvas".into()),
                        fit_canvas_to_parent: true,
                        prevent_default_event_handling: true,
                        ..default()
                    }),
                    ..default()
                }),
        )
        .add_plugins((
            EguiPlugin::default(),
            MeshPickingPlugin,
            DefaultEditorCamPlugins.build().disable::<DefaultInputPlugin>(),
            CameraInputPlugin,
        ))
        .init_resource::<Studio>()
        .add_systems(Startup, (setup_camera, scene::setup, startup_options))
        .add_systems(EguiPrimaryContextPass, ui::ui)
        .add_systems(
            Update,
            (dropped_files, scene::rebuild, scene::update_spheres, scene::update_mesh_look).chain(),
        )
        .run();
}

/// bevy_editor_cam's default mouse controls (right drag orbits, left drag
/// pans, the wheel zooms), except while the pointer is over egui.
struct CameraInputPlugin;

impl Plugin for CameraInputPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<EditorCamInputMessage>().init_resource::<CameraPointerMap>().add_systems(
            PreUpdate,
            (
                default_camera_inputs.run_if(not(egui_wants_any_pointer_input)),
                EditorCamInputMessage::receive_messages,
                EditorCamInputMessage::send_pointer_inputs,
            )
                .chain()
                .after(PickingSystems::Last)
                .before(EditorCam::update_camera_positions),
        );
    }
}

/// The 3D camera (its viewport is the area between the panels) and an
/// overlay camera that draws egui over the whole window.
fn setup_camera(mut commands: Commands, mut egui_settings: ResMut<EguiGlobalSettings>) {
    egui_settings.auto_create_primary_context = false;
    commands.spawn((
        Camera3d::default(),
        EditorCam::default(),
        Transform::from_xyz(0.6, 0.4, 0.8).looking_at(Vec3::ZERO, Vec3::Y),
    ));
    commands.spawn((
        PrimaryEguiContext,
        Camera2d,
        RenderLayers::none(),
        Camera {
            order: 1,
            output_mode: CameraOutputMode::Write {
                blend_state: Some(BlendState::ALPHA_BLENDING),
                clear_color: ClearColorConfig::None,
            },
            clear_color: ClearColorConfig::Custom(Color::NONE),
            ..default()
        },
    ));
}

/// Startup options: `--example NAME`, `--open PATH` and `--pack` on the
/// command line, `?example=NAME&pack` in the browser. `NAME` is an example
/// object or robot (`bunny`, `kinova`, ...).
fn startup_options(mut studio: ResMut<Studio>) {
    let (example, open, pack) = options();
    if let Some(name) = example {
        if let Some(ex) = io::example_objects().iter().find(|o| o.name == name) {
            studio.load(ex.label, io::load_object_example(ex));
        } else if let Some(ex) = io::example_robots().into_iter().find(|r| r.name == name) {
            studio.set_mode(state::Mode::Robot);
            studio.load(ex.label, io::load_robot_example(ex));
        } else {
            warn!("unknown example {name:?}");
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    if let Some(path) = open {
        studio.load(path.clone(), io::open_dropped(path.into()));
    }
    #[cfg(target_arch = "wasm32")]
    let _ = open;
    studio.pack_after_load = pack;
}

#[cfg(not(target_arch = "wasm32"))]
fn options() -> (Option<String>, Option<String>, bool) {
    let mut args = std::env::args().skip(1);
    let (mut example, mut open, mut pack) = (None, None, false);
    while let Some(a) = args.next() {
        match a.as_str() {
            "--example" => example = args.next(),
            "--open" => open = args.next(),
            "--pack" => pack = true,
            _ => warn!("unknown argument {a:?} (use --example NAME, --open PATH, --pack)"),
        }
    }
    (example, open, pack)
}

#[cfg(target_arch = "wasm32")]
fn options() -> (Option<String>, Option<String>, bool) {
    let search = web_sys::window().and_then(|w| w.location().search().ok()).unwrap_or_default();
    let mut example = None;
    let mut pack = false;
    for kv in search.trim_start_matches('?').split('&') {
        match kv.split_once('=') {
            Some(("example", v)) => example = Some(v.to_string()),
            _ if kv == "pack" => pack = true,
            _ => {}
        }
    }
    (example, None, pack)
}

/// Open files and folders dropped on the window (desktop).
fn dropped_files(mut drops: MessageReader<FileDragAndDrop>, mut studio: ResMut<Studio>) {
    for d in drops.read() {
        #[cfg(not(target_arch = "wasm32"))]
        if let FileDragAndDrop::DroppedFile { path_buf, .. } = d
            && !studio.busy()
        {
            let name = path_buf.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
            studio.load(name, io::open_dropped(path_buf.clone()));
        }
        #[cfg(target_arch = "wasm32")]
        let _ = (d, &mut studio);
    }
}
