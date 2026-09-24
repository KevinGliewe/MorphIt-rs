//! The egui panels: controls on the left, the loss plot and quality at the
//! bottom, the collision table of a robot on the right.

use bevy::camera::Viewport;
use bevy::prelude::*;
use bevy::window::PrimaryWindow;
use bevy_editor_cam::prelude::EditorCam;
use bevy_egui::{EguiContexts, egui};
use egui_plot::{Legend, Line, Plot, PlotPoints};
use morphit::LossId;
use morphit_robot::config::ALLOWED_VARIANTS;
use morphit_robot::inspect::Action;
use serde_json::Value;

use crate::io;
use crate::job::Phase;
use crate::state::{Export, Mode, Params, Studio};

pub fn ui(
    mut contexts: EguiContexts,
    mut studio: ResMut<Studio>,
    mut camera: Single<&mut Camera, With<EditorCam>>,
    window: Single<&Window, With<PrimaryWindow>>,
) -> Result {
    let ctx = contexts.ctx_mut()?;
    studio.poll();
    let s = &mut *studio;
    egui::SidePanel::left("controls").resizable(true).default_width(300.0).show(ctx, |ui| {
        egui::ScrollArea::vertical().show(ui, |ui| controls(ui, s));
    });
    egui::TopBottomPanel::bottom("plots").resizable(true).default_height(220.0).show(ctx, |ui| plots(ui, s));
    if s.mode == Mode::Robot && s.robot.is_some() {
        egui::SidePanel::right("links")
            .resizable(true)
            .default_width(360.0)
            .show(ctx, |ui| robot_table(ui, s));
    }
    // The 3D view gets what the panels leave free.
    let free = ctx.available_rect();
    let sf = window.scale_factor();
    let (w, h) = (window.physical_width().max(1), window.physical_height().max(1));
    let x = ((free.min.x * sf) as u32).min(w - 1);
    let y = ((free.min.y * sf) as u32).min(h - 1);
    let size = UVec2::new(
        ((free.width() * sf) as u32).clamp(1, w - x),
        ((free.height() * sf) as u32).clamp(1, h - y),
    );
    let position = UVec2::new(x, y);
    if camera.viewport.as_ref().map(|v| (v.physical_position, v.physical_size)) != Some((position, size)) {
        camera.viewport = Some(Viewport { physical_position: position, physical_size: size, ..default() });
    }
    Ok(())
}

fn controls(ui: &mut egui::Ui, s: &mut Studio) {
    ui.heading("MorphIt Studio");
    let mut mode = s.mode;
    ui.horizontal(|ui| {
        ui.selectable_value(&mut mode, Mode::Object, "Object");
        ui.selectable_value(&mut mode, Mode::Robot, "Robot");
    });
    s.set_mode(mode);
    ui.separator();

    let busy = s.busy();
    ui.add_enabled_ui(!busy, |ui| match s.mode {
        Mode::Object => {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Open mesh…").clicked() {
                    s.load("mesh", io::pick_mesh());
                }
                ui.menu_button("Examples", |ui| {
                    egui::ScrollArea::vertical().max_height(360.0).show(ui, |ui| {
                        for ex in io::example_objects() {
                            if ui.button(ex.label).clicked() {
                                s.load(ex.label, io::load_object_example(ex));
                                ui.close();
                            }
                        }
                    });
                });
            });
            if let Some(o) = &s.object {
                ui.label(format!("{}: {} faces", o.name, o.mesh.faces().len()));
            }
        }
        Mode::Robot => {
            ui.horizontal_wrapped(|ui| {
                if ui.button("Open folder…").clicked() {
                    s.load("robot folder", io::pick_robot_folder());
                }
                if ui.button("Open .zip…").clicked() {
                    s.load("robot archive", io::pick_robot_zip());
                }
                ui.menu_button("Examples", |ui| {
                    for ex in io::example_robots() {
                        if ui.button(ex.label).clicked() {
                            s.load(ex.label, io::load_robot_example(ex));
                            ui.close();
                        }
                    }
                });
            });
            if let Some(r) = &s.robot {
                ui.label(&r.doc.name);
                if r.doc.urdfs.len() > 1 {
                    let mut chosen = r.doc.urdf.clone();
                    egui::ComboBox::from_label("URDF").selected_text(&chosen).show_ui(ui, |ui| {
                        for u in &r.doc.urdfs {
                            ui.selectable_value(&mut chosen, u.clone(), u);
                        }
                    });
                    if chosen != r.doc.urdf {
                        s.switch_urdf(chosen);
                    }
                }
            }
        }
    });
    if let Some(what) = s.loading() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label(format!("Loading {what}…"));
        });
    }
    ui.separator();

    ui.add_enabled_ui(!s.job_running(), |ui| params(ui, &mut s.params, &s.devices));
    ui.separator();
    run_controls(ui, s);
    ui.separator();
    view_controls(ui, s);
    ui.separator();
    exports(ui, s);
    if let Some((msg, error)) = &s.message {
        ui.separator();
        let color = if *error { ui.visuals().error_fg_color } else { ui.visuals().text_color() };
        ui.label(egui::RichText::new(msg).color(color));
    }
}

fn params(ui: &mut egui::Ui, p: &mut Params, devices: &[morphit::GpuInfo]) {
    egui::Grid::new("params").num_columns(2).show(ui, |ui| {
        ui.label("Preset");
        let before = p.variant.clone();
        egui::ComboBox::from_id_salt("preset").selected_text(&p.variant).show_ui(ui, |ui| {
            for v in ALLOWED_VARIANTS {
                ui.selectable_value(&mut p.variant, v.to_string(), v);
            }
        });
        if p.variant != before {
            p.advanced = Params::advanced_for(&p.variant);
        }
        ui.end_row();
        ui.label("Spheres");
        ui.add(egui::DragValue::new(&mut p.num_spheres).range(1..=200));
        ui.end_row();
        ui.label("Iterations");
        ui.add(egui::DragValue::new(&mut p.iterations).range(1..=1000));
        ui.end_row();
        ui.label("Seed");
        ui.horizontal(|ui| {
            ui.checkbox(&mut p.random_seed, "random");
            ui.add_enabled(!p.random_seed, egui::DragValue::new(&mut p.seed));
        });
        ui.end_row();
        ui.label("Mesh prep")
            .on_hover_text("Merge overlapping closed bodies into their union before packing (default on)");
        ui.checkbox(&mut p.union_overlapping_bodies, "merge overlapping bodies");
        ui.end_row();
        ui.label("");
        ui.checkbox(&mut p.convex_hull, "convex hull of each body").on_hover_text(
            "Replace every body with its convex hull before merging: a simpler, closed shape;              concavities and holes are filled (default off)",
        );
        ui.end_row();
        ui.label("Device");
        egui::ComboBox::from_id_salt("device").selected_text(&p.device).show_ui(ui, |ui| {
            ui.selectable_value(&mut p.device, "auto".to_string(), "auto");
            ui.selectable_value(&mut p.device, "cpu".to_string(), "cpu");
            for g in devices {
                let id = format!("gpu:{}", g.index);
                ui.selectable_value(&mut p.device, id.clone(), format!("{id} {}", g.name));
            }
        });
        ui.end_row();
    });
    egui::CollapsingHeader::new("Advanced").show(ui, |ui| {
        egui::Grid::new("advanced").num_columns(2).show(ui, |ui| {
            for (flat, _, value) in &mut p.advanced {
                ui.label(flat.replace('_', " "));
                match value {
                    Value::Bool(b) => {
                        ui.checkbox(b, "");
                    }
                    Value::Number(n) if n.is_u64() => {
                        let mut v = n.as_u64().unwrap_or(0);
                        ui.add(egui::DragValue::new(&mut v).speed(10));
                        *value = Value::from(v);
                    }
                    Value::Number(n) => {
                        let mut v = n.as_f64().unwrap_or(0.0);
                        ui.add(egui::DragValue::new(&mut v).speed(0.01).max_decimals(4));
                        *value = serde_json::Number::from_f64(v).map_or(Value::Null, Value::Number);
                    }
                    _ => {
                        ui.label("–");
                    }
                }
                ui.end_row();
            }
        });
        if ui.button("Preset values").clicked() {
            p.advanced = Params::advanced_for(&p.variant);
        }
    });
}

fn run_controls(ui: &mut egui::Ui, s: &mut Studio) {
    let has_doc = match s.mode {
        Mode::Object => s.object.is_some(),
        Mode::Robot => s.robot.is_some(),
    };
    ui.horizontal_wrapped(|ui| {
        if s.job_running() {
            let pause = if s.paused() { "Resume" } else { "Pause" };
            if ui.button(pause).clicked() {
                s.toggle_pause();
            }
            if ui.button("Finish now").on_hover_text("Stop iterating and finalize the current mesh").clicked()
            {
                s.finish_now();
            }
            if ui.button("Cancel").clicked() {
                s.cancel();
            }
        } else {
            let label = match (s.mode, s.robot.as_ref().and_then(|r| r.selected)) {
                (Mode::Robot, Some(_)) => "Pack selected",
                (Mode::Robot, None) => "Pack all",
                _ => "Pack",
            };
            if ui.add_enabled(has_doc && s.loading().is_none(), egui::Button::new(label)).clicked() {
                s.start_pack();
            }
            if ui.add_enabled(has_doc, egui::Button::new("Clear")).clicked() {
                s.clear_results();
            }
        }
    });
    if let Some(live) = &s.live
        && s.job_running()
    {
        let label = match live.phase {
            Phase::Preparing => "preparing".to_string(),
            Phase::Paused => "paused".to_string(),
            _ => format!("{} / {}", live.iteration, live.total_iterations),
        };
        let frac = if live.total_iterations > 0 {
            live.iteration as f32 / live.total_iterations as f32
        } else {
            0.0
        };
        let item =
            if live.items > 1 { format!("mesh {}/{} · ", live.item + 1, live.items) } else { String::new() };
        ui.add(egui::ProgressBar::new(frac).text(format!("{item}{label}")));
        if !live.device.is_empty() {
            ui.small(format!("searches on {}", live.device));
        }
    }
}

fn view_controls(ui: &mut egui::Ui, s: &mut Studio) {
    let v = &mut s.view;
    let before = (v.color, v.color_variation);
    ui.horizontal(|ui| {
        ui.checkbox(&mut v.show_mesh, "Mesh");
        ui.add(egui::Slider::new(&mut v.mesh_alpha, 0.05..=1.0).text("opacity"));
    });
    ui.horizontal(|ui| {
        ui.color_edit_button_rgb(&mut v.color);
        ui.label("Spheres");
        if s.mode == Mode::Robot {
            ui.add(egui::Slider::new(&mut v.color_variation, 0.0..=1.0).text("hue spread"));
        }
    });
    if (v.color, v.color_variation) != before {
        s.scene_dirty = true;
    }
}

fn exports(ui: &mut egui::Ui, s: &mut Studio) {
    match s.mode {
        Mode::Object => {
            let packed = s.object.as_ref().is_some_and(|o| o.result.is_some());
            if let Some(o) = &mut s.object {
                ui.horizontal(|ui| {
                    ui.checkbox(&mut o.anchored, "Anchored");
                    ui.add(
                        egui::DragValue::new(&mut o.total_mass).range(1e-6..=1e6).speed(0.01).suffix(" kg"),
                    );
                });
            }
            ui.add_enabled_ui(packed && !s.job_running(), |ui| {
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Save JSON").clicked() {
                        s.save(Export::ResultJson);
                    }
                    if ui.button("Save URDF").clicked() {
                        s.save(Export::Urdf);
                    }
                    if ui.button("Save MJCF").clicked() {
                        s.save(Export::Mjcf);
                    }
                    if ui.add_enabled(!s.quality_running(), egui::Button::new("Analyze")).clicked() {
                        s.analyze();
                    }
                });
            });
        }
        Mode::Robot => {
            let Some(r) = &s.robot else { return };
            let any = !r.results.is_empty();
            let assembled = r.assembled.is_some();
            ui.add_enabled_ui(any && !s.job_running(), |ui| {
                ui.horizontal_wrapped(|ui| {
                    if ui.button("Assemble").clicked() {
                        s.assemble();
                    }
                    if ui.add_enabled(assembled, egui::Button::new("Save spherical URDF")).clicked() {
                        s.save(Export::SphericalUrdf);
                    }
                    if ui.add_enabled(!s.quality_running(), egui::Button::new("Analyze")).clicked() {
                        s.analyze();
                    }
                });
            });
        }
    }
    if s.quality_running() {
        ui.horizontal(|ui| {
            ui.spinner();
            ui.label("Analyzing…");
        });
    }
}

fn plots(ui: &mut egui::Ui, s: &mut Studio) {
    ui.horizontal(|ui| {
        ui.strong("Loss");
        ui.checkbox(&mut s.view.loss_terms, "terms");
    });
    let history = s.live.as_ref().map(|l| l.history.as_slice()).unwrap_or(&[]);
    ui.columns(2, |cols| {
        let log = |v: f64| v.max(1e-12).log10();
        Plot::new("loss").legend(Legend::default()).y_axis_label("log10").allow_scroll(false).show(
            &mut cols[0],
            |plot| {
                let pts: PlotPoints = history.iter().map(|(i, total, _)| [*i as f64, log(*total)]).collect();
                plot.line(Line::new("total", pts).width(2.0_f32));
                if s.view.loss_terms {
                    for id in LossId::ALL {
                        if history.iter().all(|(_, _, w)| w[id.index()] == 0.0) {
                            continue;
                        }
                        let pts: PlotPoints =
                            history.iter().map(|(i, _, w)| [*i as f64, log(w[id.index()])]).collect();
                        plot.line(Line::new(id.name(), pts));
                    }
                }
            },
        );
        egui::ScrollArea::vertical().id_salt("quality").show(&mut cols[1], |ui| quality(ui, s));
    });
}

fn quality(ui: &mut egui::Ui, s: &Studio) {
    match s.mode {
        Mode::Object => {
            let Some(q) = s.object.as_ref().and_then(|o| o.quality.as_ref()) else {
                ui.weak("Pack, then Analyze for quality metrics.");
                return;
            };
            egui::Grid::new("quality").striped(true).show(ui, |ui| {
                let row = |ui: &mut egui::Ui, k: &str, v: String| {
                    ui.label(k);
                    ui.monospace(v);
                    ui.end_row();
                };
                row(ui, "spheres", format!("{} ({} outside, {} tiny)", q.actual_n, q.n_out, q.n_tiny));
                row(ui, "covered inside", format!("{:.1} %", q.r_in * 100.0));
                row(ui, "outside the mesh", format!("{:.1} %", q.r_out * 100.0));
                row(ui, "union / mesh volume", format!("{:.3}", q.r_uni));
                row(ui, "surface distance", format!("mean {:.2}, max {:.2} (×1000)", q.d_avg_mm, q.d_max_mm));
                row(ui, "mass error", format!("{:.1} %", q.mass_rel * 100.0));
                row(ui, "center of mass error", format!("{:.2} %", q.com_rel * 100.0));
                row(ui, "inertia error", format!("{:.1} %", q.i_rel * 100.0));
            });
        }
        Mode::Robot => {
            let Some((links, overall)) = s.robot.as_ref().and_then(|r| r.quality.as_ref()) else {
                ui.weak("Pack, then Analyze for per-link metrics.");
                return;
            };
            ui.label(format!(
                "{} spheres · coverage {:.1} % · distance mean {:.4}, max {:.4}",
                overall.num_spheres,
                overall.coverage * 100.0,
                overall.d_mean,
                overall.d_max
            ));
            egui::Grid::new("link_quality").striped(true).show(ui, |ui| {
                for h in ["link", "spheres", "coverage", "d mean", "d max"] {
                    ui.strong(h);
                }
                ui.end_row();
                for q in links {
                    ui.label(format!("{}[{}]", q.link_name, q.collision_index));
                    ui.monospace(q.num_spheres.to_string());
                    ui.monospace(format!("{:.1} %", q.coverage * 100.0));
                    ui.monospace(format!("{:.4}", q.d_mean));
                    ui.monospace(format!("{:.4}", q.d_max));
                    ui.end_row();
                }
            });
        }
    }
}

fn robot_table(ui: &mut egui::Ui, s: &mut Studio) {
    let live_key = s.live_key().cloned();
    let Some(r) = &mut s.robot else { return };
    ui.horizontal(|ui| {
        ui.strong(format!("{} collisions", r.doc.report.collisions.len()));
        ui.label(format!("· {} packed", r.results.len()));
        if r.selected.is_some() && ui.small_button("select none").clicked() {
            r.selected = None;
        }
    });
    let mut save_link = None;
    egui::ScrollArea::vertical().show(ui, |ui| {
        egui::Grid::new("collisions").striped(true).num_columns(4).show(ui, |ui| {
            for h in ["link", "geometry", "action", "spheres"] {
                ui.strong(h);
            }
            ui.end_row();
            for (i, c) in r.doc.report.collisions.iter().enumerate() {
                let selected = r.selected == Some(i);
                let label = format!("{}[{}]", c.link_name, c.collision_index);
                if ui.selectable_label(selected, label).clicked() {
                    r.selected = if selected { None } else { Some(i) };
                }
                ui.label(&c.geometry_type);
                let action = match c.action {
                    Action::Pack => "pack",
                    Action::RemovePrimitive => "remove",
                    Action::RemoveAlreadySphere => "remove",
                    Action::Error => "error",
                };
                let resp = ui.label(action);
                if let Some(w) = &c.warning {
                    resp.on_hover_text(w);
                }
                let key = (c.link_name.clone(), c.collision_index);
                match (r.results.get(&key), c.action) {
                    (Some(res), _) => {
                        if ui
                            .small_button(format!("{} ⬇", res.num_spheres))
                            .on_hover_text("Save the sphere JSON")
                            .clicked()
                        {
                            save_link = Some(i);
                        }
                    }
                    (None, Action::Pack) if live_key.as_ref() == Some(&key) => {
                        ui.spinner();
                    }
                    (None, Action::Pack) => {
                        ui.weak("–");
                    }
                    _ => {
                        ui.label("");
                    }
                }
                ui.end_row();
            }
        });
    });
    if let Some((text, summary)) = &r.assembled {
        ui.separator();
        ui.label(summary);
        ui.small(format!("{} bytes", text.len()));
    }
    if let Some(i) = save_link {
        s.save(Export::LinkJson(i));
    }
}
