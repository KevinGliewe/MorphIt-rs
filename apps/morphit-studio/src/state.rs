//! What the studio holds: the open object or robot, the packing
//! parameters, the running job and the background tasks, plus the actions
//! the UI triggers.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::atomic::Ordering;

use bevy::prelude::*;
use morphit::glam::DVec3;
use morphit::{
    Config, GpuInfo, Mesh, MeshFormat, MeshPrepOptions, PackResult, QualityMetrics, QualityOptions,
    evaluate_packing,
};
use morphit_robot::assemble::{MemSpheres, rewrite_urdf_text};
use morphit_robot::color::vary_color;
use morphit_robot::config::{ADVANCED_OVERRIDE_MAP, ALLOWED_VARIANTS, PackParams};
use morphit_robot::inspect::Action;
use morphit_robot::object_model::{ObjectModelOptions, write_object_mjcf, write_object_urdf};
use morphit_robot::quality::{LinkQuality, Overall, aggregate_overall, quality_metrics};
use serde_json::Value;

use crate::io::{self, Loaded, RobotDoc};
use crate::job::{self, JobShared, JobSpec, JobStatus, Phase};
use crate::task::{Slot, Yielder, spawn};

/// A sphere group: `None` for the object, `(link, collision index)` for a
/// robot collision.
pub type GroupKey = Option<(String, usize)>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mode {
    Object,
    Robot,
}

/// The packing parameters (the web UI's form).
pub struct Params {
    pub variant: String,
    pub num_spheres: usize,
    pub iterations: usize,
    pub random_seed: bool,
    pub seed: u64,
    pub device: String,
    /// Mesh preparation: merge overlapping closed bodies before packing.
    pub union_overlapping_bodies: bool,
    /// Mesh preparation: replace each body with its convex hull before merging.
    pub convex_hull: bool,
    /// `(flat key, dotted key, value)` of the advanced overrides, starting
    /// at the preset's values.
    pub advanced: Vec<(&'static str, &'static str, Value)>,
}

impl Params {
    pub fn advanced_for(variant: &str) -> Vec<(&'static str, &'static str, Value)> {
        let c = Config::from_preset_name(variant).unwrap_or_else(|_| Config::from_preset(morphit::Preset::B));
        ADVANCED_OVERRIDE_MAP
            .iter()
            .map(|&(flat, dotted)| (flat, dotted, c.get(dotted).unwrap_or(Value::Null)))
            .collect()
    }

    /// The mesh preparation these parameters ask for.
    pub fn prep_options(&self) -> MeshPrepOptions {
        MeshPrepOptions {
            union_overlapping_bodies: self.union_overlapping_bodies,
            convex_hull: self.convex_hull,
        }
    }

    fn pack_params(&self) -> PackParams {
        PackParams {
            variant: self.variant.clone(),
            num_spheres: self.num_spheres,
            iterations: self.iterations,
            seed: (!self.random_seed).then_some(self.seed),
            advanced: self.advanced.iter().map(|(_, d, v)| (d.to_string(), v.clone())).collect(),
            union_overlapping_bodies: self.union_overlapping_bodies,
            convex_hull: self.convex_hull,
        }
    }
}

impl Default for Params {
    fn default() -> Self {
        Params {
            variant: ALLOWED_VARIANTS[2].into(),
            num_spheres: 20,
            iterations: 200,
            random_seed: false,
            seed: 42,
            device: "auto".into(),
            union_overlapping_bodies: true,
            convex_hull: false,
            advanced: Params::advanced_for(ALLOWED_VARIANTS[2]),
        }
    }
}

pub struct View {
    pub show_mesh: bool,
    /// Draw the meshes as packing prepares them (merged, convex hulls).
    pub show_prepared: bool,
    /// Format of "Save prepared mesh".
    pub mesh_format: MeshFormat,
    pub mesh_alpha: f32,
    /// Sphere color.
    pub color: [f32; 3],
    /// Hue spread over robot links (0..1).
    pub color_variation: f32,
    pub loss_terms: bool,
}

impl Default for View {
    fn default() -> Self {
        View {
            show_mesh: true,
            show_prepared: false,
            mesh_format: MeshFormat::Obj,
            mesh_alpha: 0.35,
            color: [0.2, 0.6, 1.0],
            color_variation: 0.6,
            loss_terms: false,
        }
    }
}

pub struct ObjectDoc {
    pub name: String,
    pub mesh: Arc<Mesh>,
    pub result: Option<PackResult>,
    pub quality: Option<QualityMetrics>,
    /// Export options.
    pub anchored: bool,
    pub total_mass: f64,
}

pub struct RobotState {
    pub doc: RobotDoc,
    pub results: HashMap<(String, usize), PackResult>,
    /// Assembled spherical URDF and the summary line.
    pub assembled: Option<(String, String)>,
    pub quality: Option<(Vec<LinkQuality>, Overall)>,
    pub selected: Option<usize>,
}

pub enum QualityOut {
    Object(QualityMetrics),
    Robot(Vec<LinkQuality>, Overall),
    Error(String),
}

#[derive(Resource)]
pub struct Studio {
    pub mode: Mode,
    pub params: Params,
    pub view: View,
    pub devices: Vec<GpuInfo>,
    devices_task: Option<Slot<Vec<GpuInfo>>>,
    pub object: Option<ObjectDoc>,
    pub robot: Option<RobotState>,
    pub job: Option<Arc<JobShared>>,
    job_mode: Mode,
    job_items: Vec<Option<(String, usize)>>,
    job_version: u64,
    job_finished_seen: usize,
    /// Latest snapshot of the job.
    pub live: Option<JobStatus>,
    loading: Option<(String, Slot<Loaded>)>,
    saving: Vec<Slot<Result<Option<String>, String>>>,
    quality_task: Option<Slot<QualityOut>>,
    /// Status line; `true` for errors.
    pub message: Option<(String, bool)>,
    pub scene_dirty: bool,
    pub spheres_dirty: bool,
    results_version: u64,
    /// Start packing once the pending load has finished.
    pub pack_after_load: bool,
    /// Prepared meshes of the document (the object, or every robot
    /// collision) for [`View::show_prepared`], with the options used.
    prepared: Option<(MeshPrepOptions, PreparedMeshes)>,
    prepared_task: Option<(MeshPrepOptions, Slot<PreparedMeshes>)>,
    /// A "Save prepared mesh" waiting for its mesh: file name, format, mesh.
    prepared_save: Option<(String, MeshFormat, Slot<Arc<Mesh>>)>,
}

/// One entry per mesh of the document; `None` where there is no mesh.
type PreparedMeshes = Vec<Option<Arc<Mesh>>>;

impl Default for Studio {
    fn default() -> Self {
        Studio {
            mode: Mode::Object,
            params: Params::default(),
            view: View::default(),
            devices: Vec::new(),
            devices_task: Some(spawn(async { morphit::init_gpu().await })),
            object: None,
            robot: None,
            job: None,
            job_mode: Mode::Object,
            job_items: Vec::new(),
            job_version: 0,
            job_finished_seen: 0,
            live: None,
            loading: None,
            saving: Vec::new(),
            quality_task: None,
            message: None,
            scene_dirty: false,
            spheres_dirty: false,
            results_version: 1,
            pack_after_load: false,
            prepared: None,
            prepared_task: None,
            prepared_save: None,
        }
    }
}

impl Studio {
    fn info(&mut self, s: impl Into<String>) {
        self.message = Some((s.into(), false));
    }

    fn error(&mut self, s: impl Into<String>) {
        self.message = Some((s.into(), true));
    }

    pub fn loading(&self) -> Option<&str> {
        self.loading.as_ref().map(|(what, _)| what.as_str())
    }

    pub fn job_running(&self) -> bool {
        self.job.as_ref().is_some_and(|j| j.is_running())
    }

    pub fn busy(&self) -> bool {
        self.loading.is_some() || self.job_running()
    }

    /// The robot item the running job is packing.
    pub fn live_key(&self) -> Option<&(String, usize)> {
        let live = self.live.as_ref().filter(|_| self.job_running() && self.job_mode == Mode::Robot)?;
        self.job_items.get(live.item)?.as_ref()
    }

    pub fn quality_running(&self) -> bool {
        self.quality_task.is_some()
    }

    /// The meshes of the current document: the object, or every robot collision.
    fn doc_meshes(&self) -> PreparedMeshes {
        match self.mode {
            Mode::Object => self.object.iter().map(|o| Some(o.mesh.clone())).collect(),
            Mode::Robot => self.robot.as_ref().map(|r| r.doc.meshes.clone()).unwrap_or_default(),
        }
    }

    /// The mesh to draw at `index` of [`Studio::doc_meshes`]: the prepared one
    /// when [`View::show_prepared`] is on and it is ready.
    pub fn shown_mesh(&self, index: usize, loaded: &Arc<Mesh>) -> Arc<Mesh> {
        self.prepared
            .as_ref()
            .filter(|_| self.view.show_prepared)
            .and_then(|(_, meshes)| meshes.get(index).cloned().flatten())
            .unwrap_or_else(|| loaded.clone())
    }

    /// True while prepared meshes are being computed for display or saving.
    pub fn preparing(&self) -> bool {
        self.prepared_task.is_some() || self.prepared_save.is_some()
    }

    /// Forget prepared meshes (a new document or mode).
    fn clear_prepared(&mut self) {
        self.prepared = None;
        self.prepared_task = None;
    }

    /// Start preparing the document's meshes when they are shown and the
    /// preparation options changed; pick up finished work.
    fn poll_prepared(&mut self) {
        if let Some(meshes) = self.prepared_task.as_ref().and_then(|(_, s)| s.take()) {
            let (options, _) = self.prepared_task.take().expect("task present");
            self.prepared = Some((options, meshes));
            self.scene_dirty |= self.view.show_prepared;
        }
        if self.view.show_prepared {
            let want = self.params.prep_options();
            let current = self.prepared.as_ref().map(|p| p.0);
            let running = self.prepared_task.as_ref().map(|t| t.0);
            if current != Some(want) && running != Some(want) {
                let meshes = self.doc_meshes();
                if !meshes.is_empty() {
                    self.prepared_task = Some((
                        want,
                        spawn(async move {
                            let mut y = Yielder::default();
                            let mut out = Vec::with_capacity(meshes.len());
                            for m in meshes {
                                y.yield_now().await;
                                out.push(m.map(|m| m.prepared_with(want).0));
                            }
                            out
                        }),
                    ));
                }
            }
        }
        if let Some(mesh) = self.prepared_save.as_ref().and_then(|(_, _, s)| s.take()) {
            let (name, format, _) = self.prepared_save.take().expect("save present");
            self.saving.push(io::save(name, mesh.to_bytes(format)));
        }
    }

    /// Save the mesh as packing prepares it with the current options: the
    /// object, or the selected robot collision.
    pub fn save_prepared(&mut self) {
        let format = self.view.mesh_format;
        let (stem, mesh) = match self.mode {
            Mode::Object => match &self.object {
                Some(o) => (o.name.clone(), o.mesh.clone()),
                None => return self.error("Open a mesh first."),
            },
            Mode::Robot => {
                let picked = self.robot.as_ref().and_then(|r| {
                    let i = r.selected?;
                    let c = r.doc.report.collisions.get(i)?;
                    Some((format!("{}_{}", c.link_name, c.collision_index), r.doc.meshes.get(i)?.clone()?))
                });
                match picked {
                    Some(p) => p,
                    None => return self.error("Select a mesh collision in the table first."),
                }
            }
        };
        let options = self.params.prep_options();
        let name = format!("{stem}_prepared.{}", format.extension());
        self.info(format!("Preparing {name}…"));
        self.prepared_save = Some((
            name,
            format,
            spawn(async move {
                Yielder::default().yield_now().await;
                mesh.prepared_with(options).0
            }),
        ));
    }

    /// Start loading something; `what` is shown meanwhile.
    pub fn load(&mut self, what: impl Into<String>, slot: Slot<Loaded>) {
        let what = what.into();
        self.info(format!("Loading {what}…"));
        self.loading = Some((what, slot));
    }

    fn opened(&mut self, loaded: Loaded) {
        match loaded {
            Loaded::Object { name, mesh } => {
                self.info(format!("{name}: {} faces, volume {:.4e}", mesh.faces().len(), mesh.volume()));
                self.object = Some(ObjectDoc {
                    name,
                    mesh: Arc::new(*mesh),
                    result: None,
                    quality: None,
                    anchored: false,
                    total_mass: 1.0,
                });
                self.mode = Mode::Object;
            }
            Loaded::Robot(doc) => {
                let packs = doc.report.to_pack().count();
                let mut msg = format!(
                    "{}: {} ({} collisions, {packs} to pack)",
                    doc.name,
                    doc.urdf,
                    doc.report.collisions.len()
                );
                if !doc.mesh_errors.is_empty() {
                    msg +=
                        &format!("; {} meshes failed: {}", doc.mesh_errors.len(), doc.mesh_errors.join("; "));
                }
                self.message = Some((msg, !doc.mesh_errors.is_empty()));
                self.robot = Some(RobotState {
                    doc: *doc,
                    results: HashMap::new(),
                    assembled: None,
                    quality: None,
                    selected: None,
                });
                self.mode = Mode::Robot;
            }
            Loaded::Error(e) => self.error(e),
            Loaded::Nothing => self.message = None,
        }
        self.live = None;
        self.clear_prepared();
        self.scene_dirty = true;
    }

    pub fn set_mode(&mut self, mode: Mode) {
        if self.mode != mode {
            self.mode = mode;
            self.clear_prepared();
            self.scene_dirty = true;
        }
    }

    /// Colors of the sphere groups: the base color for the object; per
    /// robot link a hue step, as the assembled URDF colors them.
    pub fn group_colors(&self) -> Vec<(GroupKey, [f64; 4])> {
        let [r, g, b] = self.view.color;
        let base = [r as f64, g as f64, b as f64, 1.0];
        let Some(robot) = &self.robot else { return vec![(None, base)] };
        let mut links: Vec<&str> = Vec::new();
        for c in robot.doc.report.to_pack() {
            if !links.contains(&c.link_name.as_str()) {
                links.push(&c.link_name);
            }
        }
        let n = links.len();
        let mut out = vec![(None, base)];
        for c in robot.doc.report.to_pack() {
            let i = links.iter().position(|l| *l == c.link_name).unwrap_or(0);
            let color = if n > 1 && self.view.color_variation > 0.0 {
                vary_color(base, (i as f64 / (n - 1) as f64) * 2.0 - 1.0, self.view.color_variation as f64)
            } else {
                base
            };
            out.push((Some((c.link_name.clone(), c.collision_index)), color));
        }
        out
    }

    /// The spheres a group shows, with a signature of their source.
    pub fn spheres_for(&self, key: Option<&(String, usize)>) -> Option<(u64, Vec<[f64; 3]>, Vec<f64>)> {
        if let (Some(live), true) = (&self.live, self.job_running())
            && self.job_mode == self.mode
            && self.job_items.get(live.item).map(Option::as_ref) == Some(key)
            && matches!(live.phase, Phase::Running | Phase::Paused)
        {
            return Some(((self.job_version << 1) | 1, live.centers.clone(), live.radii.clone()));
        }
        let result = match key {
            None => self.object.as_ref().and_then(|o| o.result.as_ref()),
            Some(k) => self.robot.as_ref().and_then(|r| r.results.get(k)),
        }?;
        Some((self.results_version << 1, result.centers.clone(), result.radii.clone()))
    }

    // -----------------------------------------------------------------
    // Actions
    // -----------------------------------------------------------------

    pub fn start_pack(&mut self) {
        let params = self.params.pack_params();
        if let Err(e) = params.validate() {
            return self.error(e.to_string());
        }
        let device = self.params.device.clone();
        let spec = match self.mode {
            Mode::Object => {
                let Some(doc) = &mut self.object else { return };
                let mut config = match params.config(&device, "", &format!("{}.json", doc.name)) {
                    Ok(c) => c,
                    Err(e) => return self.error(e.to_string()),
                };
                config.model.mesh_path = doc.name.clone();
                doc.result = None;
                doc.quality = None;
                self.job_items = vec![None];
                JobSpec::Object { mesh: doc.mesh.clone(), config: Box::new(config) }
            }
            Mode::Robot => {
                let Some(r) = &mut self.robot else { return };
                // Pack the selected item, or everything not packed yet.
                let items: Vec<_> = match r.selected {
                    Some(i) if r.doc.report.collisions.get(i).is_some_and(|c| c.action == Action::Pack) => {
                        vec![r.doc.report.collisions[i].clone()]
                    }
                    _ => r
                        .doc
                        .report
                        .collisions
                        .iter()
                        .zip(&r.doc.meshes)
                        .filter(|(c, m)| c.action == Action::Pack && m.is_some())
                        .filter(|(c, _)| !r.results.contains_key(&(c.link_name.clone(), c.collision_index)))
                        .map(|(c, _)| c.clone())
                        .collect(),
                };
                if items.is_empty() {
                    return self.info("Every link is packed. Select a row to re-pack it.");
                }
                r.assembled = None;
                r.quality = None;
                self.job_items =
                    items.iter().map(|c| Some((c.link_name.clone(), c.collision_index))).collect();
                JobSpec::Robot { pkg: r.doc.pkg.clone(), items, params, device }
            }
        };
        self.job_mode = self.mode;
        self.job_finished_seen = 0;
        self.live = None;
        self.job = Some(job::start(spec));
        self.info("Packing…");
    }

    pub fn toggle_pause(&mut self) {
        if let Some(j) = &self.job {
            j.pause.fetch_xor(true, Ordering::Relaxed);
        }
    }

    pub fn paused(&self) -> bool {
        self.job.as_ref().is_some_and(|j| j.pause.load(Ordering::Relaxed))
    }

    pub fn cancel(&mut self) {
        if let Some(j) = &self.job {
            j.cancel.store(true, Ordering::Relaxed);
            j.pause.store(false, Ordering::Relaxed);
        }
    }

    pub fn finish_now(&mut self) {
        if let Some(j) = &self.job {
            j.finish_now.store(true, Ordering::Relaxed);
        }
    }

    pub fn clear_results(&mut self) {
        match self.mode {
            Mode::Object => {
                if let Some(o) = &mut self.object {
                    o.result = None;
                    o.quality = None;
                }
            }
            Mode::Robot => {
                if let Some(r) = &mut self.robot {
                    match r.selected {
                        Some(i) => {
                            let c = &r.doc.report.collisions[i];
                            r.results.remove(&(c.link_name.clone(), c.collision_index));
                        }
                        None => r.results.clear(),
                    }
                    r.assembled = None;
                    r.quality = None;
                }
            }
        }
        self.results_version += 1;
        self.spheres_dirty = true;
    }

    pub fn assemble(&mut self) {
        let Some(r) = &mut self.robot else { return };
        let text = String::from_utf8_lossy(r.doc.pkg.read(&r.doc.urdf).unwrap_or_default()).into_owned();
        let spheres = MemSpheres(
            r.results.iter().map(|(k, res)| (k.clone(), (res.centers.clone(), res.radii.clone()))).collect(),
        );
        let [cr, cg, cb] = self.view.color;
        let base = [cr as f64, cg as f64, cb as f64, 1.0];
        match rewrite_urdf_text(&text, &r.doc.report, &spheres, base, self.view.color_variation as f64) {
            Ok((urdf, stats)) => {
                let mut summary = format!(
                    "{} sphere links on {} links; {} primitive and {} sphere collisions removed",
                    stats.sphere_children_added,
                    stats.links_with_collisions_replaced,
                    stats.primitive_collisions_removed,
                    stats.sphere_collisions_removed
                );
                let skipped = stats.skipped_summary();
                if !skipped.is_empty() {
                    summary += &format!("; not packed: {skipped}");
                }
                r.assembled = Some((urdf, summary.clone()));
                self.info(summary);
            }
            Err(e) => self.error(e.to_string()),
        }
    }

    pub fn analyze(&mut self) {
        match self.mode {
            Mode::Object => {
                let Some(o) = &self.object else { return };
                let Some(r) = o.result.clone() else { return };
                let mesh = o.mesh.clone();
                self.quality_task = Some(spawn(async move {
                    Yielder::default().yield_now().await;
                    // Score against the mesh the packing used.
                    let (prepared, _) = mesh.for_model(&r.config.model);
                    let centers: Vec<DVec3> = r.centers.iter().map(|c| DVec3::from_array(*c)).collect();
                    let masses = r.per_sphere_mass.then_some(r.masses.as_slice());
                    QualityOut::Object(evaluate_packing(
                        &prepared,
                        &centers,
                        &r.radii,
                        masses,
                        &QualityOptions::default(),
                    ))
                }));
            }
            Mode::Robot => {
                let Some(r) = &self.robot else { return };
                let mut work = Vec::new();
                for (c, m) in r.doc.report.collisions.iter().zip(&r.doc.meshes) {
                    if let (Some(m), Some(res)) =
                        (m, r.results.get(&(c.link_name.clone(), c.collision_index)))
                    {
                        work.push((
                            c.link_name.clone(),
                            c.collision_index,
                            m.clone(),
                            res.centers.clone(),
                            res.radii.clone(),
                            MeshPrepOptions::from(&res.config.model),
                        ));
                    }
                }
                if work.is_empty() {
                    return self.info("Nothing packed yet.");
                }
                self.quality_task = Some(spawn(async move {
                    let mut y = Yielder::default();
                    let mut links = Vec::new();
                    for (link, idx, mesh, centers, radii, prep) in work {
                        y.yield_now().await;
                        // Score against the mesh the packing used.
                        let (mesh, _) = mesh.prepared_with(prep);
                        let q = quality_metrics(&mesh, &link, idx, &centers, &radii);
                        links.push(q);
                    }
                    let overall = aggregate_overall(&links);
                    QualityOut::Robot(links, overall)
                }));
            }
        }
        self.info("Analyzing…");
    }

    /// Text and file name of an export.
    pub fn export(&self, what: Export) -> Result<(String, String), String> {
        match what {
            Export::ResultJson => {
                let o = self.object.as_ref().ok_or("no object")?;
                let r = o.result.as_ref().ok_or("not packed")?;
                Ok((format!("{}.spheres.json", o.name), r.to_json_string()))
            }
            Export::Urdf | Export::Mjcf => {
                let o = self.object.as_ref().ok_or("no object")?;
                let r = o.result.as_ref().ok_or("not packed")?;
                let [cr, cg, cb] = self.view.color;
                let opts = ObjectModelOptions {
                    robot_name: o.name.clone(),
                    color_rgba: [cr as f64, cg as f64, cb as f64, 1.0],
                    total_mass: o.total_mass,
                    anchored: o.anchored,
                    ..Default::default()
                };
                let (write, ext): (fn(_, _, _) -> _, _) = match what {
                    Export::Urdf => (write_object_urdf, "urdf"),
                    _ => (write_object_mjcf, "xml"),
                };
                let m = write(&r.centers, &r.radii, &opts).map_err(|e| e.to_string())?;
                Ok((format!("{}.{ext}", o.name), m.text))
            }
            Export::SphericalUrdf => {
                let r = self.robot.as_ref().ok_or("no robot")?;
                let (text, _) = r.assembled.as_ref().ok_or("assemble first")?;
                let stem =
                    std::path::Path::new(&r.doc.urdf).file_stem().map(|s| s.to_string_lossy().into_owned());
                Ok((format!("{}_spherical.urdf", stem.unwrap_or_else(|| "robot".into())), text.clone()))
            }
            Export::LinkJson(i) => {
                let r = self.robot.as_ref().ok_or("no robot")?;
                let c = r.doc.report.collisions.get(i).ok_or("no such item")?;
                let res = r.results.get(&(c.link_name.clone(), c.collision_index)).ok_or("not packed")?;
                let name = morphit_robot::pack::json_filename(&c.link_name, c.collision_index);
                Ok((name, res.to_json_string()))
            }
        }
    }

    pub fn save(&mut self, what: Export) {
        match self.export(what) {
            Ok((name, text)) => self.saving.push(io::save(name, text.into_bytes())),
            Err(e) => self.error(e),
        }
    }

    pub fn switch_urdf(&mut self, urdf: String) {
        if let Some(r) = &self.robot {
            let slot = io::switch_urdf(&r.doc, &urdf);
            self.load(urdf, slot);
        }
    }

    /// Pick up finished background work; called every frame.
    pub fn poll(&mut self) {
        self.poll_prepared();
        if let Some(d) = self.devices_task.as_ref().and_then(Slot::take) {
            self.devices = d;
            self.devices_task = None;
        }
        if let Some(loaded) = self.loading.as_ref().and_then(|(_, s)| s.take()) {
            self.loading = None;
            self.opened(loaded);
            if std::mem::take(&mut self.pack_after_load) {
                self.start_pack();
            }
        }
        let mut notes = Vec::new();
        self.saving.retain(|s| match s.take() {
            Some(r) => {
                notes.push(r);
                false
            }
            None => true,
        });
        for n in notes {
            match n {
                Ok(Some(path)) => self.info(format!("Saved {path}")),
                Ok(None) => {}
                Err(e) => self.error(e),
            }
        }
        if let Some(q) = self.quality_task.as_ref().and_then(Slot::take) {
            self.quality_task = None;
            match q {
                QualityOut::Object(m) => {
                    if let Some(o) = &mut self.object {
                        o.quality = Some(m);
                    }
                    self.info("Analysis done.");
                }
                QualityOut::Robot(links, overall) => {
                    if let Some(r) = &mut self.robot {
                        r.quality = Some((links, overall));
                    }
                    self.info("Analysis done.");
                }
                QualityOut::Error(e) => self.error(e),
            }
        }
        self.poll_job();
    }

    fn poll_job(&mut self) {
        let Some(job) = self.job.clone() else { return };
        let v = job.version();
        if v == self.job_version {
            return;
        }
        self.job_version = v;
        let st = job.status();
        for (key, result) in st.finished.iter().skip(self.job_finished_seen) {
            match (key, self.job_mode) {
                (None, Mode::Object) => {
                    if let Some(o) = &mut self.object {
                        o.result = Some(result.clone());
                    }
                }
                (Some(k), Mode::Robot) => {
                    if let Some(r) = &mut self.robot {
                        r.results.insert(k.clone(), result.clone());
                    }
                }
                _ => {}
            }
            self.results_version += 1;
        }
        self.job_finished_seen = st.finished.len();
        match st.phase {
            Phase::Done => self.info(format!(
                "Packed {} in {} iterations on {}.",
                self.job_label(&st),
                st.iteration,
                st.device
            )),
            Phase::Cancelled => self.info("Cancelled."),
            Phase::Failed => self.error(st.error.clone().unwrap_or_else(|| "failed".into())),
            _ => {}
        }
        self.live = Some(st);
        self.spheres_dirty = true;
    }

    fn job_label(&self, st: &JobStatus) -> String {
        match self.job_mode {
            Mode::Object => self.object.as_ref().map_or("object".into(), |o| o.name.clone()),
            Mode::Robot => format!("{} collision meshes", st.items),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub enum Export {
    ResultJson,
    Urdf,
    Mjcf,
    SphericalUrdf,
    LinkJson(usize),
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;
    use morphit::glam::DVec3;

    /// Poll until `done` holds (the preparation runs on a thread).
    fn poll_until(s: &mut Studio, done: impl Fn(&Studio) -> bool) {
        let t0 = std::time::Instant::now();
        while !done(s) {
            assert!(t0.elapsed().as_secs() < 60, "preparation did not finish");
            std::thread::sleep(std::time::Duration::from_millis(10));
            s.poll();
        }
    }

    #[test]
    fn prepared_meshes_are_shown_on_request() {
        // A concave body: the union of two overlapping boxes, whose hull is larger.
        let a = morphit::shapes::box_mesh(DVec3::ZERO, DVec3::ONE);
        let b = morphit::shapes::box_mesh(DVec3::new(0.4, 0.2, 0.1), DVec3::new(1.4, 1.2, 1.1));
        let mut v = a.vertices().to_vec();
        let mut f = a.faces().to_vec();
        let base = v.len() as u32;
        v.extend_from_slice(b.vertices());
        f.extend(b.faces().iter().map(|t| t.map(|i| i + base)));
        let (union, _) = Arc::new(Mesh::from_vertices(v, f, None).unwrap()).prepared();
        let body = Mesh::from_vertices(union.vertices().to_vec(), union.faces().to_vec(), None).unwrap();

        let mut s = Studio::default();
        s.opened(Loaded::Object { name: "l-shape".into(), mesh: Box::new(body) });
        let loaded = s.object.as_ref().unwrap().mesh.clone();
        s.scene_dirty = false;
        s.poll();
        assert!(!s.preparing(), "nothing is prepared while the toggle is off");
        assert!(Arc::ptr_eq(&s.shown_mesh(0, &loaded), &loaded));

        // Default preparation: a single body stays as it is.
        s.view.show_prepared = true;
        poll_until(&mut s, |s| s.prepared.is_some() && !s.preparing());
        assert!(s.scene_dirty);
        assert!(Arc::ptr_eq(&s.shown_mesh(0, &loaded), &loaded));

        // Convex hull: prepared again and drawn instead.
        s.scene_dirty = false;
        s.params.convex_hull = true;
        s.poll();
        assert!(s.preparing());
        poll_until(&mut s, |s| !s.preparing());
        assert!(s.scene_dirty);
        let hull = s.shown_mesh(0, &loaded);
        assert!(hull.volume() > loaded.volume() + 0.05, "{} vs {}", hull.volume(), loaded.volume());

        // Off again: the loaded mesh.
        s.view.show_prepared = false;
        assert!(Arc::ptr_eq(&s.shown_mesh(0, &loaded), &loaded));
    }
}
