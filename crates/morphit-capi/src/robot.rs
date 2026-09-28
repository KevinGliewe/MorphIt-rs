//! The robot pipeline: a URDF package in memory, inspected, packed link by
//! link (each link is a normal session) and assembled into a spherical URDF.

use std::ffi::{c_char, c_int};
use std::path::Path;
use std::ptr::null_mut;
use std::sync::{Arc, Mutex, MutexGuard};

use ::morphit_robot::assemble::{MemSpheres, load_spheres_str, rewrite_urdf_text};
use ::morphit_robot::color::safe_color_rgba;
use ::morphit_robot::config::{PackParams, parse_advanced};
use ::morphit_robot::inspect::{CollisionItem, InspectionReport, inspect_urdf_in, select_urdf_in};
use ::morphit_robot::pack::{json_filename, link_config, pack_mesh_path};
use ::morphit_robot::vfs::MemPackage;
use morphit::Session;

use crate::ffi::morphit_status::*;
use crate::ffi::{FfiError, FfiResult, cstr, guard, out, write_string};
use crate::handles::morphit_session;
use crate::morphit_status;

/// A robot description package (URDF plus meshes) held in memory, and the
/// sphere results recorded for its links. Locks internally; may be used from
/// several threads.
pub struct morphit_robot {
    pkg: Mutex<Arc<MemPackage>>,
    spheres: Mutex<MemSpheres>,
}

impl morphit_robot {
    fn new(pkg: MemPackage) -> Self {
        morphit_robot { pkg: Mutex::new(Arc::new(pkg)), spheres: Mutex::default() }
    }

    fn pkg(&self) -> Arc<MemPackage> {
        Arc::clone(&self.pkg.lock().unwrap_or_else(|p| p.into_inner()))
    }

    fn spheres(&self) -> MutexGuard<'_, MemSpheres> {
        self.spheres.lock().unwrap_or_else(|p| p.into_inner())
    }
}

/// The inspection of one URDF: every `<collision>` and what to do with it.
/// Immutable. The "pack items" are the collisions to replace with spheres,
/// numbered `0 .. morphit_report_pack_count()`.
pub struct morphit_robot_report {
    report: Arc<InspectionReport>,
    pack: Vec<CollisionItem>,
}

/// Parameters of `morphit_robot_pack_link` (the web API's pack request). Fill
/// with `morphit_pack_params_default`, then change fields.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct morphit_pack_params {
    /// "MorphIt-V", "MorphIt-S" or "MorphIt-B"; NULL means MorphIt-B.
    pub variant: *const c_char,
    /// 1 to 200.
    pub num_spheres: usize,
    /// 1 to 1000.
    pub iterations: usize,
    /// Random seed; negative draws one at random.
    pub seed: i64,
    /// Mesh preparation: merge overlapping bodies (nonzero = on).
    pub union_overlapping_bodies: c_int,
    /// Mesh preparation: convex hull of each body first (nonzero = on).
    pub convex_hull: c_int,
    /// The web UI's advanced overrides as a JSON object (e.g.
    /// `{"coverage_weight": 2000}`), or NULL.
    pub advanced_json: *const c_char,
}

/// What `morphit_robot_assemble` changed.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct morphit_assemble_stats {
    pub links_with_collisions_replaced: usize,
    pub mesh_collisions_replaced: usize,
    pub primitive_collisions_removed: usize,
    pub sphere_collisions_removed: usize,
    pub sphere_children_added: usize,
}

unsafe fn robot_ref<'a>(p: *const morphit_robot) -> FfiResult<&'a morphit_robot> {
    unsafe { p.as_ref() }.ok_or_else(|| FfiError::null("robot"))
}

unsafe fn report_ref<'a>(p: *const morphit_robot_report) -> FfiResult<&'a morphit_robot_report> {
    unsafe { p.as_ref() }.ok_or_else(|| FfiError::null("report"))
}

fn item(r: &morphit_robot_report, index: usize) -> FfiResult<&CollisionItem> {
    r.pack.get(index).ok_or_else(|| {
        FfiError::invalid(format!("pack item {index} out of range (the report has {})", r.pack.len()))
    })
}

fn put(o: &mut *mut morphit_robot, pkg: MemPackage) -> FfiResult {
    *o = Box::into_raw(Box::new(morphit_robot::new(pkg)));
    Ok(MORPHIT_OK)
}

/// An empty package; add files with `morphit_robot_add_file`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_robot_new(out_robot: *mut *mut morphit_robot) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_robot, "out") }?;
        put(o, MemPackage::new())
    })
}

/// Every file under the folder `path` (the package root).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_robot_from_folder(
    path: *const c_char,
    out_robot: *mut *mut morphit_robot,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_robot, "out") }?;
        *o = null_mut();
        let path = unsafe { cstr(path, "path") }?;
        if !Path::new(path).is_dir() {
            return Err(FfiError::new(MORPHIT_ERR_IO, format!("not a folder: {path}")));
        }
        put(o, MemPackage::from_dir(Path::new(path))?)
    })
}

/// A package from the bytes of a `.zip` archive.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_robot_from_zip(
    data: *const u8,
    len: usize,
    out_robot: *mut *mut morphit_robot,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_robot, "out") }?;
        *o = null_mut();
        if data.is_null() {
            return Err(FfiError::null("data"));
        }
        // SAFETY: the caller provides `len` readable bytes.
        put(o, MemPackage::from_zip(unsafe { std::slice::from_raw_parts(data, len) })?)
    })
}

/// Add (or replace) a file under its path relative to the package root.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_robot_add_file(
    robot: *mut morphit_robot,
    path: *const c_char,
    data: *const u8,
    len: usize,
) -> morphit_status {
    guard(|| {
        let r = unsafe { robot_ref(robot) }?;
        let path = unsafe { cstr(path, "path") }?;
        if data.is_null() && len > 0 {
            return Err(FfiError::null("data"));
        }
        // SAFETY: the caller provides `len` readable bytes.
        let bytes =
            if len == 0 { Vec::new() } else { unsafe { std::slice::from_raw_parts(data, len) }.to_vec() };
        let mut pkg = r.pkg.lock().unwrap_or_else(|p| p.into_inner());
        Arc::make_mut(&mut pkg).insert(path, bytes)?;
        Ok(MORPHIT_OK)
    })
}

/// Release a package. NULL is ignored. Sessions created from it stay valid.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_robot_free(robot: *mut morphit_robot) {
    if !robot.is_null() {
        let _ = std::panic::catch_unwind(|| drop(unsafe { Box::from_raw(robot) }));
    }
}

/// Inspect a URDF of the package: `urdf` selects one by file name, NULL
/// takes the only one. `*out` receives a report to release with
/// `morphit_report_free`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_robot_inspect(
    robot: *const morphit_robot,
    urdf: *const c_char,
    out_report: *mut *mut morphit_robot_report,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_report, "out") }?;
        *o = null_mut();
        let r = unsafe { robot_ref(robot) }?;
        let urdf = if urdf.is_null() { None } else { Some(unsafe { cstr(urdf, "urdf") }?) };
        let pkg = r.pkg();
        let key = select_urdf_in(&pkg, urdf)?;
        let report = inspect_urdf_in(&pkg, &key)?;
        let pack = report.to_pack().cloned().collect();
        *o = Box::into_raw(Box::new(morphit_robot_report { report: Arc::new(report), pack }));
        Ok(MORPHIT_OK)
    })
}

/// The full inspection report as JSON (the web API's `/api/robot/inspect`
/// response: `urdf_path`, `collisions`, `warnings`, ...).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_report_json(
    report: *const morphit_robot_report,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
) -> morphit_status {
    guard(|| {
        let r = unsafe { report_ref(report) }?;
        let json = serde_json::to_string_pretty(&*r.report).expect("report serializes");
        unsafe { write_string(&json, buf, capacity, needed) }
    })
}

/// Number of collisions to pack.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_report_pack_count(
    report: *const morphit_robot_report,
    out_count: *mut usize,
) -> morphit_status {
    guard(|| {
        let r = unsafe { report_ref(report) }?;
        *unsafe { out(out_count, "out") }? = r.pack.len();
        Ok(MORPHIT_OK)
    })
}

/// Pack item `index`: its link name (string buffer rules) and the index of
/// the collision within the link (`collision_index`, may be NULL).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_report_pack_item(
    report: *const morphit_robot_report,
    index: usize,
    link_buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
    collision_index: *mut usize,
) -> morphit_status {
    guard(|| {
        let it = item(unsafe { report_ref(report) }?, index)?;
        if let Some(ci) = unsafe { collision_index.as_mut() } {
            *ci = it.collision_index;
        }
        unsafe { write_string(&it.link_name, link_buf, capacity, needed) }
    })
}

/// Release a report. NULL is ignored.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_report_free(report: *mut morphit_robot_report) {
    if !report.is_null() {
        let _ = std::panic::catch_unwind(|| drop(unsafe { Box::from_raw(report) }));
    }
}

/// Fill `*out` with the web API's defaults: MorphIt-B, 20 spheres, 200
/// iterations, random seed, union on, convex hull off.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_pack_params_default(out_params: *mut morphit_pack_params) -> morphit_status {
    guard(|| {
        *unsafe { out(out_params, "out") }? = morphit_pack_params {
            variant: std::ptr::null(),
            num_spheres: 20,
            iterations: 200,
            seed: -1,
            union_overlapping_bodies: 1,
            convex_hull: 0,
            advanced_json: std::ptr::null(),
        };
        Ok(MORPHIT_OK)
    })
}

unsafe fn pack_params(p: *const morphit_pack_params) -> FfiResult<PackParams> {
    let mut d = morphit_pack_params {
        variant: std::ptr::null(),
        num_spheres: 20,
        iterations: 200,
        seed: -1,
        union_overlapping_bodies: 1,
        convex_hull: 0,
        advanced_json: std::ptr::null(),
    };
    if let Some(p) = unsafe { p.as_ref() } {
        d = *p;
    }
    let variant = if d.variant.is_null() {
        "MorphIt-B".to_string()
    } else {
        unsafe { cstr(d.variant, "variant") }?.to_string()
    };
    let advanced = if d.advanced_json.is_null() {
        Vec::new()
    } else {
        parse_advanced(unsafe { cstr(d.advanced_json, "advanced_json") }?)?
    };
    let params = PackParams {
        variant,
        num_spheres: d.num_spheres,
        iterations: d.iterations,
        seed: u64::try_from(d.seed).ok(),
        advanced,
        union_overlapping_bodies: d.union_overlapping_bodies != 0,
        convex_hull: d.convex_hull != 0,
    };
    params.validate()?;
    Ok(params)
}

/// A session packing pack item `index` of `report`. `params` may be NULL for
/// the defaults; `device` may be NULL for "auto". Run it like any session,
/// then record its spheres with `morphit_robot_set_link_result`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_robot_pack_link(
    robot: *const morphit_robot,
    report: *const morphit_robot_report,
    index: usize,
    params: *const morphit_pack_params,
    device: *const c_char,
    out_session: *mut *mut morphit_session,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_session, "out") }?;
        *o = null_mut();
        let r = unsafe { robot_ref(robot) }?;
        let it = item(unsafe { report_ref(report) }?, index)?;
        let params = unsafe { pack_params(params) }?;
        let device = if device.is_null() { "auto" } else { unsafe { cstr(device, "device") }? };
        let path = pack_mesh_path(it)?;
        let (config, _) = link_config(it, &params, device, "spheres")?;
        let mesh = Arc::new(r.pkg().load_mesh(path)?);
        let session = Session::new(config, mesh)?;
        *o = Box::into_raw(Box::new(morphit_session::new(session)));
        Ok(MORPHIT_OK)
    })
}

/// Record the current spheres of `session` as the result of pack item
/// `index` (reads the snapshot; typically after the run finished).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_robot_set_link_result(
    robot: *const morphit_robot,
    report: *const morphit_robot_report,
    index: usize,
    session: *const morphit_session,
) -> morphit_status {
    guard(|| {
        let r = unsafe { robot_ref(robot) }?;
        let it = item(unsafe { report_ref(report) }?, index)?;
        let s = unsafe { session.as_ref() }.ok_or_else(|| FfiError::null("session"))?;
        let snap = s.snapshot();
        let set = (snap.centers.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect(), snap.radii.clone());
        r.spheres().0.insert((it.link_name.clone(), it.collision_index), set);
        Ok(MORPHIT_OK)
    })
}

/// Record the spheres of `link[collision_index]` from a result JSON (the
/// format of `morphit_result_json`, or a Python MorphIt result file).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_robot_set_link_result_json(
    robot: *const morphit_robot,
    link: *const c_char,
    collision_index: usize,
    result_json: *const c_char,
) -> morphit_status {
    guard(|| {
        let r = unsafe { robot_ref(robot) }?;
        let link = unsafe { cstr(link, "link") }?;
        let json = unsafe { cstr(result_json, "result_json") }?;
        let set = load_spheres_str(json, &json_filename(link, collision_index))?;
        r.spheres().0.insert((link.to_string(), collision_index), set);
        Ok(MORPHIT_OK)
    })
}

/// Forget every recorded link result.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_robot_clear_link_results(robot: *const morphit_robot) -> morphit_status {
    guard(|| {
        unsafe { robot_ref(robot) }?.spheres().0.clear();
        Ok(MORPHIT_OK)
    })
}

/// The URDF of `report` with every packed collision replaced by sphere links
/// (pack items without a recorded result keep their mesh). `base_color` is
/// "#rrggbb" or NULL; `color_variation` (0..1) spreads the hue over the links.
/// The URDF text follows the string buffer rules; `stats` may be NULL.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn morphit_robot_assemble(
    robot: *const morphit_robot,
    report: *const morphit_robot_report,
    base_color: *const c_char,
    color_variation: f64,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
    stats: *mut morphit_assemble_stats,
) -> morphit_status {
    guard(|| {
        let r = unsafe { robot_ref(robot) }?;
        let rep = unsafe { report_ref(report) }?;
        let color =
            if base_color.is_null() { None } else { Some(unsafe { cstr(base_color, "base_color") }?) };
        let pkg = r.pkg();
        let text = String::from_utf8_lossy(pkg.read(&rep.report.urdf_path).unwrap_or_default()).into_owned();
        let spheres = r.spheres();
        let variation = if color_variation.is_finite() { color_variation.clamp(0.0, 1.0) } else { 0.0 };
        let (urdf, st) = rewrite_urdf_text(&text, &rep.report, &*spheres, safe_color_rgba(color), variation)?;
        drop(spheres);
        if let Some(o) = unsafe { stats.as_mut() } {
            *o = morphit_assemble_stats {
                links_with_collisions_replaced: st.links_with_collisions_replaced,
                mesh_collisions_replaced: st.mesh_collisions_replaced,
                primitive_collisions_removed: st.primitive_collisions_removed,
                sphere_collisions_removed: st.sphere_collisions_removed,
                sphere_children_added: st.sphere_children_added,
            };
        }
        unsafe { write_string(&urdf, buf, capacity, needed) }
    })
}
