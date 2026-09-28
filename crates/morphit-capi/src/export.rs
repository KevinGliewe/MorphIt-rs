//! Simulator models from spheres (URDF, MJCF) and packing quality metrics.

use std::ffi::{c_char, c_int};
use std::ptr::null_mut;
use std::sync::Arc;

use morphit::glam::DVec3;
use morphit::{QualityMetrics, QualityOptions, evaluate_packing};
use morphit_robot::object_model::{ObjectModel, ObjectModelOptions, write_object_mjcf, write_object_urdf};

use crate::ffi::morphit_status::*;
use crate::ffi::{FfiError, FfiResult, cstr, guard, out, write_string};
use crate::handles::{morphit_mesh, morphit_session};
use crate::morphit_status;

/// Options of `morphit_object_urdf` / `morphit_object_mjcf`. Fill with
/// `morphit_object_options_default`, then change fields.
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct morphit_object_options {
    /// `<robot name>` / `<mujoco model>`; NULL means "object".
    pub name: *const c_char,
    /// Sphere color, RGBA in 0..1.
    pub rgba: [f64; 4],
    /// Total mass in kg, split over the spheres in proportion to r^3.
    pub total_mass: f64,
    /// Nonzero welds the object to the world instead of letting it float.
    pub anchored: c_int,
    /// Decimal places of every number written.
    pub decimals: c_int,
}

/// Sampling settings of `morphit_evaluate_packing` (defaults match the
/// Python `debug_quick_eval.py`).
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct morphit_quality_options {
    pub seed: u64,
    pub surface_samples: usize,
    pub volume_samples: usize,
    /// Volume samples come from the mesh bounding box scaled by this factor.
    pub bounds_expand: f64,
    /// Density used for the sphere masses when none are given.
    pub density: f64,
}

/// Quality of a packing (the `debug_quick_eval` metrics).
#[allow(non_camel_case_types)]
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct morphit_quality_metrics {
    pub actual_n: usize,
    /// Spheres whose center lies outside the mesh.
    pub n_out: usize,
    /// Spheres with radius below 0.001 x mesh scale.
    pub n_tiny: usize,
    /// Volume covered by spheres inside the mesh, relative to the mesh volume.
    pub r_in: f64,
    /// Volume covered by spheres outside the mesh, relative to the mesh volume.
    pub r_out: f64,
    /// Union volume of the spheres relative to the mesh volume.
    pub r_uni: f64,
    /// Mean absolute surface distance in millimetres (mesh units x 1000).
    pub d_avg_mm: f64,
    /// Maximum absolute surface distance in millimetres.
    pub d_max_mm: f64,
    pub mass_abs: f64,
    pub mass_rel: f64,
    pub com_abs: f64,
    pub com_rel: f64,
    pub i_abs: f64,
    pub i_rel: f64,
}

impl From<QualityMetrics> for morphit_quality_metrics {
    fn from(q: QualityMetrics) -> Self {
        morphit_quality_metrics {
            actual_n: q.actual_n,
            n_out: q.n_out,
            n_tiny: q.n_tiny,
            r_in: q.r_in,
            r_out: q.r_out,
            r_uni: q.r_uni,
            d_avg_mm: q.d_avg_mm,
            d_max_mm: q.d_max_mm,
            mass_abs: q.mass_abs,
            mass_rel: q.mass_rel,
            com_abs: q.com_abs,
            com_rel: q.com_rel,
            i_abs: q.i_abs,
            i_rel: q.i_rel,
        }
    }
}

impl From<QualityOptions> for morphit_quality_options {
    fn from(o: QualityOptions) -> Self {
        morphit_quality_options {
            seed: o.seed,
            surface_samples: o.surface_samples,
            volume_samples: o.volume_samples,
            bounds_expand: o.bounds_expand,
            density: o.density,
        }
    }
}

impl From<morphit_quality_options> for QualityOptions {
    fn from(o: morphit_quality_options) -> Self {
        QualityOptions {
            seed: o.seed,
            surface_samples: o.surface_samples,
            volume_samples: o.volume_samples,
            bounds_expand: o.bounds_expand,
            density: o.density,
        }
    }
}

/// `count` spheres: `centers` holds `3 * count` doubles, `radii` `count`.
unsafe fn spheres<'a>(
    centers: *const f64,
    radii: *const f64,
    count: usize,
) -> FfiResult<(Vec<[f64; 3]>, &'a [f64])> {
    if count == 0 {
        return Err(FfiError::invalid("count must be at least 1"));
    }
    if centers.is_null() {
        return Err(FfiError::null("centers"));
    }
    if radii.is_null() {
        return Err(FfiError::null("radii"));
    }
    let n3 = count.checked_mul(3).ok_or_else(|| FfiError::invalid("count too large"))?;
    // SAFETY: the caller provides arrays of the stated sizes.
    let (c, r) =
        unsafe { (std::slice::from_raw_parts(centers, n3), std::slice::from_raw_parts(radii, count)) };
    Ok((c.chunks_exact(3).map(|p| [p[0], p[1], p[2]]).collect(), r))
}

/// Fill `*out` with the default object-model options.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_object_options_default(
    out_options: *mut morphit_object_options,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_options, "out") }?;
        let d = ObjectModelOptions::default();
        *o = morphit_object_options {
            name: std::ptr::null(),
            rgba: d.color_rgba,
            total_mass: d.total_mass,
            anchored: d.anchored as c_int,
            decimals: d.decimals as c_int,
        };
        Ok(MORPHIT_OK)
    })
}

unsafe fn object_options(o: *const morphit_object_options) -> FfiResult<ObjectModelOptions> {
    let d = ObjectModelOptions::default();
    let Some(o) = (unsafe { o.as_ref() }) else { return Ok(d) };
    if o.decimals < 0 {
        return Err(FfiError::invalid("decimals must not be negative"));
    }
    let name =
        if o.name.is_null() { d.robot_name.clone() } else { unsafe { cstr(o.name, "name") }?.to_string() };
    Ok(ObjectModelOptions {
        robot_name: name,
        color_rgba: o.rgba,
        total_mass: o.total_mass,
        anchored: o.anchored != 0,
        decimals: o.decimals as usize,
        ..d
    })
}

type Writer = fn(&[[f64; 3]], &[f64], &ObjectModelOptions) -> morphit_robot::Result<ObjectModel>;

#[allow(clippy::too_many_arguments)]
unsafe fn write_model(
    write: Writer,
    centers: *const f64,
    radii: *const f64,
    count: usize,
    options: *const morphit_object_options,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
    centroid: *mut f64,
) -> FfiResult {
    let (c, r) = unsafe { spheres(centers, radii, count) }?;
    let opts = unsafe { object_options(options) }?;
    let m = write(&c, r, &opts)?;
    if !centroid.is_null() {
        // SAFETY: the caller provides room for three doubles.
        unsafe { std::ptr::copy_nonoverlapping(m.centroid.as_ptr(), centroid, 3) };
    }
    unsafe { write_string(&m.text, buf, capacity, needed) }
}

/// The spheres as a URDF: one link per sphere on fixed joints, masses split
/// by volume (as the Python `create_object_urdf.py`). `centers` holds
/// `3 * count` doubles; `options` may be NULL for the defaults. `centroid`
/// (NULL or room for 3 doubles) receives the point the positions are relative
/// to. String buffer rules as everywhere.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn morphit_object_urdf(
    centers: *const f64,
    radii: *const f64,
    count: usize,
    options: *const morphit_object_options,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
    centroid: *mut f64,
) -> morphit_status {
    guard(|| unsafe {
        write_model(write_object_urdf, centers, radii, count, options, buf, capacity, needed, centroid)
    })
}

/// The spheres as MJCF for MuJoCo: a body with one sphere geom per sphere.
/// Arguments as for `morphit_object_urdf`.
#[unsafe(no_mangle)]
#[allow(clippy::too_many_arguments)]
pub unsafe extern "C" fn morphit_object_mjcf(
    centers: *const f64,
    radii: *const f64,
    count: usize,
    options: *const morphit_object_options,
    buf: *mut c_char,
    capacity: usize,
    needed: *mut usize,
    centroid: *mut f64,
) -> morphit_status {
    guard(|| unsafe {
        write_model(write_object_mjcf, centers, radii, count, options, buf, capacity, needed, centroid)
    })
}

/// Fill `*out` with the default quality options.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_quality_options_default(
    out_options: *mut morphit_quality_options,
) -> morphit_status {
    guard(|| {
        *unsafe { out(out_options, "out") }? = QualityOptions::default().into();
        Ok(MORPHIT_OK)
    })
}

fn quality_options(o: *const morphit_quality_options) -> QualityOptions {
    unsafe { o.as_ref() }.map_or_else(QualityOptions::default, |o| (*o).into())
}

/// Score `count` spheres against `mesh` (pass the prepared mesh the spheres
/// were packed on, e.g. from `morphit_session_mesh`). `masses` may be NULL to
/// derive masses from the density; `options` may be NULL for the defaults.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_evaluate_packing(
    mesh: *const morphit_mesh,
    centers: *const f64,
    radii: *const f64,
    masses: *const f64,
    count: usize,
    options: *const morphit_quality_options,
    out_metrics: *mut morphit_quality_metrics,
) -> morphit_status {
    guard(|| {
        let m = &unsafe { mesh.as_ref() }.ok_or_else(|| FfiError::null("mesh"))?.mesh;
        let o = unsafe { out(out_metrics, "out") }?;
        let (c, r) = unsafe { spheres(centers, radii, count) }?;
        let c: Vec<DVec3> = c.into_iter().map(DVec3::from_array).collect();
        // SAFETY: when given, `masses` holds `count` doubles.
        let masses = (!masses.is_null()).then(|| unsafe { std::slice::from_raw_parts(masses, count) });
        *o = evaluate_packing(m, &c, r, masses, &quality_options(options)).into();
        Ok(MORPHIT_OK)
    })
}

/// Score a session's current spheres against its prepared mesh. Reads the
/// snapshot, so it may be called while the session runs on another thread.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_session_evaluate(
    session: *const morphit_session,
    options: *const morphit_quality_options,
    out_metrics: *mut morphit_quality_metrics,
) -> morphit_status {
    guard(|| {
        let s = unsafe { session.as_ref() }.ok_or_else(|| FfiError::null("session"))?;
        let o = unsafe { out(out_metrics, "out") }?;
        let snap = s.snapshot();
        let centers: Vec<DVec3> =
            snap.centers.chunks_exact(3).map(|c| DVec3::new(c[0], c[1], c[2])).collect();
        let masses = snap.per_sphere_mass.then_some(snap.masses.as_slice());
        *o = evaluate_packing(&s.mesh, &centers, &snap.radii, masses, &quality_options(options)).into();
        Ok(MORPHIT_OK)
    })
}

/// The mesh a session packs (after mesh preparation), as a new handle to be
/// released with `morphit_mesh_free`. Never waits for a running iteration.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn morphit_session_mesh(
    session: *const morphit_session,
    out_mesh: *mut *mut morphit_mesh,
) -> morphit_status {
    guard(|| {
        let o = unsafe { out(out_mesh, "out") }?;
        *o = null_mut();
        let s = unsafe { session.as_ref() }.ok_or_else(|| FfiError::null("session"))?;
        *o = Box::into_raw(Box::new(morphit_mesh { mesh: Arc::clone(&s.mesh) }));
        Ok(MORPHIT_OK)
    })
}
