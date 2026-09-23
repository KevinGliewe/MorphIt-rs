//! Per-session GPU buffers, dispatch, readback and the f64 decision step.
//!
//! # Why the results are bit-identical to the CPU
//!
//! The kernels work in f32 on coordinates shifted by the sample bounding-box
//! centre. Let `u = 2^-24` (f32 unit roundoff), `B` the largest |coordinate|
//! after the shift over the samples and centers of a call, `R` the largest
//! radius. For one (sample, sphere) pair the f32 gap `|p - c| - r` differs
//! from the exact value by at most about `30 u B + u R` (rounding both points
//! and their difference: `7 u B`; squares, sums and a `sqrt` of at most 2.5
//! ulp: `23 u B`; rounding `r`: `u R`). Two gaps therefore differ from their
//! exact values by less than `2 e <= 62 u (B + R)`.
//!
//! The kernel reports, per row, the f32 winner and the f32 margin to the
//! runner-up. If the exact winner were a different index, the f32 margin
//! would be below `2 e`; so whenever `margin > TOL = 256 u (B + R)` the f32
//! winner is the unique exact winner, and the CPU only recomputes its
//! distance with the same f64 [`dist`] the CPU backend uses. Every other row
//! (near ties, exact ties, non-finite values) is rescanned on the CPU with
//! the CPU backend's own code. Boundary candidates use the same bound:
//! everything with `r - |p - c| > -TOL` is listed and the loss re-tests each
//! pair exactly.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use crate::par::*;
use bytemuck::{Pod, Zeroable};
use glam::DVec3;

use super::GpuContext;
use crate::distances::{self, RowMin, dist};
use crate::loss::Samples;
use crate::search::{BoundaryPairs, QueryResults, SampleSet, Wants, cpu_boundary_pairs};

/// `256 u = 128 * f32::EPSILON`; see the module documentation.
const TOL_FACTOR: f64 = 128.0 * f32::EPSILON as f64;
const WORKGROUP: u32 = 256;
const MAX_WORKGROUPS: usize = 65_535;
const INITIAL_PAIRS_PER_SAMPLE: usize = 4;

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Params {
    n_samples: u32,
    n_spheres: u32,
    want_pairs: u32,
    pair_capacity: u32,
    tol: f32,
    _pad: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable, Debug)]
struct RowOut {
    idx: u32,
    margin: f32,
}

struct SampleBuf {
    buf: wgpu::Buffer,
    len: usize,
    /// Largest |coordinate| after the shift.
    abs_max: f64,
}

struct Scratch {
    spheres: wgpu::Buffer,
    spheres_cap: usize,
    out_inside: wgpu::Buffer,
    out_surface: wgpu::Buffer,
    out_spheres: wgpu::Buffer,
    out_spheres_cap: usize,
    /// `[count, pad, (i, j)...]`; see `PairBuf` in the shader.
    pairs: wgpu::Buffer,
    /// Capacity in pairs.
    pairs_cap: usize,
    /// One uniform buffer per pass: inside rows, surface rows, spheres.
    params: [wgpu::Buffer; 3],
    staging: wgpu::Buffer,
    staging_cap: u64,
}

/// Counters for tests and diagnostics.
#[derive(Default)]
pub(crate) struct Stats {
    /// Rows decided from the GPU proposal alone.
    pub fast: AtomicUsize,
    /// Rows rescanned on the CPU because the margin was too small.
    pub rescanned: AtomicUsize,
    /// Pair passes repeated because the pair buffer was too small.
    pub pair_reruns: AtomicUsize,
    /// Calls answered entirely by the CPU (GPU error or non-finite input).
    pub cpu_calls: AtomicUsize,
}

pub(crate) struct SessionSearch {
    ctx: Arc<GpuContext>,
    origin: DVec3,
    inside: SampleBuf,
    surface: SampleBuf,
    scratch: Mutex<Scratch>,
    /// Sticky after a GPU failure: every later call runs on the CPU.
    broken: AtomicBool,
    /// Why `broken` was set.
    error: Mutex<Option<String>>,
    pub(crate) stats: Stats,
}

fn storage(device: &wgpu::Device, label: &str, bytes: u64, extra: wgpu::BufferUsages) -> wgpu::Buffer {
    device.create_buffer(&wgpu::BufferDescriptor {
        label: Some(label),
        size: bytes.max(16),
        usage: wgpu::BufferUsages::STORAGE | extra,
        mapped_at_creation: false,
    })
}

fn pair_buffer(device: &wgpu::Device, cap: usize) -> wgpu::Buffer {
    storage(
        device,
        "pairs",
        8 + (cap * 8) as u64,
        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::COPY_SRC,
    )
}

fn upload_samples(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    label: &str,
    pts: &[DVec3],
    origin: DVec3,
) -> SampleBuf {
    let mut abs_max: f64 = 0.0;
    let data: Vec<[f32; 4]> = pts
        .iter()
        .map(|&p| {
            let q = p - origin;
            abs_max = abs_max.max(q.abs().max_element());
            [q.x as f32, q.y as f32, q.z as f32, 0.0]
        })
        .collect();
    let buf = storage(device, label, (data.len() * 16) as u64, wgpu::BufferUsages::COPY_DST);
    if !data.is_empty() {
        queue.write_buffer(&buf, 0, bytemuck::cast_slice(&data));
    }
    SampleBuf { buf, len: pts.len(), abs_max }
}

impl SessionSearch {
    pub(crate) fn new(ctx: Arc<GpuContext>, samples: &Samples) -> SessionSearch {
        let all = samples.inside.iter().chain(&samples.surface);
        let (lo, hi) =
            all.fold((DVec3::INFINITY, DVec3::NEG_INFINITY), |(lo, hi), &p| (lo.min(p), hi.max(p)));
        let origin = if lo.is_finite() && hi.is_finite() { (lo + hi) * 0.5 } else { DVec3::ZERO };
        let d = &ctx.device;
        let inside = upload_samples(d, &ctx.queue, "inside samples", &samples.inside, origin);
        let surface = upload_samples(d, &ctx.queue, "surface samples", &samples.surface, origin);
        let pairs_cap = (samples.surface.len() * INITIAL_PAIRS_PER_SAMPLE).max(1);
        let uniform = || {
            d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("params"),
                size: std::mem::size_of::<Params>() as u64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            })
        };
        let scratch = Scratch {
            spheres: storage(d, "spheres", 16, wgpu::BufferUsages::COPY_DST),
            spheres_cap: 1,
            out_inside: storage(d, "out inside", (inside.len * 8) as u64, wgpu::BufferUsages::COPY_SRC),
            out_surface: storage(d, "out surface", (surface.len * 8) as u64, wgpu::BufferUsages::COPY_SRC),
            out_spheres: storage(d, "out spheres", 8, wgpu::BufferUsages::COPY_SRC),
            out_spheres_cap: 1,
            pairs: pair_buffer(d, pairs_cap),
            pairs_cap,
            params: [uniform(), uniform(), uniform()],
            staging: d.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: 16,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            staging_cap: 16,
        };
        SessionSearch {
            ctx,
            origin,
            inside,
            surface,
            scratch: Mutex::new(scratch),
            broken: AtomicBool::new(false),
            error: Mutex::new(None),
            stats: Stats::default(),
        }
    }

    /// Shrink the pair buffer (tests exercise the regrow path).
    #[cfg(test)]
    pub(crate) fn set_pair_capacity(&self, cap: usize) {
        let mut s = self.scratch.lock().unwrap_or_else(|e| e.into_inner());
        s.pairs_cap = cap.max(1);
        s.pairs = pair_buffer(&self.ctx.device, s.pairs_cap);
    }

    #[cfg(test)]
    pub(crate) fn gpu_ok_for_test(&self) -> bool {
        self.gpu_ok()
    }

    fn gpu_ok(&self) -> bool {
        !self.broken.load(Ordering::Relaxed) && !self.ctx.failed.load(Ordering::Relaxed)
    }

    /// The failure that moved this session (or the shared device) to the CPU, if any.
    pub(crate) fn last_error(&self) -> Option<String> {
        self.error
            .lock()
            .unwrap_or_else(|p| p.into_inner())
            .clone()
            .or_else(|| self.ctx.error.lock().unwrap_or_else(|p| p.into_inner()).clone())
    }

    fn give_up(&self, why: &str) {
        *self.error.lock().unwrap_or_else(|p| p.into_inner()) = Some(why.to_string());
        if !self.broken.swap(true, Ordering::Relaxed) {
            tracing::warn!(gpu = %self.ctx.info.name, "{why}; this session continues on the CPU (results are unaffected)");
        }
    }

    /// Shifted f32 spheres, the largest |center coordinate| and the largest
    /// radius; `None` if anything is non-finite.
    fn prepare(&self, centers: &[DVec3], radii: &[f64]) -> Option<(Vec<[f32; 4]>, f64, f64)> {
        let mut c_max: f64 = 0.0;
        let mut r_max: f64 = 0.0;
        let mut data = Vec::with_capacity(centers.len());
        for (&c, &r) in centers.iter().zip(radii) {
            if !(c.is_finite() && r.is_finite()) {
                return None;
            }
            let q = c - self.origin;
            c_max = c_max.max(q.abs().max_element());
            r_max = r_max.max(r.abs());
            data.push([q.x as f32, q.y as f32, q.z as f32, r as f32]);
        }
        Some((data, c_max, r_max))
    }

    fn tol(&self, set_abs: f64, c_max: f64, r_max: f64) -> f32 {
        (TOL_FACTOR * (set_abs.max(c_max) + r_max)) as f32
    }

    // ------------------------------------------------------------------
    // Public queries
    // ------------------------------------------------------------------

    pub(crate) fn queries(
        &self,
        samples: &Samples,
        centers: &[DVec3],
        radii: &[f64],
        want: Wants,
    ) -> QueryResults {
        crate::exec::block_on(self.queries_async(samples, centers, radii, want))
    }

    pub(crate) async fn queries_async(
        &self,
        samples: &Samples,
        centers: &[DVec3],
        radii: &[f64],
        want: Wants,
    ) -> QueryResults {
        let n = centers.len();
        let rows = self.inside.len.max(self.surface.len);
        let cpu = || QueryResults {
            inside_min: if want.inside_min {
                distances::row_minima(&samples.inside, centers, radii)
            } else {
                vec![]
            },
            surface_min: if want.surface_min {
                distances::row_minima(&samples.surface, centers, radii)
            } else {
                vec![]
            },
            pairs: if want.pairs {
                cpu_boundary_pairs(&samples.surface, centers, radii)
            } else {
                BoundaryPairs::default()
            },
        };
        if n == 0 || rows.div_ceil(WORKGROUP as usize) > MAX_WORKGROUPS || !self.gpu_ok() {
            self.stats.cpu_calls.fetch_add(1, Ordering::Relaxed);
            return cpu();
        }
        let Some((data, c_max, r_max)) = self.prepare(centers, radii) else {
            self.stats.cpu_calls.fetch_add(1, Ordering::Relaxed);
            return cpu();
        };
        match self.run_rows(&data, c_max, r_max, want).await {
            Ok(raw) => QueryResults {
                inside_min: if want.inside_min {
                    self.decide_rows(&samples.inside, centers, radii, &raw.inside, raw.tol_inside)
                } else {
                    vec![]
                },
                surface_min: if want.surface_min {
                    self.decide_rows(&samples.surface, centers, radii, &raw.surface, raw.tol_surface)
                } else {
                    vec![]
                },
                pairs: if want.pairs { to_csr(n, &raw.pairs) } else { BoundaryPairs::default() },
            },
            Err(e) => {
                self.give_up(&e);
                self.stats.cpu_calls.fetch_add(1, Ordering::Relaxed);
                cpu()
            }
        }
    }

    pub(crate) fn row_minima(
        &self,
        set: SampleSet,
        samples: &[DVec3],
        centers: &[DVec3],
        radii: &[f64],
    ) -> Vec<RowMin> {
        let want = match set {
            SampleSet::Inside => Wants { inside_min: true, ..Default::default() },
            SampleSet::Surface => Wants { surface_min: true, ..Default::default() },
        };
        // `queries` only touches the requested set, so the other one may be empty.
        let s = match set {
            SampleSet::Inside => Samples { inside: samples.to_vec(), ..Default::default() },
            SampleSet::Surface => Samples { surface: samples.to_vec(), ..Default::default() },
        };
        let r = self.queries(&s, centers, radii, want);
        match set {
            SampleSet::Inside => r.inside_min,
            SampleSet::Surface => r.surface_min,
        }
    }

    pub(crate) fn boundary_pairs(
        &self,
        surface: &[DVec3],
        centers: &[DVec3],
        radii: &[f64],
    ) -> BoundaryPairs {
        let s = Samples { surface: surface.to_vec(), ..Default::default() };
        self.queries(&s, centers, radii, Wants { pairs: true, ..Default::default() }).pairs
    }

    #[cfg(test)]
    pub(crate) fn nearest_sample_per_center(&self, surface: &[DVec3], centers: &[DVec3]) -> Vec<u32> {
        crate::exec::block_on(self.nearest_sample_per_center_async(surface, centers))
    }

    pub(crate) async fn nearest_sample_per_center_async(
        &self,
        surface: &[DVec3],
        centers: &[DVec3],
    ) -> Vec<u32> {
        let n = centers.len();
        if n == 0 || n > MAX_WORKGROUPS || self.surface.len == 0 || !self.gpu_ok() {
            self.stats.cpu_calls.fetch_add(1, Ordering::Relaxed);
            return distances::nearest_sample_per_center(centers, surface);
        }
        let radii = vec![0.0; n];
        let Some((data, c_max, _)) = self.prepare(centers, &radii) else {
            self.stats.cpu_calls.fetch_add(1, Ordering::Relaxed);
            return distances::nearest_sample_per_center(centers, surface);
        };
        let tol = self.tol(self.surface.abs_max, c_max, 0.0);
        match self.run_nearest(&data, tol).await {
            Ok(rows) => {
                let rescans = AtomicUsize::new(0);
                let out: Vec<u32> = centers
                    .par_iter()
                    .zip(&rows)
                    .map(|(&c, ro)| {
                        if ro.margin > tol {
                            ro.idx
                        } else {
                            rescans.fetch_add(1, Ordering::Relaxed);
                            distances::nearest_one(c, surface)
                        }
                    })
                    .collect();
                let r = rescans.into_inner();
                self.stats.rescanned.fetch_add(r, Ordering::Relaxed);
                self.stats.fast.fetch_add(n - r, Ordering::Relaxed);
                out
            }
            Err(e) => {
                self.give_up(&e);
                self.stats.cpu_calls.fetch_add(1, Ordering::Relaxed);
                distances::nearest_sample_per_center(centers, surface)
            }
        }
    }

    // ------------------------------------------------------------------
    // Decision step
    // ------------------------------------------------------------------

    fn decide_rows(
        &self,
        samples: &[DVec3],
        centers: &[DVec3],
        radii: &[f64],
        rows: &[RowOut],
        tol: f32,
    ) -> Vec<RowMin> {
        debug_assert_eq!(samples.len(), rows.len());
        let rescans = AtomicUsize::new(0);
        let out: Vec<RowMin> = samples
            .par_iter()
            .zip(rows)
            .map(|(&p, ro)| {
                if ro.margin > tol {
                    let j = ro.idx as usize;
                    let d = dist(p, centers[j]);
                    RowMin { gap: d - radii[j], d, j: j as u32 }
                } else {
                    rescans.fetch_add(1, Ordering::Relaxed);
                    distances::row_min(p, centers, radii)
                }
            })
            .collect();
        let r = rescans.into_inner();
        self.stats.rescanned.fetch_add(r, Ordering::Relaxed);
        self.stats.fast.fetch_add(samples.len() - r, Ordering::Relaxed);
        out
    }

    // ------------------------------------------------------------------
    // GPU passes
    // ------------------------------------------------------------------

    fn scratch(&self) -> std::sync::MutexGuard<'_, Scratch> {
        self.scratch.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn ensure_spheres(&self, s: &mut Scratch, data: &[[f32; 4]]) {
        let d = &self.ctx.device;
        if data.len() > s.spheres_cap {
            s.spheres_cap = data.len().next_power_of_two();
            s.spheres = storage(d, "spheres", (s.spheres_cap * 16) as u64, wgpu::BufferUsages::COPY_DST);
        }
        self.ctx.queue.write_buffer(&s.spheres, 0, bytemuck::cast_slice(data));
    }

    fn ensure_staging(&self, s: &mut Scratch, bytes: u64) {
        if bytes > s.staging_cap {
            s.staging_cap = bytes.next_power_of_two();
            s.staging = self.ctx.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("staging"),
                size: s.staging_cap,
                usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
        }
    }

    fn bind(
        &self,
        s: &Scratch,
        params: &wgpu::Buffer,
        samples: &wgpu::Buffer,
        out: &wgpu::Buffer,
    ) -> wgpu::BindGroup {
        fn e<'a>(binding: u32, buf: &'a wgpu::Buffer) -> wgpu::BindGroupEntry<'a> {
            wgpu::BindGroupEntry { binding, resource: buf.as_entire_binding() }
        }
        self.ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("morphit search"),
            layout: &self.ctx.layout,
            entries: &[e(0, params), e(1, samples), e(2, &s.spheres), e(3, out), e(4, &s.pairs)],
        })
    }

    /// Submit, wait, and read `bytes` from `staging`.
    ///
    /// The caller must not hold the scratch lock: on the web the wait yields
    /// to the browser's event loop, which resolves `mapAsync`.
    async fn readback(
        &self,
        staging: wgpu::Buffer,
        encoder: wgpu::CommandEncoder,
        bytes: u64,
    ) -> Result<Vec<u8>, String> {
        let index = self.ctx.queue.submit([encoder.finish()]);
        #[cfg(not(target_arch = "wasm32"))]
        {
            let dev = &self.ctx.device;
            let (tx, rx) = std::sync::mpsc::channel();
            staging.map_async(wgpu::MapMode::Read, 0..bytes, move |r| {
                let _ = tx.send(r);
            });
            // Other sessions poll the same device concurrently and may consume the
            // completion of our submission, so keep polling until our own callback
            // has fired.
            let deadline = web_time::Instant::now() + std::time::Duration::from_secs(30);
            loop {
                dev.poll(wgpu::PollType::Wait { submission_index: Some(index.clone()), timeout: None })
                    .map_err(|e| format!("poll failed: {e}"))?;
                match rx.try_recv() {
                    Ok(Ok(())) => break,
                    Ok(Err(e)) => return Err(format!("buffer map failed: {e}")),
                    Err(std::sync::mpsc::TryRecvError::Disconnected) => {
                        return Err("buffer map callback was dropped".to_string());
                    }
                    Err(std::sync::mpsc::TryRecvError::Empty) => {
                        if web_time::Instant::now() > deadline {
                            return Err("buffer map did not complete within 30 s".to_string());
                        }
                        std::thread::yield_now();
                    }
                }
            }
        }
        #[cfg(target_arch = "wasm32")]
        {
            // The browser completes the mapping once control returns to its
            // event loop; there is nothing to poll.
            let _ = index;
            let (tx, rx) = futures_channel::oneshot::channel();
            staging.map_async(wgpu::MapMode::Read, 0..bytes, move |r| {
                let _ = tx.send(r);
            });
            match rx.await {
                Ok(Ok(())) => {}
                Ok(Err(e)) => return Err(format!("buffer map failed: {e}")),
                Err(_) => return Err("buffer map callback was dropped".to_string()),
            }
        }
        let data = {
            let view = staging.get_mapped_range(0..bytes).map_err(|e| format!("mapped range: {e}"))?;
            view.to_vec()
        };
        staging.unmap();
        if self.ctx.failed.load(Ordering::Relaxed) {
            return Err("GPU reported an error".to_string());
        }
        Ok(data)
    }

    async fn run_rows(
        &self,
        data: &[[f32; 4]],
        c_max: f64,
        r_max: f64,
        want: Wants,
    ) -> Result<RawRows, String> {
        self.ensure_spheres(&mut self.scratch(), data);
        let pass = RowPass {
            n: data.len() as u32,
            do_inside: want.inside_min && self.inside.len > 0,
            do_surface: (want.surface_min || want.pairs) && self.surface.len > 0,
            want_pairs: want.pairs && self.surface.len > 0,
            tol_inside: self.tol(self.inside.abs_max, c_max, r_max),
            tol_surface: self.tol(self.surface.abs_max, c_max, r_max),
        };
        loop {
            // The scratch lock is taken and released inside the sync helpers,
            // never held across the await.
            let (enc, staging, lay) = self.encode_rows(&pass);
            let bytes = self.readback(staging, enc, lay.total).await?;

            let rows = |off: u64, len: usize| -> Vec<RowOut> {
                bytemuck::cast_slice(&bytes[off as usize..off as usize + len * 8]).to_vec()
            };
            let inside = if pass.do_inside { rows(0, self.inside.len) } else { vec![] };
            let surface = if pass.do_surface { rows(lay.off_surface, self.surface.len) } else { vec![] };
            let mut pairs = Vec::new();
            if pass.want_pairs {
                let off = lay.off_count as usize;
                let count = u32::from_le_bytes(bytes[off..off + 4].try_into().unwrap()) as usize;
                if self.grow_pairs(count) {
                    continue;
                }
                pairs = bytemuck::cast_slice::<u8, u32>(&bytes[off + 8..off + 8 + count * 8]).to_vec();
            }
            return Ok(RawRows {
                inside,
                surface,
                pairs,
                tol_inside: pass.tol_inside,
                tol_surface: pass.tol_surface,
            });
        }
    }

    /// Record the row passes of one [`SessionSearch::run_rows`] attempt.
    fn encode_rows(&self, pass: &RowPass) -> (wgpu::CommandEncoder, wgpu::Buffer, RowLayout) {
        let mut s = self.scratch();
        let inside_bytes = if pass.do_inside { (self.inside.len * 8) as u64 } else { 0 };
        let surface_bytes = if pass.do_surface { (self.surface.len * 8) as u64 } else { 0 };
        let pairs_bytes = if pass.want_pairs { 8 + (s.pairs_cap * 8) as u64 } else { 0 };
        // Layout in staging: [inside rows][surface rows][count, pad, pairs].
        let off_surface = inside_bytes;
        let off_count = off_surface + surface_bytes;
        let total = off_count + pairs_bytes;
        self.ensure_staging(&mut s, total);

        let q = &self.ctx.queue;
        if pass.want_pairs {
            q.write_buffer(&s.pairs, 0, &[0u8; 8]);
        }
        let params = |n_samples: usize, pairs: bool, tol: f32| Params {
            n_samples: n_samples as u32,
            n_spheres: pass.n,
            want_pairs: pairs as u32,
            pair_capacity: s.pairs_cap as u32,
            tol,
            _pad: [0; 3],
        };
        let mut enc =
            self.ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("rows") });
        if pass.do_inside {
            q.write_buffer(
                &s.params[0],
                0,
                bytemuck::bytes_of(&params(self.inside.len, false, pass.tol_inside)),
            );
            let bg = self.bind(&s, &s.params[0], &self.inside.buf, &s.out_inside);
            dispatch(&mut enc, &self.ctx.row_min, &bg, self.inside.len.div_ceil(WORKGROUP as usize) as u32);
            enc.copy_buffer_to_buffer(&s.out_inside, 0, &s.staging, 0, inside_bytes);
        }
        if pass.do_surface {
            q.write_buffer(
                &s.params[1],
                0,
                bytemuck::bytes_of(&params(self.surface.len, pass.want_pairs, pass.tol_surface)),
            );
            let bg = self.bind(&s, &s.params[1], &self.surface.buf, &s.out_surface);
            dispatch(&mut enc, &self.ctx.row_min, &bg, self.surface.len.div_ceil(WORKGROUP as usize) as u32);
            enc.copy_buffer_to_buffer(&s.out_surface, 0, &s.staging, off_surface, surface_bytes);
        }
        if pass.want_pairs {
            enc.copy_buffer_to_buffer(&s.pairs, 0, &s.staging, off_count, pairs_bytes);
        }
        (enc, s.staging.clone(), RowLayout { off_surface, off_count, total })
    }

    /// Enlarge the pair buffer if `count` pairs did not fit; true if the pass
    /// must run again.
    fn grow_pairs(&self, count: usize) -> bool {
        let mut s = self.scratch();
        if count <= s.pairs_cap {
            return false;
        }
        self.stats.pair_reruns.fetch_add(1, Ordering::Relaxed);
        s.pairs_cap = (count + count / 2).max(s.pairs_cap * 2);
        s.pairs = pair_buffer(&self.ctx.device, s.pairs_cap);
        true
    }

    async fn run_nearest(&self, data: &[[f32; 4]], tol: f32) -> Result<Vec<RowOut>, String> {
        let (enc, staging, bytes) = self.encode_nearest(data, tol);
        let raw = self.readback(staging, enc, bytes).await?;
        Ok(bytemuck::cast_slice(&raw).to_vec())
    }

    fn encode_nearest(&self, data: &[[f32; 4]], tol: f32) -> (wgpu::CommandEncoder, wgpu::Buffer, u64) {
        let mut s = self.scratch();
        self.ensure_spheres(&mut s, data);
        let n = data.len();
        if n > s.out_spheres_cap {
            s.out_spheres_cap = n.next_power_of_two();
            s.out_spheres = storage(
                &self.ctx.device,
                "out spheres",
                (s.out_spheres_cap * 8) as u64,
                wgpu::BufferUsages::COPY_SRC,
            );
        }
        let bytes = (n * 8) as u64;
        self.ensure_staging(&mut s, bytes);
        let p = Params {
            n_samples: self.surface.len as u32,
            n_spheres: n as u32,
            want_pairs: 0,
            pair_capacity: 0,
            tol,
            _pad: [0; 3],
        };
        self.ctx.queue.write_buffer(&s.params[2], 0, bytemuck::bytes_of(&p));
        let bg = self.bind(&s, &s.params[2], &self.surface.buf, &s.out_spheres);
        let mut enc = self
            .ctx
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("nearest") });
        dispatch(&mut enc, &self.ctx.nearest_sample, &bg, n as u32);
        enc.copy_buffer_to_buffer(&s.out_spheres, 0, &s.staging, 0, bytes);
        (enc, s.staging.clone(), bytes)
    }
}

/// What one row query computes.
struct RowPass {
    n: u32,
    do_inside: bool,
    do_surface: bool,
    want_pairs: bool,
    tol_inside: f32,
    tol_surface: f32,
}

/// Byte offsets of one row query's results in the staging buffer.
struct RowLayout {
    off_surface: u64,
    off_count: u64,
    total: u64,
}

struct RawRows {
    inside: Vec<RowOut>,
    surface: Vec<RowOut>,
    /// Flat `(sample, sphere)` pairs in arbitrary order.
    pairs: Vec<u32>,
    tol_inside: f32,
    tol_surface: f32,
}

fn dispatch(
    enc: &mut wgpu::CommandEncoder,
    pipeline: &wgpu::ComputePipeline,
    bg: &wgpu::BindGroup,
    groups: u32,
) {
    let mut pass =
        enc.begin_compute_pass(&wgpu::ComputePassDescriptor { label: None, timestamp_writes: None });
    pass.set_pipeline(pipeline);
    pass.set_bind_group(0, bg, &[]);
    pass.dispatch_workgroups(groups, 1, 1);
}

/// Counting-sort flat `(i, j)` pairs by sphere, each bucket ascending in `i`.
fn to_csr(n: usize, flat: &[u32]) -> BoundaryPairs {
    let mut counts = vec![0usize; n + 1];
    for p in flat.chunks_exact(2) {
        counts[p[1] as usize + 1] += 1;
    }
    for j in 0..n {
        counts[j + 1] += counts[j];
    }
    let offsets = counts.clone();
    let mut samples = vec![0u32; flat.len() / 2];
    let mut fill = counts;
    for p in flat.chunks_exact(2) {
        let j = p[1] as usize;
        samples[fill[j]] = p[0];
        fill[j] += 1;
    }
    let mut buckets: Vec<&mut [u32]> = Vec::with_capacity(n);
    let mut rest = samples.as_mut_slice();
    for j in 0..n {
        let len = offsets[j + 1] - offsets[j];
        let (head, tail) = rest.split_at_mut(len);
        buckets.push(head);
        rest = tail;
    }
    buckets.par_iter_mut().for_each(|b| b.sort_unstable());
    BoundaryPairs { offsets, samples }
}

#[cfg(test)]
mod csr_tests {
    use super::*;

    #[test]
    fn csr_sorts_by_sphere_then_sample() {
        let flat = [5, 1, 2, 0, 9, 1, 0, 1, 7, 0];
        let p = to_csr(3, &flat);
        assert_eq!(p.offsets, vec![0, 2, 5, 5]);
        assert_eq!(p.of(0), &[2, 7]);
        assert_eq!(p.of(1), &[0, 5, 9]);
        assert_eq!(p.of(2), &[] as &[u32]);
        assert_eq!(to_csr(2, &[]), BoundaryPairs { offsets: vec![0, 0, 0], samples: vec![] });
    }
}
