//! wgpu backend: adapter discovery, one shared context per adapter, and the
//! `Device` resolution rules.
//!
//! Adapters are enumerated once per process and ordered discrete, integrated,
//! virtual, other, software; `gpu:N` indexes that list. Software rasterizers
//! (DX12 WARP, lavapipe) are skipped by `gpu`/`auto` unless
//! `MORPHIT_GPU_ALLOW_SOFTWARE=1` is set, and are never chosen by `auto`.
//!
//! In the browser (wasm32) the backend is WebGPU, whose adapter and device
//! requests are asynchronous: nothing is found until [`init_gpu`] has been
//! awaited, and until then `auto` runs on the CPU.

mod session;
#[cfg(test)]
mod tests;

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};

pub(crate) use session::SessionSearch;

use crate::device::{Device, GpuInfo, ResolvedDevice};
use crate::error::{Error, Result};
use crate::loss::Samples;
use crate::search::Search;

/// `auto` uses a GPU only when `num_spheres * (inside + surface samples)` is at
/// least this large; below it, dispatch latency outweighs the work.
pub const AUTO_GPU_MIN_WORK: usize = 4_000_000;

const SHADER: &str = include_str!("search.wgsl");

struct AdapterEntry {
    adapter: wgpu::Adapter,
    info: GpuInfo,
}

/// One wgpu device with the compiled pipelines, shared by every session on it.
pub(crate) struct GpuContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub info: GpuInfo,
    pub layout: wgpu::BindGroupLayout,
    pub row_min: wgpu::ComputePipeline,
    pub nearest_sample: wgpu::ComputePipeline,
    /// Set by the uncaptured-error handler; sessions then fall back to the CPU.
    pub failed: AtomicBool,
    /// Message of the first uncaptured error.
    pub error: Mutex<Option<String>>,
}

#[cfg(not(target_arch = "wasm32"))]
const BACKENDS: wgpu::Backends = wgpu::Backends::PRIMARY;
#[cfg(target_arch = "wasm32")]
const BACKENDS: wgpu::Backends = wgpu::Backends::BROWSER_WEBGPU;

fn instance() -> &'static wgpu::Instance {
    static INSTANCE: OnceLock<wgpu::Instance> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: BACKENDS,
            ..wgpu::InstanceDescriptor::new_without_display_handle_from_env()
        })
    })
}

fn kind_rank(t: wgpu::DeviceType) -> (u8, &'static str) {
    match t {
        wgpu::DeviceType::DiscreteGpu => (0, "discrete"),
        wgpu::DeviceType::IntegratedGpu => (1, "integrated"),
        wgpu::DeviceType::VirtualGpu => (2, "virtual"),
        wgpu::DeviceType::Other => (3, "other"),
        wgpu::DeviceType::Cpu => (4, "software"),
    }
}

static ADAPTERS: OnceLock<Vec<AdapterEntry>> = OnceLock::new();
static CONTEXTS: OnceLock<Mutex<HashMap<usize, Arc<GpuContext>>>> = OnceLock::new();

/// The adapter list, enumerated on first use. In the browser it stays empty
/// until [`init_gpu`] has run.
fn adapters() -> &'static [AdapterEntry] {
    #[cfg(not(target_arch = "wasm32"))]
    {
        ADAPTERS.get_or_init(|| rank_adapters(pollster::block_on(instance().enumerate_adapters(BACKENDS))))
    }
    #[cfg(target_arch = "wasm32")]
    {
        ADAPTERS.get().map(Vec::as_slice).unwrap_or(&[])
    }
}

fn contexts() -> std::sync::MutexGuard<'static, HashMap<usize, Arc<GpuContext>>> {
    CONTEXTS.get_or_init(Default::default).lock().unwrap_or_else(|e| e.into_inner())
}

/// De-duplicate and order adapters best first.
fn rank_adapters(raw: Vec<wgpu::Adapter>) -> Vec<AdapterEntry> {
    {
        let mut entries: Vec<(u8, usize, wgpu::Adapter, wgpu::AdapterInfo)> = Vec::new();
        for (order, adapter) in raw.into_iter().enumerate() {
            let info = adapter.get_info();
            // The same physical device shows up once per backend; keep the first.
            if entries.iter().any(|(_, _, _, i)| i.name == info.name && i.device_type == info.device_type) {
                continue;
            }
            entries.push((kind_rank(info.device_type).0, order, adapter, info));
        }
        entries.sort_by_key(|(rank, order, _, _)| (*rank, *order));
        entries
            .into_iter()
            .enumerate()
            .map(|(index, (_, _, adapter, info))| {
                let (_, kind) = kind_rank(info.device_type);
                let gi = GpuInfo {
                    index,
                    // Browsers hide the adapter name.
                    name: if info.name.is_empty() { "WebGPU adapter".to_string() } else { info.name.clone() },
                    backend: format!("{:?}", info.backend),
                    kind: kind.to_string(),
                    software: info.device_type == wgpu::DeviceType::Cpu,
                };
                AdapterEntry { adapter, info: gi }
            })
            .collect()
    }
}

/// Discover the GPUs and open a device on each. Natively this is the same as
/// [`list_devices`] (devices open lazily). In the browser it must be awaited
/// once before a session can use WebGPU; without `navigator.gpu` it returns
/// an empty list and every session runs on the CPU. Idempotent.
pub async fn init_gpu() -> Vec<GpuInfo> {
    #[cfg(target_arch = "wasm32")]
    if ADAPTERS.get().is_none() {
        let entries = rank_adapters(instance().enumerate_adapters(BACKENDS).await);
        let mut opened = Vec::new();
        for entry in &entries {
            match create_context(entry).await {
                Ok(ctx) => {
                    let ctx = Arc::new(ctx);
                    ctx.arm_error_handler();
                    opened.push((entry.info.index, ctx));
                }
                Err(e) => tracing::warn!("{e}"),
            }
        }
        if ADAPTERS.set(entries).is_ok() {
            contexts().extend(opened);
        }
    }
    list_devices()
}

/// All usable GPU adapters, best first (`gpu:N` refers to this order).
pub fn list_devices() -> Vec<GpuInfo> {
    adapters().iter().map(|e| e.info.clone()).collect()
}

fn software_allowed() -> bool {
    std::env::var("MORPHIT_GPU_ALLOW_SOFTWARE").map(|v| v == "1").unwrap_or(false)
}

/// Adapter index for `device`, or `None` for the CPU.
pub(crate) fn select(device: Device, work: usize, allow_software: bool) -> Result<Option<usize>> {
    let list = adapters();
    let allow_software = allow_software || software_allowed();
    let best_real = list.iter().find(|e| !e.info.software).map(|e| e.info.index);
    let best_any = list.first().map(|e| e.info.index);
    match device {
        Device::Cpu => Ok(None),
        Device::Auto => {
            if work < AUTO_GPU_MIN_WORK {
                return Ok(None);
            }
            Ok(best_real.or(if allow_software { best_any } else { None }))
        }
        Device::Gpu(Some(i)) => {
            if i < list.len() {
                Ok(Some(i))
            } else {
                Err(Error::config("model.device", format!("gpu:{i} does not exist; {}", describe(list))))
            }
        }
        Device::Gpu(None) => best_real
            .or(if allow_software { best_any } else { None })
            .map(Some)
            .ok_or_else(|| Error::config("model.device", format!("no usable GPU; {}", describe(list)))),
    }
}

fn describe(list: &[AdapterEntry]) -> String {
    if list.is_empty() {
        return "no adapters found".to_string();
    }
    let names: Vec<String> =
        list.iter().map(|e| format!("gpu:{} {} ({})", e.info.index, e.info.name, e.info.kind)).collect();
    format!("available: {}", names.join(", "))
}

/// The shared context for adapter `index`, created on first use.
pub(crate) fn context(index: usize) -> Result<Arc<GpuContext>> {
    #[cfg_attr(target_arch = "wasm32", allow(unused_mut))]
    let mut map = contexts();
    if let Some(ctx) = map.get(&index) {
        return Ok(ctx.clone());
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let entry = &adapters()[index];
        let ctx = Arc::new(pollster::block_on(create_context(entry))?);
        ctx.arm_error_handler();
        map.insert(index, ctx.clone());
        Ok(ctx)
    }
    #[cfg(target_arch = "wasm32")]
    Err(Error::config("model.device", format!("gpu:{index} could not be opened (see init_gpu)")))
}

async fn create_context(entry: &AdapterEntry) -> Result<GpuContext> {
    let desc = wgpu::DeviceDescriptor {
        label: Some("morphit"),
        required_features: wgpu::Features::empty(),
        required_limits: wgpu::Limits::downlevel_defaults().using_resolution(entry.adapter.limits()),
        ..Default::default()
    };
    let (device, queue) = entry
        .adapter
        .request_device(&desc)
        .await
        .map_err(|e| Error::config("model.device", format!("cannot open {}: {e}", entry.info.name)))?;
    let failed = AtomicBool::new(false);
    let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("morphit search"),
        source: wgpu::ShaderSource::Wgsl(SHADER.into()),
    });
    let storage = |binding, read_only| wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    };
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("morphit search"),
        entries: &[
            wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
            storage(1, true),
            storage(2, true),
            storage(3, false),
            storage(4, false),
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("morphit search"),
        bind_group_layouts: &[Some(&layout)],
        immediate_size: 0,
    });
    let pipeline = |entry_point: &str| {
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(entry_point),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some(entry_point),
            compilation_options: Default::default(),
            cache: None,
        })
    };
    let row_min = pipeline("row_min");
    let nearest_sample = pipeline("nearest_sample");
    let ctx = GpuContext {
        device,
        queue,
        info: entry.info.clone(),
        layout,
        row_min,
        nearest_sample,
        failed,
        error: Mutex::new(None),
    };
    Ok(ctx)
}

/// Build the search for `device`; falls back to the CPU when `auto` decides so.
pub(crate) fn open(samples: &Samples, device: Device, num_spheres: usize) -> Result<Search> {
    open_with(samples, device, num_spheres, false)
}

pub(crate) fn open_with(
    samples: &Samples,
    device: Device,
    num_spheres: usize,
    allow_software: bool,
) -> Result<Search> {
    let work = num_spheres.saturating_mul(samples.inside.len() + samples.surface.len());
    let Some(index) = select(device, work, allow_software)? else {
        tracing::info!(%device, work, "searches run on the CPU");
        return Ok(Search::cpu());
    };
    let ctx = context(index)?;
    let name = ctx.info.name.clone();
    let session = SessionSearch::new(ctx, samples);
    tracing::info!(%device, work, gpu = %name, "searches run on the GPU");
    Ok(Search::with_gpu(session, ResolvedDevice::Gpu { index, name }))
}

impl GpuContext {
    /// Install the error handler that turns wgpu's default panic into a CPU fallback.
    pub(crate) fn arm_error_handler(self: &Arc<Self>) {
        let flag = Arc::downgrade(self);
        self.device.on_uncaptured_error(Arc::new(move |e: wgpu::Error| {
            tracing::warn!("GPU error, falling back to the CPU: {e}");
            if let Some(ctx) = flag.upgrade() {
                ctx.error.lock().unwrap_or_else(|p| p.into_inner()).get_or_insert_with(|| e.to_string());
                ctx.failed.store(true, Ordering::Relaxed);
            }
        }));
    }
}
