//! Compute device selection (`model.device` in the config).
//!
//! The value is parsed here whether or not the crate is built with the `gpu`
//! feature; only the resolution of `Auto`/`Gpu` to an adapter needs wgpu.
//!
//! Whichever device runs the searches, results are bit-identical: the GPU only
//! proposes candidates and the CPU makes every decision in f64 (see `search`).

use std::{fmt, str::FromStr};

use crate::error::{Error, Result};

/// Requested compute device.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Device {
    /// GPU when one is present and the problem is large enough, else CPU.
    Auto,
    /// CPU only.
    Cpu,
    /// A GPU; `None` picks the best adapter, `Some(n)` the n-th enumerated adapter.
    Gpu(Option<usize>),
}

impl Device {
    /// Parse a `model.device` string.
    ///
    /// Accepts `auto`, `cpu`, `gpu`, `gpu:N` and, for Python config files, the
    /// aliases `cuda`, `cuda:N` and `mps` (all treated as `gpu`). Case-insensitive.
    pub fn parse(s: &str) -> Result<Device> {
        let lower = s.trim().to_ascii_lowercase();
        let (name, index) = match lower.split_once(':') {
            Some((n, i)) => (n, Some(i)),
            None => (lower.as_str(), None),
        };
        let bad =
            || Error::config("model.device", format!("unknown device `{s}`; use auto, cpu, gpu or gpu:N"));
        match (name, index) {
            ("auto", None) => Ok(Device::Auto),
            ("cpu", None) => Ok(Device::Cpu),
            ("gpu" | "cuda" | "mps", None) => Ok(Device::Gpu(None)),
            ("gpu" | "cuda", Some(i)) => i.parse::<usize>().map(|i| Device::Gpu(Some(i))).map_err(|_| bad()),
            _ => Err(bad()),
        }
    }
}

impl FromStr for Device {
    type Err = Error;
    fn from_str(s: &str) -> Result<Device> {
        Device::parse(s)
    }
}

impl fmt::Display for Device {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Device::Auto => f.write_str("auto"),
            Device::Cpu => f.write_str("cpu"),
            Device::Gpu(None) => f.write_str("gpu"),
            Device::Gpu(Some(i)) => write!(f, "gpu:{i}"),
        }
    }
}

/// The device a session actually runs its searches on.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ResolvedDevice {
    Cpu,
    /// Adapter index (as listed by [`crate::list_devices`]) and adapter name.
    Gpu {
        index: usize,
        name: String,
    },
}

impl fmt::Display for ResolvedDevice {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            ResolvedDevice::Cpu => f.write_str("cpu"),
            ResolvedDevice::Gpu { index, name } => write!(f, "gpu:{index} {name}"),
        }
    }
}

/// Description of one GPU adapter.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GpuInfo {
    /// Index to use in `gpu:N`.
    pub index: usize,
    pub name: String,
    /// Graphics API in use (Vulkan, Dx12, Metal, ...).
    pub backend: String,
    /// `discrete`, `integrated`, `virtual`, `software` or `other`.
    pub kind: String,
    /// True for software rasterizers such as DX12 WARP or lavapipe.
    pub software: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_forms() {
        assert_eq!(Device::parse("auto").unwrap(), Device::Auto);
        assert_eq!(Device::parse("CPU").unwrap(), Device::Cpu);
        assert_eq!(Device::parse("gpu").unwrap(), Device::Gpu(None));
        assert_eq!(Device::parse("gpu:2").unwrap(), Device::Gpu(Some(2)));
        assert_eq!(Device::parse("cuda").unwrap(), Device::Gpu(None));
        assert_eq!(Device::parse("cuda:0").unwrap(), Device::Gpu(Some(0)));
        assert_eq!(Device::parse("mps").unwrap(), Device::Gpu(None));
        assert_eq!(" gpu:1 ".parse::<Device>().unwrap(), Device::Gpu(Some(1)));
    }

    #[test]
    fn rejects_garbage() {
        for s in ["", "tpu", "gpu:", "gpu:x", "gpu:-1", "cpu:0", "auto:1", "mps:0"] {
            let e = Device::parse(s).unwrap_err();
            assert!(matches!(e, Error::Config { ref key, .. } if key == "model.device"), "{s}: {e}");
        }
    }

    #[test]
    fn display_round_trips() {
        for d in [Device::Auto, Device::Cpu, Device::Gpu(None), Device::Gpu(Some(3))] {
            assert_eq!(Device::parse(&d.to_string()).unwrap(), d);
        }
        assert_eq!(ResolvedDevice::Cpu.to_string(), "cpu");
        assert_eq!(ResolvedDevice::Gpu { index: 1, name: "X".into() }.to_string(), "gpu:1 X");
    }
}
