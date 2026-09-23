//! MorphIt: approximate a triangle mesh with a fixed budget of spheres by
//! gradient-based optimization. Rust port of <https://github.com/HIRO-group/MorphIt-1>.
//!
//! ```no_run
//! use std::sync::Arc;
//! use morphit::{Config, Mesh, Preset, Session};
//!
//! let mesh = Arc::new(Mesh::load("link0.obj")?);
//! let mut config = Config::from_preset(Preset::B);
//! config.model.num_spheres = 20;
//! config.random_seed = Some(42);
//! let result = morphit::pack(config, mesh)?;
//! result.save("spheres.json")?;
//! # Ok::<(), morphit::Error>(())
//! ```

pub mod config;
pub mod contains;
pub mod density;
pub mod device;
pub mod distances;
pub mod error;
mod exec;
pub mod loss;
pub mod math;
pub mod mesh;
mod mesh_io;
pub mod mesh_prep;
pub mod metrics;
pub mod optim;
#[doc(hidden)]
pub mod par;
pub mod quality;
pub mod result;
pub mod sampling;
pub mod search;
pub mod shapes;
pub mod state;
pub mod trainer;

/// Re-export of the vector math crate used in the public API.
pub use glam;

pub use config::{Config, LossId, LossWeights, Preset};
pub use device::{Device, GpuInfo, ResolvedDevice};
pub use error::{Error, Result};
pub use mesh::Mesh;
pub use mesh_prep::{MeshPrepReport, prepare_mesh};
pub use quality::{QualityMetrics, QualityOptions, evaluate_packing};
pub use result::PackResult;
pub use search::{init_gpu, list_devices};
pub use state::Spheres;
pub use trainer::{InitInfo, RunOutcome, Session, SessionState, StepInfo, pack};
