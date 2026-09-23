//! `morphit` command-line tool: pack meshes with spheres and score the result.

use std::ops::ControlFlow;
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand};
use morphit::glam::DVec3;
use morphit::{Config, LossId, Mesh, PackResult, Preset, QualityOptions, RunOutcome, Session};
use serde_json::Value;

#[derive(Parser)]
#[command(name = "morphit", version, about = "Approximate a mesh with spheres (MorphIt)")]
struct Cli {
    /// Log level for library messages on stderr (error, warn, info, debug, trace).
    #[arg(long, global = true, default_value = "warn")]
    log: String,
    /// Worker threads for the parallel parts (default: all cores).
    #[arg(long, global = true)]
    threads: Option<usize>,
    #[command(subcommand)]
    cmd: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Pack a mesh (.obj, .stl, .ply or .dae) with spheres and write the result JSON.
    Pack(PackArgs),
    /// Turn a result JSON into a simulator model: URDF or MJCF (MuJoCo).
    Export(ExportArgs),
    /// Score a result JSON against its mesh (debug_quick_eval metrics).
    Metrics(MetricsArgs),
    /// Print mesh properties.
    Info { mesh: PathBuf },
    /// List the loss-weight presets.
    Presets,
    /// List the GPU adapters usable with `--device gpu:N`.
    Devices,
}

#[derive(clap::Args)]
struct PackArgs {
    /// Mesh file (.obj, .stl, .ply or .dae).
    mesh: PathBuf,
    /// Loss-weight preset: MorphIt-V, MorphIt-S, MorphIt-B, MorphIt-Obj, MorphIt-Obj-mass.
    #[arg(short, long, default_value = "MorphIt-B")]
    preset: String,
    /// Full or partial config JSON applied before the other options.
    #[arg(long)]
    config: Option<PathBuf>,
    /// Number of spheres.
    #[arg(short = 'n', long)]
    spheres: Option<usize>,
    /// Training iterations.
    #[arg(short, long)]
    iterations: Option<usize>,
    /// Random seed (omit for a random run).
    #[arg(short, long)]
    seed: Option<u64>,
    /// Learn per-sphere masses.
    #[arg(long)]
    per_sphere_mass: bool,
    /// Compute device: auto (GPU for large problems when present), cpu, gpu or gpu:N.
    /// Results do not depend on the device.
    #[arg(long, value_name = "DEVICE")]
    device: Option<String>,
    /// Override any config value by dotted key, e.g. --set training.center_lr=0.001.
    #[arg(long = "set", value_name = "KEY=VALUE")]
    sets: Vec<String>,
    /// Output JSON file (default: stdout).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Also write the training history JSON here.
    #[arg(long)]
    history: Option<PathBuf>,
    /// Do not print progress.
    #[arg(short, long)]
    quiet: bool,
}

#[derive(clap::Args)]
struct MetricsArgs {
    mesh: PathBuf,
    result: PathBuf,
    #[arg(long, default_value_t = 0)]
    seed: u64,
    #[arg(long, default_value_t = 50_000)]
    surface_samples: usize,
    #[arg(long, default_value_t = 50_000)]
    volume_samples: usize,
    /// Density used when the result has no masses.
    #[arg(long, default_value_t = 1000.0)]
    density: f64,
    /// Print JSON instead of a table row.
    #[arg(long)]
    json: bool,
    /// Score the mesh as loaded. By default overlapping closed bodies are
    /// unioned first, as for packing, so metrics compare like with like.
    #[arg(long)]
    raw_mesh: bool,
}

#[derive(Clone, Copy, PartialEq, Eq, clap::ValueEnum)]
enum ModelFormat {
    /// URDF: one link per sphere on fixed joints (ROS, PyBullet, Genesis, ...).
    Urdf,
    /// MJCF: MuJoCo XML, one sphere body per sphere.
    Mjcf,
}

#[derive(clap::Args)]
struct ExportArgs {
    /// Result JSON written by `morphit pack` (or the Python code).
    result: PathBuf,
    /// Output format [default: mjcf for a .xml output, else urdf].
    #[arg(short, long, value_enum)]
    format: Option<ModelFormat>,
    /// Output file (default: stdout).
    #[arg(short, long)]
    output: Option<PathBuf>,
    /// Model name [default: the mesh file name in the result, else "object"].
    #[arg(long)]
    name: Option<String>,
    /// Sphere color as #rrggbb.
    #[arg(long, default_value = "#3399ff")]
    color: String,
    /// Total mass in kg, split over the spheres by volume.
    #[arg(long, default_value_t = 1.0)]
    total_mass: f64,
    /// Weld the object to the world instead of letting it move freely.
    #[arg(long)]
    anchored: bool,
    /// Decimal places of the numbers written.
    #[arg(long, default_value_t = 6)]
    decimals: usize,
}

fn main() -> ExitCode {
    let cli = Cli::parse();
    let filter = tracing_subscriber::EnvFilter::try_new(&cli.log)
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn"));
    tracing_subscriber::fmt().with_env_filter(filter).with_writer(std::io::stderr).init();
    if let Some(n) = cli.threads
        && let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(n).build_global()
    {
        eprintln!("error: cannot configure {n} threads: {e}");
        return ExitCode::FAILURE;
    }
    let result = match cli.cmd {
        Command::Pack(a) => pack(a),
        Command::Export(a) => export(a),
        Command::Metrics(a) => metrics(a),
        Command::Info { mesh } => info(mesh),
        Command::Presets => {
            presets();
            Ok(())
        }
        Command::Devices => {
            devices();
            Ok(())
        }
    };
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// `KEY=VALUE` where VALUE is parsed as JSON when possible, else as a string.
fn parse_set(s: &str) -> Result<(String, Value), String> {
    let (k, v) = s.split_once('=').ok_or_else(|| format!("expected KEY=VALUE, got `{s}`"))?;
    let value = serde_json::from_str(v).unwrap_or_else(|_| Value::String(v.to_string()));
    Ok((k.trim().to_string(), value))
}

fn export(a: ExportArgs) -> Result<(), Box<dyn std::error::Error>> {
    use morphit_robot::object_model::{ObjectModelOptions, write_object_mjcf, write_object_urdf};
    let r = PackResult::load(&a.result)?;
    let format = a.format.unwrap_or_else(|| {
        let xml =
            a.output.as_ref().and_then(|o| o.extension()).is_some_and(|e| e.eq_ignore_ascii_case("xml"));
        if xml { ModelFormat::Mjcf } else { ModelFormat::Urdf }
    });
    let color = morphit_robot::color::hex_to_rgba(&a.color)
        .ok_or_else(|| format!("--color must be #rrggbb, got {}", a.color))?;
    if !(a.total_mass > 0.0 && a.total_mass.is_finite()) {
        return Err(format!("--total-mass must be positive, got {}", a.total_mass).into());
    }
    let name = a.name.unwrap_or_else(|| {
        std::path::Path::new(&r.mesh_path)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "object".into())
    });
    let opts = ObjectModelOptions {
        robot_name: name,
        color_rgba: color,
        decimals: a.decimals,
        total_mass: a.total_mass,
        anchored: a.anchored,
        ..Default::default()
    };
    let model = match format {
        ModelFormat::Urdf => write_object_urdf(&r.centers, &r.radii, &opts)?,
        ModelFormat::Mjcf => write_object_mjcf(&r.centers, &r.radii, &opts)?,
    };
    match &a.output {
        Some(path) => {
            if let Some(dir) = path.parent().filter(|d| !d.as_os_str().is_empty()) {
                std::fs::create_dir_all(dir)?;
            }
            std::fs::write(path, &model.text)?;
            let c = model.centroid;
            eprintln!(
                "wrote {} ({} spheres, {}, origin {:.6} {:.6} {:.6})",
                path.display(),
                r.radii.len(),
                if a.anchored { "anchored" } else { "free" },
                c[0],
                c[1],
                c[2]
            );
        }
        None => print!("{}", model.text),
    }
    Ok(())
}

fn build_config(a: &PackArgs) -> Result<Config, Box<dyn std::error::Error>> {
    let preset: Preset = a.preset.parse()?;
    let mut config = Config::from_preset(preset);
    if let Some(path) = &a.config {
        let text = std::fs::read_to_string(path)?;
        let from_file = Config::from_json_str(&text)?;
        let weights = from_file.training.loss_weights();
        config = from_file;
        // A config file's weights win over the preset only when it sets them.
        let v: Value = serde_json::from_str(&text)?;
        let mut w = preset.weights();
        for id in LossId::ALL {
            if v["training"].get(id.weight_key()).is_some() {
                w[id] = weights[id];
            }
        }
        config.training.set_loss_weights(&w);
    }
    if let Some(n) = a.spheres {
        config.model.num_spheres = n;
        config.model.max_spheres = n;
    }
    if let Some(i) = a.iterations {
        config.training.iterations = i;
    }
    if a.seed.is_some() {
        config.random_seed = a.seed;
    }
    if a.per_sphere_mass {
        config.model.per_sphere_mass = true;
    }
    if let Some(d) = &a.device {
        config.model.device = d.parse::<morphit::Device>()?.to_string();
    }
    for s in &a.sets {
        let (k, v) = parse_set(s)?;
        config.set(&k, v)?;
    }
    Ok(config)
}

fn pack(a: PackArgs) -> Result<(), Box<dyn std::error::Error>> {
    let config = build_config(&a)?;
    let mesh = Arc::new(Mesh::load(&a.mesh)?);
    let started = Instant::now();
    let mut session = Session::new(config, mesh)?;
    let total = session.total_iterations();
    let every = session.config().training.verbose_frequency.max(1);
    if !a.quiet {
        eprintln!(
            "packing {} with {} spheres ({}), {} iterations, seed {}, device {}",
            a.mesh.display(),
            session.config().model.num_spheres,
            a.preset,
            total,
            session.init_info().seed,
            session.device()
        );
        let prep = session.mesh_prep();
        if prep.is_unioned() {
            eprintln!(
                "mesh prep: unioned {} overlapping bodies: faces {} -> {}, volume {:.6e} -> {:.6e}",
                prep.n_bodies, prep.faces_before, prep.faces_after, prep.volume_before, prep.volume_after
            );
        }
    }
    let quiet = a.quiet;
    let outcome = session.run(|info| {
        if !quiet && (info.iteration % every == 0 || info.done || info.density_control.is_some()) {
            let dc = match info.density_control {
                Some(o) => format!("  density control: replaced {}", o.removed),
                None => String::new(),
            };
            eprintln!(
                "iter {:>5}/{total}  loss {:>12.6}  spheres {:>3}{dc}",
                info.iteration, info.total_loss, info.num_spheres
            );
        }
        ControlFlow::Continue(())
    })?;
    let pruned = session.finalize();
    let result = session.result();
    if !a.quiet {
        let how = match outcome {
            RunOutcome::Converged => "converged",
            _ => "completed",
        };
        eprintln!(
            "{how} in {:.2}s: {} spheres ({} pruned)",
            started.elapsed().as_secs_f64(),
            result.num_spheres,
            pruned
        );
    }
    match &a.output {
        Some(path) => result.save(path)?,
        None => println!("{}", result.to_json_string()),
    }
    if let Some(path) = &a.history {
        std::fs::write(path, serde_json::to_string_pretty(&session.history().to_json())?)?;
    }
    Ok(())
}

fn metrics(a: MetricsArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mesh = Arc::new(Mesh::load(&a.mesh)?);
    let mesh = if a.raw_mesh {
        mesh
    } else {
        let (prepared, prep) = mesh.prepared();
        if prep.is_unioned() {
            eprintln!(
                "scoring the union of {} overlapping bodies (--raw-mesh scores the mesh as loaded)",
                prep.n_bodies
            );
        }
        prepared
    };
    let r = PackResult::load(&a.result)?;
    let centers: Vec<DVec3> = r.centers.iter().map(|c| DVec3::from_array(*c)).collect();
    let opts = QualityOptions {
        seed: a.seed,
        surface_samples: a.surface_samples,
        volume_samples: a.volume_samples,
        bounds_expand: 1.5,
        density: a.density,
    };
    let masses = (r.masses.len() == r.radii.len()).then_some(r.masses.as_slice());
    let q = morphit::evaluate_packing(&mesh, &centers, &r.radii, masses, &opts);
    if a.json {
        println!("{}", serde_json::to_string_pretty(&q)?);
    } else {
        println!(
            "{:>6} {:>5} {:>6} {:>6} {:>6} {:>6} {:>9} {:>9} {:>9} {:>9} {:>9}",
            "actual",
            "n_out",
            "n_tiny",
            "r_in",
            "r_out",
            "r_uni",
            "d_avg_mm",
            "d_max_mm",
            "mass_rel",
            "com_rel",
            "I_rel"
        );
        println!(
            "{:>6} {:>5} {:>6} {:>6.3} {:>6.3} {:>6.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3} {:>9.3}",
            q.actual_n,
            q.n_out,
            q.n_tiny,
            q.r_in,
            q.r_out,
            q.r_uni,
            q.d_avg_mm,
            q.d_max_mm,
            q.mass_rel,
            q.com_rel,
            q.i_rel
        );
    }
    Ok(())
}

fn info(path: PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    let m = Mesh::load(&path)?;
    let (lo, hi) = m.bounds();
    let i = m.moment_inertia();
    println!("file          {}", path.display());
    println!("vertices      {}", m.vertices().len());
    println!("faces         {}", m.faces().len());
    println!("volume        {:.9e}", m.volume());
    println!("area          {:.9e}", m.area());
    println!("scale         {:.9e}", m.scale());
    println!("bounds min    {:?}", lo.to_array());
    println!("bounds max    {:?}", hi.to_array());
    println!("center mass   {:?}", m.center_mass().to_array());
    println!("inertia (rho=1, about COM)");
    for r in 0..3 {
        println!("              [{:.6e}, {:.6e}, {:.6e}]", i.col(0)[r], i.col(1)[r], i.col(2)[r]);
    }
    if m.winding_flipped() {
        println!("note          faces were wound inward and have been flipped");
    }
    let (union, prep) = Arc::new(m).prepared();
    println!("mesh prep     {} ({})", prep.action, prep.reason);
    println!(
        "bodies        {} ({} closed, {} open, {} degenerate dropped), overlapping: {}",
        prep.n_bodies, prep.n_closed, prep.n_open, prep.n_degenerate_dropped, prep.overlapping
    );
    if prep.is_unioned() {
        println!("union         {} faces, volume {:.9e}", union.faces().len(), union.volume());
    }
    Ok(())
}

fn presets() {
    for p in Preset::ALL {
        let w = p.weights();
        let active: Vec<String> = LossId::ALL
            .iter()
            .filter(|id| w[**id] != 0.0)
            .map(|id| format!("{}={}", id.weight_key().trim_end_matches("_weight"), w[*id]))
            .collect();
        println!("{:<17} {}", p.name(), active.join(" "));
    }
}

fn devices() {
    let list = morphit::list_devices();
    if list.is_empty() {
        println!("no GPU adapters (device `gpu` is unavailable; `auto` and `cpu` use the CPU)");
        return;
    }
    for d in list {
        let sw = if d.software { ", software rasterizer" } else { "" };
        println!("gpu:{}  {}  ({}, {}{sw})", d.index, d.name, d.kind, d.backend);
    }
}
