//! `morphit-server`: the MorphIt HTTP API and web UI.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream};
use std::path::PathBuf;
use std::process::ExitCode;
use std::time::Duration;

use clap::Parser;
use morphit_server::{AppState, ServerConfig, router};

#[derive(Parser, Debug)]
#[command(
    name = "morphit-server",
    version,
    about = "MorphIt HTTP API (compatible with the Python web service)"
)]
struct Args {
    /// Address to listen on.
    #[arg(long, env = "MORPHIT_BIND", default_value = "0.0.0.0:8000")]
    bind: SocketAddr,
    /// Directory with index.html and examples/ [default: `web` next to the
    /// executable, else `./web`].
    #[arg(long, env = "MORPHIT_WEB_DIR")]
    web_dir: Option<PathBuf>,
    /// Optimizer device: auto, cpu, gpu or gpu:N.
    #[arg(long, env = "MORPHIT_DEVICE", default_value = "auto")]
    device: String,
    /// Worker threads for the optimizer [default: all cores].
    #[arg(long, env = "MORPHIT_THREADS")]
    threads: Option<usize>,
    /// Directory for robot-mode session files [default: <temp>/morphit-robot-sessions].
    #[arg(long, env = "MORPHIT_SESSION_DIR")]
    session_dir: Option<PathBuf>,
    /// Seconds of inactivity after which a robot session is deleted.
    #[arg(long, env = "MORPHIT_SESSION_TTL", default_value_t = 3600)]
    session_ttl: u64,
    /// Log filter (tracing EnvFilter syntax), e.g. `info` or `morphit=debug`.
    #[arg(long, env = "MORPHIT_LOG", default_value = "info,wgpu_hal=off,wgpu_core=warn")]
    log: String,
    /// Request /healthz on the bind address and exit 0 when it answers 200
    /// (for container health checks).
    #[arg(long)]
    healthcheck: bool,
}

fn default_web_dir() -> PathBuf {
    let beside_exe = std::env::current_exe().ok().and_then(|e| e.parent().map(|d| d.join("web")));
    match beside_exe {
        Some(d) if d.join("index.html").is_file() => d,
        _ => PathBuf::from("web"),
    }
}

/// Minimal HTTP/1.0 GET of `/healthz`; true on status 200.
fn healthcheck(bind: SocketAddr) -> bool {
    let mut addr = bind;
    if addr.ip().is_unspecified() {
        addr.set_ip(if addr.is_ipv4() {
            [127, 0, 0, 1].into()
        } else {
            std::net::Ipv6Addr::LOCALHOST.into()
        });
    }
    let Ok(mut s) = TcpStream::connect_timeout(&addr, Duration::from_secs(3)) else { return false };
    let _ = s.set_read_timeout(Some(Duration::from_secs(3)));
    if s.write_all(b"GET /healthz HTTP/1.0\r\nHost: localhost\r\n\r\n").is_err() {
        return false;
    }
    let mut buf = [0u8; 64];
    let n = s.read(&mut buf).unwrap_or(0);
    let head = String::from_utf8_lossy(&buf[..n]);
    head.starts_with("HTTP/1.") && head.split_whitespace().nth(1) == Some("200")
}

fn main() -> ExitCode {
    let args = Args::parse();
    if args.healthcheck {
        return if healthcheck(args.bind) { ExitCode::SUCCESS } else { ExitCode::FAILURE };
    }
    let filter = tracing_subscriber::EnvFilter::try_new(&args.log).unwrap_or_else(|_| "info".into());
    tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_ansi(std::io::IsTerminal::is_terminal(&std::io::stderr()))
        .with_writer(std::io::stderr)
        .init();

    if let Err(e) = args.device.parse::<morphit::Device>() {
        eprintln!("error: {e}");
        return ExitCode::FAILURE;
    }
    if let Some(n) = args.threads
        && let Err(e) = rayon::ThreadPoolBuilder::new().num_threads(n).build_global()
    {
        eprintln!("error: cannot set up {n} threads: {e}");
        return ExitCode::FAILURE;
    }
    let web_dir = args.web_dir.unwrap_or_else(default_web_dir);
    if !web_dir.join("index.html").is_file() {
        tracing::warn!("{} has no index.html; `/` will fail", web_dir.display());
    }
    let mut config = ServerConfig::new(web_dir);
    config.device = args.device;
    config.session_ttl = Duration::from_secs(args.session_ttl);
    if let Some(d) = args.session_dir {
        config.session_root = d;
    }

    let gpus = if config.device == "cpu" { Vec::new() } else { morphit::list_devices() };
    if gpus.is_empty() {
        tracing::info!("no GPU in use; packing runs on the CPU");
    }
    for g in &gpus {
        tracing::info!("GPU {}: {} ({}, {})", g.index, g.name, g.backend, g.kind);
    }
    tracing::info!(
        "web dir {}, device {}, sessions in {}",
        config.web_dir.display(),
        config.device,
        config.session_root.display()
    );

    let runtime = match tokio::runtime::Builder::new_multi_thread().enable_all().build() {
        Ok(r) => r,
        Err(e) => {
            eprintln!("error: cannot start the runtime: {e}");
            return ExitCode::FAILURE;
        }
    };
    let result = runtime.block_on(async move {
        let listener = tokio::net::TcpListener::bind(args.bind).await?;
        tracing::info!("listening on http://{}", listener.local_addr()?);
        let app = router(AppState::new(config));
        axum::serve(listener, app).with_graceful_shutdown(shutdown()).await
    });
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}

/// Ctrl-C, or SIGTERM on Unix (what `docker stop` sends).
async fn shutdown() {
    let ctrl_c = async {
        let _ = tokio::signal::ctrl_c().await;
    };
    #[cfg(unix)]
    let term = async {
        if let Ok(mut s) = tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            s.recv().await;
        }
    };
    #[cfg(not(unix))]
    let term = std::future::pending::<()>();
    tokio::select! {
        _ = ctrl_c => {},
        _ = term => {},
    }
    tracing::info!("shutting down");
}
