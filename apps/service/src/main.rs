use anyhow::Result;
use clap::Parser;
use sayit_service::{ServiceState, serve};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use tokio::net::TcpListener;

#[derive(Parser)]
#[command(version, about = "sayIt per-user background service")]
struct Args {
    #[arg(long, default_value_t = 55391)]
    port: u16,
    #[arg(long)]
    data_dir: Option<std::path::PathBuf>,
    #[arg(long)]
    config: Option<std::path::PathBuf>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Args::parse();
    let address = SocketAddr::new(IpAddr::V4(Ipv4Addr::LOCALHOST), args.port);
    let listener = TcpListener::bind(address).await?;
    let data_dir = args.data_dir.unwrap_or_else(default_data_dir);
    let config = args.config.or_else(default_config_path);
    let state = ServiceState::open(data_dir, config)?;
    state.start_worker();
    println!("sayIt service listening on http://{address}/v1");
    serve(listener, state).await
}

fn default_data_dir() -> std::path::PathBuf {
    let base = if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
    } else {
        std::env::var_os("XDG_STATE_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir)
                    .join(".local/state")
            })
    };
    let current = base.join("sayit");
    let legacy = base.join(if cfg!(windows) {
        "SayTheRest"
    } else {
        "say-the-rest"
    });
    if !current.exists() && legacy.exists() && std::fs::rename(&legacy, &current).is_err() {
        return legacy;
    }
    current
}

fn default_config_path() -> Option<std::path::PathBuf> {
    for local in ["sayit.json", "say-the-rest.json"] {
        let local = std::path::PathBuf::from(local);
        if local.is_file() {
            return Some(local);
        }
    }
    let parent = std::env::current_exe().ok()?.parent()?.to_owned();
    ["sayit.json", "say-the-rest.json"]
        .into_iter()
        .map(|name| parent.join(name))
        .find(|path| path.is_file())
}
