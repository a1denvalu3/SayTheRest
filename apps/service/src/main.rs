use anyhow::Result;
use clap::Parser;
use say_the_rest_service::{ServiceState, serve};
use std::net::{IpAddr, Ipv4Addr, SocketAddr};
use tokio::net::TcpListener;

#[derive(Parser)]
#[command(version, about = "Say the Rest per-user background service")]
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
    println!("Say the Rest service listening on http://{address}/v1");
    serve(listener, state).await
}

fn default_data_dir() -> std::path::PathBuf {
    if cfg!(windows) {
        std::env::var_os("LOCALAPPDATA")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(std::env::temp_dir)
            .join("SayTheRest")
    } else {
        std::env::var_os("XDG_STATE_HOME")
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| {
                std::env::var_os("HOME")
                    .map(std::path::PathBuf::from)
                    .unwrap_or_else(std::env::temp_dir)
                    .join(".local/state")
            })
            .join("say-the-rest")
    }
}

fn default_config_path() -> Option<std::path::PathBuf> {
    let local = std::path::PathBuf::from("say-the-rest.json");
    if local.is_file() {
        return Some(local);
    }
    std::env::current_exe()
        .ok()?
        .parent()?
        .join("say-the-rest.json")
        .is_file()
        .then(|| {
            std::env::current_exe()
                .unwrap()
                .parent()
                .unwrap()
                .join("say-the-rest.json")
        })
}
