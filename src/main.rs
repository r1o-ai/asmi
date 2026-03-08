mod ane;
mod cli;
mod daemon;
mod daemon_startup;
mod serve;
mod topology;
mod watchdog;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use asmi_core::DiscoveryMethod;

/// Apple Silicon cluster monitor — like nvidia-smi + htop for Mac.
///
/// Apple Silicon Machine Intelligence.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Seed hostnames or IPs to probe (comma-separated).
    #[arg(long, value_delimiter = ',')]
    hosts: Vec<String>,

    /// Poll interval in seconds.
    #[arg(short, long, default_value_t = 2)]
    interval: u64,

    /// Output format [default: table].
    #[arg(short, long)]
    format: Option<Format>,

    /// Continuous watch mode (re-prints on each poll tick).
    #[arg(short, long)]
    watch: bool,

    /// Discovery methods to scan for nodes (comma-separated).
    #[arg(short, long, value_delimiter = ',')]
    scan: Vec<Scan>,

    /// Run as HTTP daemon serving metrics on the given port.
    #[arg(long)]
    serve: bool,

    /// Cluster hub mode: also poll all known nodes and expose /cluster endpoint.
    #[arg(long)]
    cluster: bool,

    /// Port for --serve mode (default: 9090).
    #[arg(long, default_value_t = 9090)]
    port: u16,

    /// Directories to scan for models (comma-separated).
    #[arg(long, value_delimiter = ',')]
    models_dir: Vec<String>,

    /// Enable experimental ANE compute endpoints (requires --features ane at build time).
    #[arg(long, hide = true)]
    experimental_ane: bool,

    #[command(subcommand)]
    command: Option<Command>,
}

#[derive(Debug, clap::Subcommand)]
enum Command {
    /// Manage asmi metrics daemons on cluster nodes.
    Daemon {
        #[command(subcommand)]
        action: DaemonAction,
    },
    /// Discover TB5/RDMA mesh topology via mlx.distributed_config.
    Topology {
        /// Hosts to check (comma-separated). Defaults to config nodes.
        #[arg(long, value_delimiter = ',')]
        hosts: Vec<String>,
        /// Output format.
        #[arg(long, default_value = "table")]
        format: TopologyFormat,
        /// JACCL backend variant.
        #[arg(long, default_value = "jaccl")]
        backend: String,
    },
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum DaemonAction {
    Status,
    Start { node: Option<String> },
    Stop { node: Option<String> },
    Restart { node: Option<String> },
    Deploy { node: Option<String> },
    Logs { node: Option<String> },
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum Format {
    Table,
    Json,
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum TopologyFormat {
    Table,
    Json,
    Dot,
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum Scan {
    Thunderbolt,
    Tailscale,
    Arp,
    ArpAll,
    Bonjour,
    Profiler,
}

impl From<&Scan> for DiscoveryMethod {
    fn from(s: &Scan) -> Self {
        match s {
            Scan::Thunderbolt => DiscoveryMethod::ThunderboltBridge,
            Scan::Tailscale => DiscoveryMethod::Tailscale,
            Scan::Arp => DiscoveryMethod::Arp,
            Scan::ArpAll => DiscoveryMethod::ArpAll,
            Scan::Bonjour => DiscoveryMethod::Bonjour,
            Scan::Profiler => DiscoveryMethod::SystemProfiler,
        }
    }
}

/// Get the binary name from argv[0].
pub(crate) fn bin_name() -> &'static str {
    static NAME: std::sync::OnceLock<String> = std::sync::OnceLock::new();
    NAME.get_or_init(|| {
        std::env::args()
            .next()
            .as_deref()
            .and_then(|s| s.rsplit('/').next())
            .unwrap_or("asmi")
            .to_string()
    })
}

#[tokio::main]
async fn main() -> Result<()> {
    let args = Cli::parse();

    // Subcommand dispatch
    if let Some(command) = args.command {
        return match command {
            Command::Daemon { action } => cli::run_daemon(action, args.port).await,
            Command::Topology {
                hosts,
                format,
                backend,
            } => {
                let hosts = if hosts.is_empty() {
                    let nm = asmi_core::config::NodeMap::load();
                    nm.nodes.clone()
                } else {
                    hosts
                };
                if hosts.is_empty() {
                    anyhow::bail!("No hosts specified. Use --hosts or configure nodes in ~/.config/asmi/config.json");
                }
                let report = topology::discover_topology(&hosts, &backend)?;
                match format {
                    TopologyFormat::Json => {
                        println!("{}", serde_json::to_string_pretty(&report)?);
                    }
                    TopologyFormat::Dot => {
                        print!("{}", report.raw_dot);
                    }
                    TopologyFormat::Table => {
                        print!("{}", topology::format_table(&report));
                    }
                }
                Ok(())
            }
        };
    }

    // --serve mode: run as HTTP daemon
    if args.serve {
        return daemon_startup::run_serve(
            args.port, args.interval, args.cluster,
            args.models_dir, args.experimental_ane,
        ).await;
    }

    // CLI monitor: one-shot or streaming watch
    let format = args.format.unwrap_or(Format::Table);
    cli::run_monitor(args.hosts, args.interval, format, args.watch, args.scan).await
}
