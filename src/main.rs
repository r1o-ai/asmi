mod daemon;
mod daemon_mgmt;

use anyhow::Result;
use clap::{Parser, ValueEnum};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use asmi_core::{
    ClusterConfig, ClusterEvent, ClusterMonitor, ClusterState, DiscoveryMethod, NodeMap, RdmaLink,
};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    symbols,
    text::{Line, Span},
    widgets::{Axis, Block, Borders, Cell, Chart, Dataset, Gauge, Paragraph, Row, Table},
    Frame, Terminal,
};
use std::io::{stdout, IsTerminal};
use std::sync::Arc;
use std::time::Duration;

/// Apple Silicon cluster monitor — like nvidia-smi + htop for Mac.
///
/// Apple Silicon Machine Intelligence.
#[derive(Parser, Debug)]
#[command(version, about)]
struct Cli {
    /// Seed hostnames or IPs to probe (comma-separated).
    /// If omitted, relies on discovery methods (--scan) to find nodes.
    #[arg(long, value_delimiter = ',')]
    hosts: Vec<String>,

    /// Poll interval in seconds.
    #[arg(short, long, default_value_t = 2)]
    interval: u64,

    /// Output format [default: tui if interactive, table if piped].
    #[arg(short, long)]
    format: Option<Format>,

    /// Continuous watch mode (default for tui format).
    #[arg(short, long)]
    watch: bool,

    /// Discovery methods to scan for nodes (comma-separated).
    #[arg(short, long, value_delimiter = ',')]
    scan: Vec<Scan>,

    /// Run as HTTP daemon serving metrics on the given port.
    /// Exposes /metrics, /health, and /processes endpoints.
    #[arg(long)]
    serve: bool,

    /// Cluster hub mode: also poll all known nodes and expose /cluster endpoint.
    /// Run this on the monitoring node (e.g. mini2) to aggregate the whole fleet.
    #[arg(long)]
    cluster: bool,

    /// Port for --serve mode (default: 9090).
    #[arg(long, default_value_t = 9090)]
    port: u16,

    /// Directories to scan for models (comma-separated). Defaults to ~/Models + HuggingFace cache.
    #[arg(long, value_delimiter = ',')]
    models_dir: Vec<String>,

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
}

#[derive(Debug, clap::Subcommand)]
pub(crate) enum DaemonAction {
    /// Show daemon status on all nodes.
    Status,
    /// Start daemon on all or a specific node.
    Start {
        /// Node to target (default: all).
        node: Option<String>,
    },
    /// Stop daemon on all or a specific node.
    Stop {
        /// Node to target (default: all).
        node: Option<String>,
    },
    /// Restart daemon on all or a specific node.
    Restart {
        /// Node to target (default: all).
        node: Option<String>,
    },
    /// Deploy binary + plist to remote nodes.
    Deploy {
        /// Node to target (default: all remotes).
        node: Option<String>,
    },
    /// Tail daemon log on a node.
    Logs {
        /// Node to read logs from (default: local).
        node: Option<String>,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum Format {
    Tui,
    Table,
    Json,
}

impl std::fmt::Display for Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Format::Tui => write!(f, "tui"),
            Format::Table => write!(f, "table"),
            Format::Json => write!(f, "json"),
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
enum Scan {
    Thunderbolt,
    Tailscale,
    Arp,
    /// Scan all ARP entries (including non-Mac devices — slower).
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

// ---------------------------------------------------------------------------
// Activity log — collects real-time events for TUI display
// ---------------------------------------------------------------------------

const SPINNER: &[char] = &['\u{280B}', '\u{2819}', '\u{2838}', '\u{2834}', '\u{2826}', '\u{2807}']; // ⠋⠙⠸⠴⠦⠇

struct ActivityLog {
    entries: Vec<(String, Color)>,
    probing_total: usize,
    probed_count: usize,
    phase: Phase,
    tick: usize,
}

#[derive(PartialEq)]
enum Phase {
    Discovery,
    Probing,
    Metrics,
}

impl ActivityLog {
    fn new() -> Self {
        Self {
            entries: Vec::new(),
            probing_total: 0,
            probed_count: 0,
            phase: Phase::Discovery,
            tick: 0,
        }
    }

    fn advance_tick(&mut self) {
        self.tick = self.tick.wrapping_add(1);
    }

    fn spinner(&self) -> char {
        SPINNER[self.tick % SPINNER.len()]
    }

    fn push(&mut self, text: String, color: Color) {
        let ts = chrono::Local::now().format("%H:%M:%S").to_string();
        self.entries.push((format!(" {ts}  {text}"), color));
    }

    fn handle_event(&mut self, event: &ClusterEvent) {
        match event {
            ClusterEvent::DiscoveryStarted { method } => {
                self.phase = Phase::Discovery;
                self.push(format!("Scanning {method}..."), Color::Cyan);
            }
            ClusterEvent::DiscoveryFound { method, count } => {
                let color = if *count > 0 { Color::Green } else { Color::DarkGray };
                self.push(format!("{method}: {count} peers"), color);
            }
            ClusterEvent::ProbingStarted { count } => {
                self.phase = Phase::Probing;
                self.probing_total = *count;
                self.probed_count = 0;
                self.push(format!("Probing {count} nodes..."), Color::Yellow);
            }
            ClusterEvent::NodeProbed { hostname, online, chip, ram_gb } => {
                self.probed_count += 1;
                if *online {
                    let info = match (chip.as_deref(), ram_gb) {
                        (Some(c), Some(r)) => format!(" -- {c}, {r}GB"),
                        (Some(c), None) => format!(" -- {c}"),
                        _ => String::new(),
                    };
                    self.push(format!("+ {hostname}{info}"), Color::Green);
                } else {
                    self.push(format!("x {hostname} -- unreachable"), Color::DarkGray);
                }
            }
            ClusterEvent::ScanComplete { online, total } => {
                self.push(format!("Scan complete: {online}/{total} online"), Color::Cyan);
            }
            ClusterEvent::MetricsPollStarted { count } => {
                self.phase = Phase::Metrics;
                self.push(format!("Polling metrics from {count} nodes..."), Color::Yellow);
            }
            ClusterEvent::MetricsReceived { hostname } => {
                self.push(format!("Metrics: {hostname}"), Color::DarkGray);
            }
            ClusterEvent::RegistrySaved { count } => {
                self.push(format!("Registry saved ({count} nodes)"), Color::Green);
            }
            ClusterEvent::AliasDiscovered { alias, canonical } => {
                self.push(format!("Alias: {alias} -> {canonical}"), Color::Blue);
            }
            ClusterEvent::RdmaIpsDiscovered { canonical, ips, .. } => {
                self.push(
                    format!("RDMA: {canonical} -> {}", ips.join(", ")),
                    Color::Cyan,
                );
            }
            ClusterEvent::RdmaLinkDiscovered {
                local_interface,
                remote_hostname,
                remote_ip,
                port_state,
                ..
            } => {
                let state = port_state
                    .map(|s| format!(" [{s}]"))
                    .unwrap_or_default();
                self.push(
                    format!("Link: {local_interface} -> {remote_hostname} ({remote_ip}){state}"),
                    Color::Blue,
                );
            }
            ClusterEvent::RdmaDeviceCorrelated {
                rdma_device,
                port_state,
                ..
            } => {
                let color = if *port_state == asmi_core::PortState::Active {
                    Color::Green
                } else {
                    Color::Red
                };
                self.push(format!("RDMA: {rdma_device} {port_state}"), color);
            }
        }
    }

    fn progress_ratio(&self) -> f64 {
        if self.probing_total == 0 {
            0.0
        } else {
            self.probed_count as f64 / self.probing_total as f64
        }
    }

    fn progress_label(&self) -> String {
        let s = self.spinner();
        match self.phase {
            Phase::Discovery => format!("{s} Discovering..."),
            Phase::Probing => format!("{s} Probing {}/{}", self.probed_count, self.probing_total),
            Phase::Metrics => format!("{s} Collecting metrics..."),
        }
    }

    fn last_entry(&self) -> Option<&str> {
        self.entries.last().map(|(t, _)| t.as_str())
    }
}

// ---------------------------------------------------------------------------
// Merge mode state for TUI
// ---------------------------------------------------------------------------

struct MergeMode {
    /// The node selected as the alias (to be merged INTO another).
    source: String,
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
    let cli = Cli::parse();

    // Subcommand dispatch
    if let Some(command) = cli.command {
        return match command {
            Command::Daemon { action } => daemon_mgmt::run_daemon(action, cli.port).await,
        };
    }

    // --serve mode: run as HTTP daemon (no TUI, no cluster discovery)
    if cli.serve {
        return run_serve(cli.port, cli.interval, cli.cluster, cli.models_dir).await;
    }

    // Smart default: tui if interactive terminal, table if piped
    let format = cli.format.unwrap_or_else(|| {
        if stdout().is_terminal() { Format::Tui } else { Format::Table }
    });

    // TUI format implies watch mode
    let watch = cli.watch || matches!(format, Format::Tui);

    // Init tracing to file (never to stdout — corrupts TUI)
    let log_dir = dirs::data_local_dir()
        .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
        .join("asmi");
    std::fs::create_dir_all(&log_dir)?;
    let log_file = std::fs::File::create(log_dir.join("asmi.log"))?;
    tracing_subscriber::fmt()
        .with_env_filter("asmi_core=info")
        .with_writer(std::sync::Mutex::new(log_file))
        .with_ansi(false)
        .init();

    // Load persistent NodeMap (aliases + known nodes)
    let node_map = NodeMap::load();

    // Resolve seed hosts: CLI --hosts > NodeMap known nodes > discovery
    let seeds = if !cli.hosts.is_empty() {
        cli.hosts
    } else {
        node_map.nodes.clone()
    };

    // Start cluster monitor
    let mut config = ClusterConfig::default()
        .with_seeds(seeds)
        .with_poll_interval(Duration::from_secs(cli.interval));

    if !cli.scan.is_empty() {
        config = config.with_discovery(cli.scan.iter().map(Into::into).collect());
    }

    let mut monitor = ClusterMonitor::new(config.clone(), node_map);
    let state = monitor.state();
    let node_map = monitor.node_map();

    // Subscribe to events BEFORE starting — otherwise early events are lost
    let events_rx_bg = monitor.events();

    monitor.start();

    // Background task: handle AliasDiscovered events → update NodeMap → save
    {
        let node_map = Arc::clone(&node_map);
        let mut events_rx = events_rx_bg;
        tokio::spawn(async move {
            loop {
                match events_rx.recv().await {
                    Ok(ClusterEvent::NodeProbed { hostname, online, .. }) => {
                        if online {
                            let mut nm = node_map.write().await;
                            if nm.register_node(&hostname) {
                                nm.save();
                                tracing::info!(
                                    node = hostname.as_str(),
                                    nodes = nm.nodes.len(),
                                    "node registered and saved"
                                );
                            }
                        }
                    }
                    Ok(ClusterEvent::AliasDiscovered { alias, canonical }) => {
                        let mut nm = node_map.write().await;
                        if nm.add_alias(alias, canonical) {
                            nm.save();
                            tracing::info!(
                                aliases = nm.aliases.len(),
                                nodes = nm.nodes.len(),
                                "node map updated and saved"
                            );
                        }
                    }
                    Ok(ClusterEvent::RdmaIpsDiscovered { canonical, ips, .. }) => {
                        let mut nm = node_map.write().await;
                        if nm.add_rdma_ips(&canonical, &ips) {
                            nm.save();
                            tracing::info!(
                                node = canonical.as_str(),
                                ips = ?ips,
                                "RDMA IPs discovered and saved"
                            );
                        }
                    }
                    Ok(ClusterEvent::RdmaLinkDiscovered {
                        local_interface,
                        local_ip,
                        remote_ip,
                        remote_hostname,
                        rdma_device,
                        port_state,
                    }) => {
                        let mut nm = node_map.write().await;
                        let link = RdmaLink {
                            local_interface,
                            local_ip,
                            remote_ip,
                            remote_hostname: remote_hostname.clone(),
                            rdma_device,
                            port_state,
                        };
                        if nm.add_rdma_link(link) {
                            nm.save();
                            tracing::info!(
                                remote = remote_hostname.as_str(),
                                links = nm.rdma_links.len(),
                                "RDMA link discovered and saved"
                            );
                        }
                    }
                    Ok(ClusterEvent::RdmaDeviceCorrelated {
                        interface,
                        rdma_device,
                        port_state,
                    }) => {
                        let mut nm = node_map.write().await;
                        let mut changed = false;
                        for link in &mut nm.rdma_links {
                            if link.local_interface == interface {
                                if link.rdma_device.as_deref() != Some(&rdma_device)
                                    || link.port_state != Some(port_state)
                                {
                                    link.rdma_device = Some(rdma_device.clone());
                                    link.port_state = Some(port_state);
                                    changed = true;
                                }
                            }
                        }
                        if changed {
                            nm.save();
                            tracing::info!(
                                device = rdma_device.as_str(),
                                state = %port_state,
                                "RDMA device state correlated"
                            );
                        }
                    }
                    Err(tokio::sync::broadcast::error::RecvError::Closed) => break,
                    _ => {} // Skip other events
                }
            }
        });
    }

    if !watch {
        // One-shot: wait for first scan + metrics, print, exit
        let mut rx = monitor.subscribe();
        for _ in 0..2 {
            let _ = tokio::time::timeout(Duration::from_secs(10), rx.changed()).await;
        }

        // Persist discovered nodes to config directly
        {
            let s = state.read().await;
            let mut nm = node_map.write().await;
            for result in &s.scan_results {
                if result.ssh_ok {
                    nm.register_node(&result.hostname);
                }
            }
            // Always save if we have nodes (event handler may have added them
            // but save() can race with shutdown)
            if !nm.nodes.is_empty() {
                nm.save();
            }
        }

        let s = state.read().await;
        match format {
            Format::Json => print_json(&s),
            Format::Table | Format::Tui => print_table(&s),
        }
        monitor.stop();
        return Ok(());
    }

    // Streaming mode: --watch with table/json → continuous stdout
    if !matches!(format, Format::Tui) {
        let mut rx = monitor.subscribe();
        // Wait for initial data
        for _ in 0..2 {
            let _ = tokio::time::timeout(Duration::from_secs(10), rx.changed()).await;
        }

        let is_tty = stdout().is_terminal();
        loop {
            let s = state.read().await;
            if is_tty {
                print!("\x1b[2J\x1b[H");
            }
            match format {
                Format::Json => print_json(&s),
                Format::Table | Format::Tui => print_table(&s),
            }
            drop(s);

            tokio::select! {
                _ = tokio::time::sleep(Duration::from_secs(cli.interval)) => {}
                _ = tokio::signal::ctrl_c() => break,
            }
        }
        monitor.stop();
        return Ok(());
    }

    // Interactive TUI mode
    enable_raw_mode()?;
    stdout().execute(EnterAlternateScreen)?;

    let backend = ratatui::backend::CrosstermBackend::new(stdout());
    let mut terminal = Terminal::new(backend)?;

    // Panic hook to restore terminal
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = stdout().execute(LeaveAlternateScreen);
        original_hook(info);
    }));

    let mut tick = tokio::time::interval(Duration::from_millis(200));
    let mut selected: usize = 0;
    let mut activity = ActivityLog::new();
    let mut events_rx = monitor.events();
    let mut merge_mode: Option<MergeMode> = None;
    let mut expanded_node: Option<String> = None;

    loop {
        // Drain all pending events
        loop {
            match events_rx.try_recv() {
                Ok(ev) => activity.handle_event(&ev),
                Err(_) => break,
            }
        }

        // Render
        activity.advance_tick();
        let s = state.read().await;
        let nm = node_map.read().await;
        terminal.draw(|f| render(f, &s, &nm, selected, &activity, merge_mode.as_ref(), expanded_node.as_deref()))?;
        let node_names = s.sorted_hostnames();
        drop(nm);
        drop(s);

        // Handle input
        tokio::select! {
            _ = tick.tick() => {}
            result = tokio::task::spawn_blocking(|| {
                event::poll(Duration::from_millis(100))
                    .ok()
                    .and_then(|ready| if ready { event::read().ok() } else { None })
            }) => {
                if let Ok(Some(Event::Key(KeyEvent { code, modifiers, .. }))) = result {
                    match (code, modifiers) {
                        (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => {
                            if merge_mode.is_some() {
                                merge_mode = None; // Cancel merge
                            } else {
                                break; // Quit
                            }
                        }
                        (KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                        (KeyCode::Up | KeyCode::Char('k'), _) => {
                            selected = selected.saturating_sub(1);
                        }
                        (KeyCode::Down | KeyCode::Char('j'), _) => {
                            let max = node_names.len().saturating_sub(1);
                            selected = (selected + 1).min(max);
                        }
                        (KeyCode::Enter, _) => {
                            // Toggle node detail panel
                            if merge_mode.is_none() {
                                if let Some(name) = node_names.get(selected) {
                                    if expanded_node.as_ref() == Some(name) {
                                        expanded_node = None;
                                    } else {
                                        expanded_node = Some(name.clone());
                                    }
                                }
                            }
                        }
                        (KeyCode::Char('m'), _) => {
                            // Merge mode: first press selects source, second press merges into target
                            if let Some(source_name) = node_names.get(selected) {
                                if let Some(ref merge) = merge_mode {
                                    // Second press: merge source → target (selected)
                                    if *source_name != merge.source {
                                        let mut nm = node_map.write().await;
                                        nm.add_alias(merge.source.clone(), source_name.clone());
                                        nm.save();
                                        activity.push(
                                            format!("Merged: {} -> {}", merge.source, source_name),
                                            Color::Green,
                                        );
                                        // Remove the alias node's snapshot
                                        let mut s = state.write().await;
                                        s.snapshots.remove(&merge.source);
                                    }
                                    merge_mode = None;
                                } else {
                                    // First press: select source node
                                    merge_mode = Some(MergeMode {
                                        source: source_name.clone(),
                                    });
                                    activity.push(
                                        format!("Merge: select target for {source_name} (m=confirm, Esc=cancel)"),
                                        Color::Yellow,
                                    );
                                }
                            }
                        }
                        _ => {}
                    }
                }
            }
        }
    }

    // Cleanup
    monitor.stop();
    disable_raw_mode()?;
    stdout().execute(LeaveAlternateScreen)?;

    Ok(())
}

/// Render the TUI dashboard
fn render(
    f: &mut Frame,
    state: &ClusterState,
    node_map: &NodeMap,
    selected: usize,
    activity: &ActivityLog,
    merge_mode: Option<&MergeMode>,
    expanded_node: Option<&str>,
) {
    if state.snapshots.is_empty() {
        // Loading mode: header + progress + activity log
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),  // Header
                Constraint::Length(3),  // Progress gauge
                Constraint::Min(6),    // Activity log
                Constraint::Length(1), // Footer
            ])
            .split(f.area());

        render_header(f, state, chunks[0]);
        render_progress(f, activity, chunks[1]);
        render_activity(f, activity, chunks[2]);
        render_footer(f, activity, merge_mode, expanded_node.is_some(), chunks[3]);
    } else if let Some(hostname) = expanded_node {
        // Expanded mode: charts take full screen (no node table)
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),  // Header
                Constraint::Length(3),  // Node summary bar
                Constraint::Min(8),    // Charts (full remaining space)
                Constraint::Length(1), // Footer
            ])
            .split(f.area());

        render_header(f, state, chunks[0]);
        render_node_summary(f, state, node_map, hostname, chunks[1]);
        render_node_detail(f, state, node_map, hostname, chunks[2]);
        render_footer(f, activity, merge_mode, true, chunks[3]);
    } else {
        // Normal mode: header + node table + footer
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(4),  // Header
                Constraint::Min(10),   // Node table
                Constraint::Length(1), // Footer with latest event
            ])
            .split(f.area());

        render_header(f, state, chunks[0]);
        render_nodes(f, state, node_map, selected, merge_mode, chunks[1]);
        render_footer(f, activity, merge_mode, false, chunks[2]);
    }
}

fn render_progress(f: &mut Frame, activity: &ActivityLog, area: Rect) {
    let ratio = activity.progress_ratio();
    let label = activity.progress_label();

    let gauge_color = match activity.phase {
        Phase::Discovery => Color::Cyan,
        Phase::Probing => Color::Yellow,
        Phase::Metrics => Color::Green,
    };

    let gauge = Gauge::default()
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .gauge_style(Style::default().fg(gauge_color).bg(Color::Rgb(30, 30, 40)))
        .ratio(ratio.min(1.0))
        .label(Span::styled(label, Style::default().fg(Color::White).add_modifier(Modifier::BOLD)));

    f.render_widget(gauge, area);
}

fn render_activity(f: &mut Frame, activity: &ActivityLog, area: Rect) {
    let inner_height = area.height.saturating_sub(2) as usize; // borders
    let skip = activity.entries.len().saturating_sub(inner_height);

    let lines: Vec<Line> = activity.entries.iter()
        .skip(skip)
        .map(|(text, color)| {
            Line::from(Span::styled(text.as_str(), Style::default().fg(*color)))
        })
        .collect();

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " Activity ",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));
    let para = Paragraph::new(lines).block(block);
    f.render_widget(para, area);
}

fn render_header(f: &mut Frame, state: &ClusterState, area: Rect) {
    let online = state.online_count();
    let total = if state.total_nodes > 0 { state.total_nodes } else { state.snapshots.len() };
    let agg = &state.aggregates;

    let status_color = match online {
        n if n == total && total > 0 => Color::Green,
        0 => Color::Red,
        _ => Color::Yellow,
    };

    //   ▀▘ ▄▀▄ ▄▀▀ █▄ ▄█ █   — apple + ASMI block-art
    //  ▜█▛ █▀█ ▄▄█ █ ▀ █ █
    let content = vec![
        Line::from(vec![
            Span::styled(" ▀▘ ", Style::default().fg(Color::Green)),
            Span::styled(
                "▄▀▄ ▄▀▀ █▄ ▄█ █  ",
                Style::default().fg(Color::Cyan),
            ),
            Span::styled(
                format!("{}/{} nodes", online, total),
                Style::default().fg(status_color).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
            Span::styled(
                format!("{:.1}W", agg.total_watts),
                Style::default().fg(Color::Yellow),
            ),
            Span::raw("  "),
            Span::styled(
                format!("{:.0}/{:.0} GB", agg.total_ram_used_gib(), agg.total_ram_total_gib()),
                Style::default().fg(Color::Cyan),
            ),
        ]),
        Line::from(vec![
            Span::styled("▜█▛ ", Style::default().fg(Color::White)),
            Span::styled(
                "█▀█ ▄▄█ █ ▀ █ █  ",
                Style::default().fg(Color::Cyan),
            ),
            Span::styled("CPU: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}%", agg.cpu_avg_percent),
                Style::default().fg(usage_color(agg.cpu_avg_percent)),
            ),
            Span::raw("  "),
            Span::styled("GPU: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                format!("{:.0}%", agg.gpu_avg_percent),
                Style::default().fg(usage_color(agg.gpu_avg_percent)),
            ),
            Span::raw("  "),
            Span::styled("Models: ", Style::default().fg(Color::DarkGray)),
            Span::styled(
                if agg.models_loaded.is_empty() {
                    "none".to_string()
                } else {
                    agg.models_loaded.join(", ")
                },
                Style::default().fg(Color::Magenta),
            ),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray));
    let para = Paragraph::new(content).block(block);
    f.render_widget(para, area);
}

fn render_nodes(
    f: &mut Frame,
    state: &ClusterState,
    node_map: &NodeMap,
    selected: usize,
    merge_mode: Option<&MergeMode>,
    area: Rect,
) {
    let header_cells = ["Node", "Chip", "TB", "CPU%", "GPU%", "RAM", "Power", "RDMA", "Processes"]
        .iter()
        .map(|h| {
            Cell::from(*h).style(
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            )
        });
    let header = Row::new(header_cells).height(1).bottom_margin(1);

    let names = state.sorted_hostnames();
    let rows: Vec<Row> = names
        .iter()
        .enumerate()
        .map(|(idx, name)| {
            // Merge mode highlighting
            let is_merge_source = merge_mode.as_ref().map_or(false, |m| m.source == *name);
            let is_merge_target = merge_mode.is_some() && !is_merge_source && idx == selected;

            if let Some(snap) = state.snapshots.get(name) {
                if !snap.online {
                    return Row::new(vec![
                        Cell::from(name.as_str()),
                        Cell::from("--"),
                        Cell::from("--"),
                        Cell::from("--"),
                        Cell::from("--"),
                        Cell::from("offline"),
                        Cell::from("--"),
                        Cell::from("--"),
                        Cell::from("--"),
                    ])
                    .style(Style::default().fg(Color::DarkGray));
                }

                let proc_info = if snap.processes.is_empty() {
                    "idle".to_string()
                } else {
                    snap.processes
                        .iter()
                        .map(|p| {
                            let name = if let Some(m) = p.server_models.first() {
                                m.id.rsplit('/').next().unwrap_or(&m.id).to_string()
                            } else if let Some(m) = p.model.as_deref() {
                                m.rsplit('/').next().unwrap_or(m).to_string()
                            } else {
                                p.framework.to_string()
                            };
                            if let Some(dist) = &p.distributed {
                                format!("{name} [{dist}]")
                            } else {
                                name
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                };

                // Get chip + link speed from scan results
                let scan = state.scan_results
                    .iter()
                    .find(|r| r.hostname == *name);
                let chip = scan
                    .and_then(|r| r.chip.as_deref())
                    .unwrap_or("--");
                let tb_speed = scan
                    .and_then(|r| r.link_speed.as_deref())
                    .unwrap_or("--");

                // Show aliases count if any point to this node
                let alias_count = node_map.aliases.values()
                    .filter(|v| v.as_str() == name.as_str())
                    .count();
                let node_display = if alias_count > 0 {
                    format!("{name} (+{alias_count})")
                } else {
                    name.clone()
                };

                // RDMA info: show links to this node with state
                let (rdma_info, rdma_color) = {
                    let scan = state.scan_results.iter().find(|r| r.hostname == *name);
                    let links: Vec<&asmi_core::RdmaLink> = node_map
                        .rdma_links
                        .iter()
                        .filter(|l| l.remote_hostname == *name)
                        .collect();

                    if !links.is_empty() {
                        // Show interface + state (e.g., "en3↑" or "en4↓")
                        // Flag 192.168 local IPs — these are non-link-local and
                        // may indicate the TB interface needs re-seating
                        let has_active = links.iter().any(|l| {
                            l.port_state == Some(asmi_core::PortState::Active)
                        });
                        let has_192 = links.iter().any(|l| l.local_ip.starts_with("192.168."));
                        let info = links
                            .iter()
                            .map(|l| {
                                let state_char = match (l.rdma_device.as_ref(), l.port_state) {
                                    (Some(_), Some(asmi_core::PortState::Active)) => "\u{2191}", // ↑ RDMA active
                                    (Some(_), Some(asmi_core::PortState::Down)) => "\u{2193}",   // ↓ RDMA down
                                    (Some(_), _) => "?",            // RDMA device exists, state unknown
                                    (None, _) => "\u{2014}",       // — no RDMA device (TB-only link)
                                };
                                format!("{}{state_char}", l.local_interface)
                            })
                            .collect::<Vec<_>>()
                            .join(" ");
                        let display = if has_192 {
                            format!("{info} !")
                        } else {
                            info
                        };
                        let color = if has_192 {
                            Color::Yellow // warn: non-link-local IP, may need reseat
                        } else if has_active {
                            Color::Green
                        } else {
                            Color::Red
                        };
                        (display, color)
                    } else if let Some(rdma) = scan.and_then(|s| s.rdma.as_ref()) {
                        let active = rdma.active_count();
                        let total = rdma.devices.len();
                        if rdma.enabled {
                            let color = if active > 0 { Color::Green } else { Color::Red };
                            (format!("{active}/{total}"), color)
                        } else {
                            ("off".to_string(), Color::DarkGray)
                        }
                    } else {
                        ("--".to_string(), Color::DarkGray)
                    }
                };

                // Merge mode visual feedback
                let row_style = if is_merge_source {
                    Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)
                } else if is_merge_target {
                    Style::default().fg(Color::Green).add_modifier(Modifier::BOLD)
                } else {
                    Style::default()
                };

                Row::new(vec![
                    Cell::from(node_display).style(Style::default().fg(Color::White)),
                    Cell::from(chip).style(Style::default().fg(Color::DarkGray)),
                    Cell::from(tb_speed).style(Style::default().fg(Color::Blue)),
                    Cell::from(format!("{:.0}%", snap.cpu_percent))
                        .style(Style::default().fg(usage_color(snap.cpu_percent))),
                    Cell::from(format!("{:.0}%", snap.gpu_percent))
                        .style(Style::default().fg(gpu_color(snap.gpu_percent))),
                    Cell::from(format!("{:.0}/{:.0}G", snap.ram_used_gib(), snap.ram_total_gib()))
                        .style(Style::default().fg(Color::Cyan)),
                    Cell::from(format!("{:.1}W", snap.total_watts()))
                        .style(Style::default().fg(Color::Yellow)),
                    Cell::from(rdma_info).style(Style::default().fg(rdma_color)),
                    Cell::from(proc_info).style(Style::default().fg(Color::Magenta)),
                ]).style(row_style)
            } else {
                Row::new(vec![
                    Cell::from(name.as_str()),
                    Cell::from("--"),
                    Cell::from("--"),
                    Cell::from("--"),
                    Cell::from("--"),
                    Cell::from("pending"),
                    Cell::from("--"),
                    Cell::from("--"),
                    Cell::from("--"),
                ])
                .style(Style::default().fg(Color::DarkGray))
            }
        })
        .collect();

    let widths = [
        Constraint::Length(14), // Node (wider for alias count)
        Constraint::Length(16), // Chip
        Constraint::Length(10), // TB link speed
        Constraint::Length(6),  // CPU%
        Constraint::Length(6),  // GPU%
        Constraint::Length(12), // RAM
        Constraint::Length(8),  // Power
        Constraint::Length(10), // RDMA
        Constraint::Min(20),    // Processes
    ];

    let title = if merge_mode.is_some() {
        " Nodes — MERGE MODE (m=confirm, Esc=cancel) "
    } else {
        " Nodes "
    };

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(
                    if merge_mode.is_some() { Color::Yellow } else { Color::DarkGray }
                ))
                .title(Span::styled(
                    title,
                    Style::default()
                        .fg(if merge_mode.is_some() { Color::Yellow } else { Color::White })
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .column_spacing(1)
        .row_highlight_style(
            Style::default()
                .bg(Color::Rgb(40, 40, 50))
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        )
        .highlight_symbol("> ");

    let mut table_state = ratatui::widgets::TableState::default();
    table_state.select(Some(selected));
    f.render_stateful_widget(table, area, &mut table_state);
}

/// Compact summary bar for the expanded node (replaces the full table).
fn render_node_summary(
    f: &mut Frame,
    state: &ClusterState,
    node_map: &NodeMap,
    hostname: &str,
    area: Rect,
) {
    let snap = state.snapshots.get(hostname);
    let scan = state.scan_results.iter().find(|r| r.hostname == hostname);
    let chip = scan.and_then(|r| r.chip.as_deref()).unwrap_or("--");
    let tb_speed = scan.and_then(|r| r.link_speed.as_deref());

    // RDMA summary
    let rdma_links = node_map.rdma_links_to(hostname);
    let rdma_summary = if rdma_links.is_empty() {
        String::new()
    } else {
        let parts: Vec<String> = rdma_links
            .iter()
            .map(|l| {
                let st = match (l.rdma_device.as_ref(), l.port_state) {
                    (Some(_), Some(asmi_core::PortState::Active)) => "\u{2191}",
                    (Some(_), Some(asmi_core::PortState::Down)) => "\u{2193}",
                    _ => "\u{2014}",
                };
                format!("{}{st}", l.local_interface)
            })
            .collect();
        format!("  RDMA: {}", parts.join(" "))
    };

    let (cpu_s, gpu_s, ram_s, power_s, proc_s) = snap
        .map(|s| {
            let procs = if s.processes.is_empty() {
                "idle".to_string()
            } else {
                let names: Vec<String> = s.processes
                    .iter()
                    .flat_map(|p| {
                        if !p.server_models.is_empty() {
                            p.server_models.iter()
                                .map(|m| m.id.rsplit('/').next().unwrap_or(&m.id).to_string())
                                .collect::<Vec<_>>()
                        } else if let Some(ref m) = p.model {
                            vec![m.rsplit('/').next().unwrap_or(m).to_string()]
                        } else {
                            vec![p.framework.to_string()]
                        }
                    })
                    .collect();
                names.join(", ")
            };
            (
                format!("CPU {:.0}%", s.cpu_percent),
                format!("GPU {:.0}%", s.gpu_percent),
                format!("{:.0}/{:.0}G", s.ram_used_gib(), s.ram_total_gib()),
                format!("{:.1}W", s.total_watts()),
                procs,
            )
        })
        .unwrap_or_else(|| ("--".into(), "--".into(), "--".into(), "--".into(), "--".into()));

    let content = vec![
        Line::from(vec![
            Span::styled(
                format!(" {hostname}"),
                Style::default()
                    .fg(Color::Cyan)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(format!("  {chip}"), Style::default().fg(Color::DarkGray)),
            Span::styled(
                tb_speed.map(|s| format!("  {s}")).unwrap_or_default(),
                Style::default().fg(Color::Blue),
            ),
            Span::raw("  "),
            Span::styled(cpu_s, Style::default().fg(usage_color(snap.map(|s| s.cpu_percent).unwrap_or(0.0)))),
            Span::raw("  "),
            Span::styled(gpu_s, Style::default().fg(gpu_color(snap.map(|s| s.gpu_percent).unwrap_or(0.0)))),
            Span::raw("  "),
            Span::styled(ram_s, Style::default().fg(Color::Cyan)),
            Span::raw("  "),
            Span::styled(power_s, Style::default().fg(Color::Yellow)),
            Span::styled(rdma_summary, Style::default().fg(Color::Green)),
        ]),
        Line::from(vec![
            Span::styled(" Models: ", Style::default().fg(Color::DarkGray)),
            Span::styled(proc_s, Style::default().fg(Color::Magenta)),
        ]),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::Cyan));
    let para = Paragraph::new(content).block(block);
    f.render_widget(para, area);
}

/// Full-screen detail panel with btop-style charts.
fn render_node_detail(
    f: &mut Frame,
    state: &ClusterState,
    _node_map: &NodeMap,
    hostname: &str,
    area: Rect,
) {
    let history = state.histories.get(hostname);
    let snap = state.snapshots.get(hostname);

    // Convert ring buffer to (x, y) chart points
    let to_chart_data = |ring: &std::collections::VecDeque<f64>| -> Vec<(f64, f64)> {
        let len = ring.len();
        ring.iter()
            .enumerate()
            .map(|(i, &v)| (i as f64 - len as f64, v))
            .collect()
    };

    // Layout: big CPU/GPU chart on top, Power + RAM on bottom
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(8),   // CPU+GPU combined chart (gets most space)
            Constraint::Length(7), // Power + RAM side by side
        ])
        .split(area);

    // ── CPU + GPU combined chart (two datasets, one tall chart) ──
    let cpu_points: Vec<(f64, f64)> = history
        .map(|h| to_chart_data(&h.cpu))
        .unwrap_or_default();
    let gpu_points: Vec<(f64, f64)> = history
        .map(|h| to_chart_data(&h.gpu))
        .unwrap_or_default();

    let cpu_label = snap
        .map(|s| format!("CPU {:.0}%", s.cpu_percent))
        .unwrap_or_else(|| "CPU --%".into());
    let gpu_label = snap
        .map(|s| format!("GPU {:.0}%", s.gpu_percent))
        .unwrap_or_else(|| "GPU --%".into());

    let x_min = cpu_points
        .first()
        .map(|p| p.0)
        .unwrap_or(-60.0)
        .min(gpu_points.first().map(|p| p.0).unwrap_or(-60.0));

    let datasets = vec![
        Dataset::default()
            .name(cpu_label)
            .marker(symbols::Marker::Braille)
            .graph_type(ratatui::widgets::GraphType::Line)
            .style(Style::default().fg(Color::Green))
            .data(&cpu_points),
        Dataset::default()
            .name(gpu_label)
            .marker(symbols::Marker::Braille)
            .graph_type(ratatui::widgets::GraphType::Line)
            .style(Style::default().fg(Color::Cyan))
            .data(&gpu_points),
    ];

    let chart = Chart::new(datasets)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(Span::styled(
                    " CPU / GPU ",
                    Style::default()
                        .fg(Color::White)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .x_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .bounds([x_min, 0.0]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, 100.0])
                .labels([
                    Span::styled("0%", Style::default().fg(Color::DarkGray)),
                    Span::styled("50%", Style::default().fg(Color::DarkGray)),
                    Span::styled("100%", Style::default().fg(Color::DarkGray).add_modifier(Modifier::BOLD)),
                ]),
        )
        .legend_position(Some(ratatui::widgets::LegendPosition::TopRight));
    f.render_widget(chart, rows[0]);

    // ── Bottom row: Power chart + RAM gauge ──
    let bot_cols = Layout::default()
        .direction(Direction::Horizontal)
        .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
        .split(rows[1]);

    // Power chart
    let power_points: Vec<(f64, f64)> = history
        .map(|h| to_chart_data(&h.power))
        .unwrap_or_default();
    let power_max = power_points
        .iter()
        .map(|p| p.1)
        .fold(1.0_f64, f64::max)
        .max(5.0)
        * 1.2; // headroom
    let power_label = snap
        .map(|s| format!("Power {:.1}W", s.total_watts()))
        .unwrap_or_else(|| "Power --W".into());
    let power_max_label = format!("{:.0}W", power_max);

    let x_bound = power_points.first().map(|p| p.0).unwrap_or(-60.0);
    let power_ds = Dataset::default()
        .name(power_label)
        .marker(symbols::Marker::Braille)
        .graph_type(ratatui::widgets::GraphType::Line)
        .style(Style::default().fg(Color::Magenta))
        .data(&power_points);
    let power_chart = Chart::new(vec![power_ds])
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(Span::styled(
                    " Power ",
                    Style::default()
                        .fg(Color::Magenta)
                        .add_modifier(Modifier::BOLD),
                )),
        )
        .x_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .bounds([x_bound, 0.0]),
        )
        .y_axis(
            Axis::default()
                .style(Style::default().fg(Color::DarkGray))
                .bounds([0.0, power_max])
                .labels([
                    Span::styled("0", Style::default().fg(Color::DarkGray)),
                    Span::styled(&power_max_label, Style::default().fg(Color::DarkGray)),
                ]),
        )
        .legend_position(Some(ratatui::widgets::LegendPosition::TopRight));
    f.render_widget(power_chart, bot_cols[0]);

    // RAM gauge block
    let ram_block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " Memory ",
            Style::default()
                .fg(Color::Cyan)
                .add_modifier(Modifier::BOLD),
        ));
    let ram_inner = ram_block.inner(bot_cols[1]);
    f.render_widget(ram_block, bot_cols[1]);

    if let Some(s) = snap {
        let ratio = if s.ram_total_bytes > 0 {
            s.ram_used_bytes as f64 / s.ram_total_bytes as f64
        } else {
            0.0
        };
        let ram_color = if ratio > 0.9 {
            Color::Red
        } else if ratio > 0.7 {
            Color::Yellow
        } else {
            Color::Cyan
        };

        // Stack: gauge + text info
        let ram_rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Length(1), // label
                Constraint::Length(1), // gauge
                Constraint::Length(1), // details
                Constraint::Min(0),
            ])
            .split(ram_inner);

        let label = Line::from(vec![
            Span::styled(
                format!(" {:.0}G", s.ram_used_gib()),
                Style::default()
                    .fg(Color::White)
                    .add_modifier(Modifier::BOLD),
            ),
            Span::styled(
                format!(" / {:.0}G", s.ram_total_gib()),
                Style::default().fg(Color::DarkGray),
            ),
            Span::styled(
                format!("  ({:.0}%)", ratio * 100.0),
                Style::default().fg(ram_color),
            ),
        ]);
        f.render_widget(Paragraph::new(label), ram_rows[0]);

        let gauge = Gauge::default()
            .gauge_style(
                Style::default()
                    .fg(ram_color)
                    .bg(Color::Rgb(30, 30, 40)),
            )
            .ratio(ratio.min(1.0));
        f.render_widget(gauge, ram_rows[1]);

        // GPU footprint if processes running
        let gpu_footprint: f64 = s
            .processes
            .iter()
            .filter_map(|p| p.footprint_mb)
            .sum();
        if gpu_footprint > 0.0 {
            let ft_line = Line::from(vec![
                Span::styled(" GPU mem: ", Style::default().fg(Color::DarkGray)),
                Span::styled(
                    format!("{:.1}G", gpu_footprint / 1024.0),
                    Style::default().fg(Color::Magenta),
                ),
            ]);
            f.render_widget(Paragraph::new(ft_line), ram_rows[2]);
        }
    }
}

fn render_footer(
    f: &mut Frame,
    activity: &ActivityLog,
    merge_mode: Option<&MergeMode>,
    detail_open: bool,
    area: Rect,
) {
    let text = if let Some(merge) = merge_mode {
        format!(" MERGE: {} -> ??? | Navigate to target, press m | Esc to cancel", merge.source)
    } else if detail_open {
        let last = activity.last_entry().unwrap_or("");
        format!(" q: quit | j/k: navigate | Enter: collapse | m: merge    {last}")
    } else {
        let last = activity.last_entry().unwrap_or("");
        format!(" q: quit | j/k: navigate | Enter: detail | m: merge    {last}")
    };
    let style = if merge_mode.is_some() {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::DarkGray)
    };
    let help = Paragraph::new(text).style(style);
    f.render_widget(help, area);
}

fn usage_color(percent: f64) -> Color {
    if percent >= 80.0 {
        Color::Red
    } else if percent >= 50.0 {
        Color::Yellow
    } else if percent >= 20.0 {
        Color::Green
    } else {
        Color::DarkGray
    }
}

/// GPU color: on Apple Silicon, 100% is normal (model serving).
/// Only idle is noteworthy.
fn gpu_color(percent: f64) -> Color {
    if percent >= 50.0 {
        Color::Cyan // active inference — normal
    } else if percent >= 10.0 {
        Color::Green // light work
    } else {
        Color::DarkGray // idle
    }
}

/// Print a one-shot table (like nvidia-smi default output)
fn print_table(state: &ClusterState) {
    let agg = &state.aggregates;
    println!("+{:-<82}+", "");
    println!("| {:<80} |", format!(
        "{}   {}  nodes: {}/{}  power: {:.1}W  RAM: {:.0}/{:.0}GB",
        bin_name(),
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
        agg.nodes_online,
        agg.nodes_total,
        agg.total_watts,
        agg.total_ram_used_gib(),
        agg.total_ram_total_gib(),
    ));
    println!("+{:-<82}+", "");
    println!("| {:<10} {:<10} {:<6} {:<6} {:<12} {:<8} {:<6} {:<14} |",
        "Node", "TB", "CPU%", "GPU%", "RAM", "Power", "RDMA", "Model");
    println!("|{:-<82}|", "");

    for name in state.sorted_hostnames() {
        if let Some(snap) = state.snapshots.get(&name) {
            let proc_desc = if snap.processes.is_empty() {
                "--".to_string()
            } else {
                snap.processes.iter().map(|p| {
                    let name = if let Some(m) = p.server_models.first() {
                        m.id.rsplit('/').next().unwrap_or(&m.id).to_string()
                    } else if let Some(m) = p.model.as_deref() {
                        m.rsplit('/').next().unwrap_or(m).to_string()
                    } else {
                        p.framework.to_string()
                    };
                    if let Some(dist) = &p.distributed {
                        format!("{name} [{dist}]")
                    } else {
                        name
                    }
                }).collect::<Vec<_>>().join(", ")
            };
            let scan = state.scan_results.iter().find(|r| r.hostname == name);
            let tb_speed = scan
                .and_then(|r| r.link_speed.as_deref())
                .unwrap_or("--");
            let rdma_info = scan
                .and_then(|r| r.rdma.as_ref())
                .map(|r| {
                    let a = r.active_count();
                    let t = r.devices.len();
                    format!("{a}/{t}")
                })
                .unwrap_or_else(|| "--".to_string());
            println!("| {:<10} {:<10} {:>4.0}% {:>4.0}% {:>4.0}/{:<4.0}GB {:>5.1}W  {:<6} {:<14} |",
                name,
                tb_speed,
                snap.cpu_percent,
                snap.gpu_percent,
                snap.ram_used_gib(),
                snap.ram_total_gib(),
                snap.total_watts(),
                rdma_info,
                proc_desc,
            );
        }
    }
    println!("+{:-<82}+", "");
}

/// Print JSON output
// ---------------------------------------------------------------------------
// HTTP daemon mode (--serve)
// ---------------------------------------------------------------------------

/// Collect hardware identity from `system_profiler SPHardwareDataType`.
/// Returns (chip_model, serial_number, model_name). Runs synchronously (once at startup).
fn collect_hardware_identity() -> (Option<String>, Option<String>, Option<String>) {
    let output = std::process::Command::new("system_profiler")
        .arg("SPHardwareDataType")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .unwrap_or_default();

    let mut chip_model = None;
    let mut serial_number = None;
    let mut model_name = None;

    for line in output.lines() {
        let line = line.trim();
        if let Some(v) = line.strip_prefix("Chip:") {
            chip_model = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("Serial Number (system):") {
            serial_number = Some(v.trim().to_string());
        } else if let Some(v) = line.strip_prefix("Model Name:") {
            model_name = Some(v.trim().to_string());
        }
    }

    (chip_model, serial_number, model_name)
}

/// Run asmi as an HTTP daemon serving local node metrics.
///
/// Polls local metrics every `interval` seconds and serves them via:
/// - GET /metrics   → full NodeSnapshot JSON (with hardware identity injected)
/// - GET /health    → lightweight health check
/// - GET /processes → MLX process list only
///
/// With `--cluster`:
/// - GET /cluster   → Vec<NodeSnapshot> for all known nodes (hub aggregator mode)
/// - GET /nodes     → list of known node hostnames from NodeMap
async fn run_serve(port: u16, interval: u64, cluster_hub: bool, cli_models_dir: Vec<String>) -> Result<()> {
    // Init tracing to stderr (no TUI to corrupt)
    tracing_subscriber::fmt()
        .with_env_filter("asmi_core=info,asmi=info")
        .with_ansi(true)
        .init();

    let hostname = std::process::Command::new("hostname")
        .arg("-s")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_else(|_| "unknown".to_string());

    tracing::info!(
        hostname = %hostname,
        port = port,
        interval_secs = interval,
        cluster_hub = cluster_hub,
        "{} daemon starting", bin_name()
    );

    // Collect hardware identity once at startup
    let (chip_model, serial_number, model_name) = collect_hardware_identity();
    tracing::info!(
        chip = chip_model.as_deref().unwrap_or("unknown"),
        serial = serial_number.as_deref().unwrap_or("unknown"),
        model = model_name.as_deref().unwrap_or("unknown"),
        "hardware identity"
    );

    // Config for local-only collection (no SSH, no discovery)
    let config = ClusterConfig::default()
        .with_poll_interval(Duration::from_secs(interval));

    // Shared state: latest snapshot from this node
    let snapshot: Arc<tokio::sync::RwLock<Option<asmi_core::NodeSnapshot>>> =
        Arc::new(tokio::sync::RwLock::new(None));
    let started_at = std::time::Instant::now();

    // Broadcast channel for SSE streaming — subscribers get pushed on every poll tick
    let (metrics_tx, _) = tokio::sync::broadcast::channel::<String>(16);

    // Hardware identity shared into handlers
    let hw_chip = chip_model.clone();
    let hw_serial = serial_number.clone();
    let hw_model = model_name.clone();

    // Background polling loop — collect local metrics every N seconds
    {
        let snapshot = Arc::clone(&snapshot);
        let config = config.clone();
        let hostname = hostname.clone();
        let hw_chip = hw_chip.clone();
        let hw_serial = hw_serial.clone();
        let hw_model = hw_model.clone();
        let metrics_tx = metrics_tx.clone();
        tokio::spawn(async move {
            loop {
                let mut snap = asmi_core::collect_node_metrics(&hostname, &config, true).await;
                // Inject hardware identity into every snapshot
                snap.chip_model = hw_chip.clone();
                snap.serial_number = hw_serial.clone();
                snap.model_name = hw_model.clone();
                tracing::debug!(
                    cpu = format!("{:.1}%", snap.cpu_percent),
                    gpu = format!("{:.1}%", snap.gpu_percent),
                    ram = format!("{:.1}/{:.1} GiB", snap.ram_used_gib(), snap.ram_total_gib()),
                    procs = snap.processes.len(),
                    "metrics collected"
                );
                // Broadcast serialized snapshot for SSE subscribers
                if let Ok(json) = serde_json::to_string(&snap) {
                    let _ = metrics_tx.send(json);
                }
                *snapshot.write().await = Some(snap);
                tokio::time::sleep(Duration::from_secs(interval)).await;
            }
        });
    }

    // Optional cluster hub: spin up ClusterMonitor to poll all known nodes
    let cluster_state: Option<Arc<tokio::sync::RwLock<asmi_core::ClusterState>>> = if cluster_hub {
        let node_map = asmi_core::NodeMap::load();
        if node_map.nodes.is_empty() {
            tracing::warn!("--cluster requested but NodeMap is empty; run `asmi` first to discover nodes");
            None
        } else {
            // Filter self from seed list — defense-in-depth against self-referencing
            // loop where the daemon fetches from its own HTTP endpoint.
            let remote_nodes: Vec<String> = node_map.nodes.iter()
                .filter(|n| n.as_str() != hostname.as_str())
                .cloned()
                .collect();
            tracing::info!(
                all_nodes = ?node_map.nodes,
                remote_nodes = ?remote_nodes,
                "cluster hub: polling {} remote nodes (excluded self: {})",
                remote_nodes.len(), hostname
            );
            let cfg = ClusterConfig::default()
                .with_seeds(remote_nodes)
                .with_poll_interval(Duration::from_secs(interval));
            let mut monitor = asmi_core::ClusterMonitor::new(cfg, node_map);
            let state = monitor.state();
            monitor.start();
            // Leak the monitor so it lives for the process lifetime.
            // ClusterMonitor::drop() sends a shutdown signal — we must not drop it.
            std::mem::forget(monitor);
            Some(state)
        }
    } else {
        None
    };

    // Probe runtime versions once at startup
    let runtime = Arc::new(daemon::probe_runtime().await);
    tracing::info!(
        python = runtime.python_version.as_deref().unwrap_or("none"),
        mlx = runtime.mlx_version.as_deref().unwrap_or("none"),
        "runtime probed"
    );

    // Model cache — populated by background scan loop
    let model_cache: Arc<tokio::sync::RwLock<Option<(Vec<asmi_core::LocalModel>, std::time::Instant)>>> =
        Arc::new(tokio::sync::RwLock::new(None));

    // Prepare model directories: use CLI override if provided, else defaults
    let model_dirs: Vec<std::path::PathBuf> = if cli_models_dir.is_empty() {
        asmi_core::default_model_dirs()
    } else {
        cli_models_dir.iter().map(std::path::PathBuf::from).collect()
    };

    // Background model scan — refresh every 60s
    {
        let model_cache = Arc::clone(&model_cache);
        let dirs = model_dirs.clone();
        tokio::spawn(async move {
            loop {
                let models = asmi_core::scan_models(&dirs);
                tracing::info!(count = models.len(), "model scan complete");
                *model_cache.write().await = Some((models, std::time::Instant::now()));
                tokio::time::sleep(Duration::from_secs(60)).await;
            }
        });
    }

    // Build axum router
    let app_state = daemon::AppState {
        snapshot,
        cluster_state,
        node_map: Arc::new(tokio::sync::RwLock::new(asmi_core::NodeMap::load())),
        hostname: hostname.clone(),
        started_at,
        metrics_tx: metrics_tx.clone(),
        model_cache,
        runtime,
    };

    let app = daemon::build_router(app_state);

    let addr = format!("0.0.0.0:{port}");
    tracing::info!(%addr, "HTTP server listening");
    if cluster_hub {
        eprintln!("{} cluster hub: http://{hostname}:{port}/cluster", bin_name());
    } else {
        eprintln!("{} daemon: http://{hostname}:{port}/metrics", bin_name());
    }

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}

fn print_json(state: &ClusterState) {
    // Use serde_json::to_value() on each struct so ALL fields are included
    // automatically. No more manual cherry-picking that drifts out of sync.
    let nodes: Vec<serde_json::Value> = state
        .snapshots
        .values()
        .map(|snap| serde_json::to_value(snap).unwrap_or_default())
        .collect();

    let output = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "aggregates": serde_json::to_value(&state.aggregates).unwrap_or_default(),
        "nodes": nodes,
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
