use anyhow::Result;
use clap::{Parser, ValueEnum};
use crossterm::{
    event::{self, Event, KeyCode, KeyEvent, KeyModifiers},
    terminal::{disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen},
    ExecutableCommand,
};
use apple_smi_core::{ClusterConfig, ClusterMonitor, ClusterState, DiscoveryMethod};
use ratatui::{
    layout::{Constraint, Direction, Layout, Rect},
    style::{Color, Modifier, Style},
    text::{Line, Span},
    widgets::{Block, Borders, Cell, Paragraph, Row, Table},
    Frame, Terminal,
};
use serde::{Deserialize, Serialize};
use std::io::{stdout, IsTerminal};
use std::path::PathBuf;
use std::time::Duration;

/// Apple Silicon cluster monitor — like nvidia-smi for Mac.
///
/// Also available as `asmi` and `mlx-smi`.
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
    Bonjour,
    Profiler,
}

impl From<&Scan> for DiscoveryMethod {
    fn from(s: &Scan) -> Self {
        match s {
            Scan::Thunderbolt => DiscoveryMethod::ThunderboltBridge,
            Scan::Tailscale => DiscoveryMethod::Tailscale,
            Scan::Arp => DiscoveryMethod::Arp,
            Scan::Bonjour => DiscoveryMethod::Bonjour,
            Scan::Profiler => DiscoveryMethod::SystemProfiler,
        }
    }
}

// ---------------------------------------------------------------------------
// Config file: ~/.config/apple-smi/config.toml
// ---------------------------------------------------------------------------

/// Persistent config stored at ~/.config/apple-smi/config.toml
#[derive(Debug, Default, Serialize, Deserialize)]
struct AppConfig {
    #[serde(default)]
    hosts: Vec<String>,
    #[serde(default)]
    interval: Option<u64>,
}

fn config_path() -> PathBuf {
    dirs::config_dir()
        .unwrap_or_else(|| PathBuf::from("~/.config"))
        .join("apple-smi")
        .join("config.toml")
}

fn load_config() -> AppConfig {
    let path = config_path();
    match std::fs::read_to_string(&path) {
        Ok(contents) => toml::from_str(&contents).unwrap_or_default(),
        Err(_) => AppConfig::default(),
    }
}

fn save_config(cfg: &AppConfig) {
    let path = config_path();
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    if let Ok(contents) = toml::to_string_pretty(cfg) {
        let _ = std::fs::write(&path, contents);
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Smart default: tui if interactive terminal, table if piped
    let format = cli.format.unwrap_or_else(|| {
        if stdout().is_terminal() { Format::Tui } else { Format::Table }
    });

    // TUI format implies watch mode
    let watch = cli.watch || matches!(format, Format::Tui);

    // Init tracing to file (never to stdout — corrupts TUI)
    let log_file = std::fs::File::create("/tmp/apple-smi.log")?;
    tracing_subscriber::fmt()
        .with_env_filter("apple_smi_core=info")
        .with_writer(std::sync::Mutex::new(log_file))
        .with_ansi(false)
        .init();

    // Resolve hosts: CLI flag > config file > discovery
    let app_config = load_config();
    let hosts = if !cli.hosts.is_empty() {
        cli.hosts
    } else if !app_config.hosts.is_empty() {
        app_config.hosts
    } else {
        Vec::new() // discovery will find them
    };
    let interval = cli.interval;

    // Start cluster monitor
    let mut config = ClusterConfig::default()
        .with_seeds(hosts)
        .with_poll_interval(Duration::from_secs(interval));

    if !cli.scan.is_empty() {
        config = config.with_discovery(cli.scan.iter().map(Into::into).collect());
    }

    let mut monitor = ClusterMonitor::new(config);
    let state = monitor.state();
    monitor.start();

    if !watch {
        // One-shot: wait for first scan + metrics, print, exit
        let mut rx = monitor.subscribe();
        for _ in 0..2 {
            let _ = tokio::time::timeout(Duration::from_secs(10), rx.changed()).await;
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
                // Clear screen for table on TTY (ephemeral)
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

    let mut tick = tokio::time::interval(Duration::from_millis(250));
    let mut selected: usize = 0;
    let started = std::time::Instant::now();

    loop {
        // Render
        let s = state.read().await;
        let elapsed = started.elapsed();
        terminal.draw(|f| render(f, &s, selected, elapsed))?;
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
                        (KeyCode::Char('q'), _) | (KeyCode::Esc, _) => break,
                        (KeyCode::Char('c'), KeyModifiers::CONTROL) => break,
                        (KeyCode::Up | KeyCode::Char('k'), _) => {
                            selected = selected.saturating_sub(1);
                        }
                        (KeyCode::Down | KeyCode::Char('j'), _) => {
                            let s = state.read().await;
                            let max = s.snapshots.len().saturating_sub(1);
                            selected = (selected + 1).min(max);
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
fn render(f: &mut Frame, state: &ClusterState, selected: usize, elapsed: Duration) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(4),  // Header / summary
            Constraint::Min(10),   // Node table
            Constraint::Length(1), // Footer
        ])
        .split(f.area());

    render_header(f, state, chunks[0]);

    if state.snapshots.is_empty() {
        render_loading(f, state, elapsed, chunks[1]);
    } else {
        render_nodes(f, state, selected, chunks[1]);
    }

    render_footer(f, chunks[2]);
}

fn render_loading(f: &mut Frame, state: &ClusterState, elapsed: Duration, area: Rect) {
    let secs = elapsed.as_secs();
    let spinner = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    let frame = spinner[(secs as usize * 4 + (elapsed.subsec_millis() / 250) as usize) % spinner.len()];

    let scan_count = state.scan_results.len();
    let status = if scan_count > 0 {
        format!("{frame}  Discovered {scan_count} nodes, polling metrics...")
    } else {
        format!("{frame}  Scanning cluster... ({secs}s)")
    };

    let content = vec![
        Line::raw(""),
        Line::from(Span::styled(
            status,
            Style::default().fg(Color::Yellow),
        )),
        Line::raw(""),
        Line::from(Span::styled(
            "  Seed hosts, Thunderbolt bridges, Tailscale peers...",
            Style::default().fg(Color::DarkGray),
        )),
    ];

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(
            " Nodes ",
            Style::default().fg(Color::White).add_modifier(Modifier::BOLD),
        ));
    let para = Paragraph::new(content).block(block);
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

    let content = vec![
        Line::from(vec![
            Span::styled(
                " apple-smi ",
                Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD),
            ),
            Span::raw("  "),
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
            Span::raw(" "),
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

fn render_nodes(f: &mut Frame, state: &ClusterState, selected: usize, area: Rect) {
    let header_cells = ["Node", "Chip", "CPU%", "GPU%", "RAM", "Power", "Processes"]
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
        .map(|name| {
            if let Some(snap) = state.snapshots.get(name) {
                if !snap.online {
                    return Row::new(vec![
                        Cell::from(name.as_str()),
                        Cell::from("--"),
                        Cell::from("--"),
                        Cell::from("--"),
                        Cell::from("offline"),
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
                            let name = p.model.as_deref()
                                .map(|m| m.rsplit('/').next().unwrap_or(m))
                                .unwrap_or(&p.framework.to_string())
                                .to_string();
                            if let Some(dist) = &p.distributed {
                                format!("{name} [{dist}]")
                            } else {
                                name
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(", ")
                };

                // Get chip from scan results
                let chip = state.scan_results
                    .iter()
                    .find(|r| r.hostname == *name)
                    .and_then(|r| r.chip.as_deref())
                    .unwrap_or("--");

                Row::new(vec![
                    Cell::from(name.as_str()).style(Style::default().fg(Color::White)),
                    Cell::from(chip).style(Style::default().fg(Color::DarkGray)),
                    Cell::from(format!("{:.0}%", snap.cpu_percent))
                        .style(Style::default().fg(usage_color(snap.cpu_percent))),
                    Cell::from(format!("{:.0}%", snap.gpu_percent))
                        .style(Style::default().fg(usage_color(snap.gpu_percent))),
                    Cell::from(format!("{:.0}/{:.0}G", snap.ram_used_gib(), snap.ram_total_gib()))
                        .style(Style::default().fg(Color::Cyan)),
                    Cell::from(format!("{:.1}W", snap.total_watts()))
                        .style(Style::default().fg(Color::Yellow)),
                    Cell::from(proc_info).style(Style::default().fg(Color::Magenta)),
                ])
            } else {
                Row::new(vec![
                    Cell::from(name.as_str()),
                    Cell::from("--"),
                    Cell::from("--"),
                    Cell::from("--"),
                    Cell::from("pending"),
                    Cell::from("--"),
                    Cell::from("--"),
                ])
                .style(Style::default().fg(Color::DarkGray))
            }
        })
        .collect();

    let widths = [
        Constraint::Length(10), // Node
        Constraint::Length(16), // Chip
        Constraint::Length(6),  // CPU%
        Constraint::Length(6),  // GPU%
        Constraint::Length(12), // RAM
        Constraint::Length(8),  // Power
        Constraint::Min(20),    // Processes
    ];

    let table = Table::new(rows, widths)
        .header(header)
        .block(
            Block::default()
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray))
                .title(Span::styled(
                    " Nodes ",
                    Style::default()
                        .fg(Color::White)
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

fn render_footer(f: &mut Frame, area: Rect) {
    let help = Paragraph::new(" q: quit | j/k: navigate | Ctrl+C: exit")
        .style(Style::default().fg(Color::DarkGray));
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

/// Print a one-shot table (like nvidia-smi default output)
fn print_table(state: &ClusterState) {
    let agg = &state.aggregates;
    println!("+{:-<74}+", "");
    println!("| {:<72} |", format!(
        "apple-smi   {}  nodes: {}/{}  power: {:.1}W  RAM: {:.0}/{:.0}GB",
        chrono::Local::now().format("%Y-%m-%d %H:%M:%S"),
        agg.nodes_online,
        agg.nodes_total,
        agg.total_watts,
        agg.total_ram_used_gib(),
        agg.total_ram_total_gib(),
    ));
    println!("+{:-<74}+", "");
    println!("| {:<10} {:<6} {:<6} {:<12} {:<8} {:<26} |",
        "Node", "CPU%", "GPU%", "RAM", "Power", "Model");
    println!("|{:-<74}|", "");

    for name in state.sorted_hostnames() {
        if let Some(snap) = state.snapshots.get(&name) {
            let proc_desc = if snap.processes.is_empty() {
                "--".to_string()
            } else {
                snap.processes.iter().map(|p| {
                    let name = p.model.as_deref()
                        .map(|m| m.rsplit('/').next().unwrap_or(m))
                        .unwrap_or(&p.framework.to_string())
                        .to_string();
                    if let Some(dist) = &p.distributed {
                        format!("{name} [{dist}]")
                    } else {
                        name
                    }
                }).collect::<Vec<_>>().join(", ")
            };
            println!("| {:<10} {:>4.0}% {:>4.0}% {:>4.0}/{:<4.0}GB {:>5.1}W  {:<26} |",
                name,
                snap.cpu_percent,
                snap.gpu_percent,
                snap.ram_used_gib(),
                snap.ram_total_gib(),
                snap.total_watts(),
                proc_desc,
            );
        }
    }
    println!("+{:-<74}+", "");
}

/// Print JSON output
fn print_json(state: &ClusterState) {
    let output = serde_json::json!({
        "timestamp": chrono::Utc::now().to_rfc3339(),
        "aggregates": {
            "total_watts": state.aggregates.total_watts,
            "ram_used_gib": state.aggregates.total_ram_used_gib(),
            "ram_total_gib": state.aggregates.total_ram_total_gib(),
            "cpu_avg_percent": state.aggregates.cpu_avg_percent,
            "gpu_avg_percent": state.aggregates.gpu_avg_percent,
            "nodes_online": state.aggregates.nodes_online,
            "nodes_total": state.aggregates.nodes_total,
            "models_loaded": state.aggregates.models_loaded,
        },
        "nodes": state.snapshots.iter().map(|(name, snap)| {
            serde_json::json!({
                "hostname": name,
                "online": snap.online,
                "cpu_percent": snap.cpu_percent,
                "gpu_percent": snap.gpu_percent,
                "ram_used_gib": snap.ram_used_gib(),
                "ram_total_gib": snap.ram_total_gib(),
                "total_watts": snap.total_watts(),
                "processes": snap.processes.iter().map(|p| {
                    serde_json::json!({
                        "pid": p.pid,
                        "framework": p.framework.to_string(),
                        "model": p.model,
                        "port": p.port,
                        "footprint_mb": p.footprint_mb,
                        "distributed": p.distributed.as_ref().map(|d| d.to_string()),
                    })
                }).collect::<Vec<_>>(),
            })
        }).collect::<Vec<_>>(),
    });
    println!("{}", serde_json::to_string_pretty(&output).unwrap());
}
