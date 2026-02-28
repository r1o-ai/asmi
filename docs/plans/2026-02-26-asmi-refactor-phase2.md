# asmi Daemon Refactor — Phase 2 API Extensions & Architecture Cleanup

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Extend the asmi HTTP daemon with 4 new endpoints (`/models`, `/logs`, `/runtime`, `/health/setup`), split the 2138-line `main.rs` monolith into modules, fix broken tests, and harden the cluster hub mode.

**Architecture:** The daemon (`--serve --port 9090`) runs on each cluster node and collects metrics via local shell commands every 2s. It serves a `NodeSnapshot` JSON at `GET /metrics`. Phase 2 adds 4 new endpoints that replace SSH-based operations from the r1o web app's `cluster-core.ts`. The binary is a workspace: `src/main.rs` (CLI + TUI + HTTP server) depends on `asmi-core` crate (`crates/cluster-monitor/`) for types, collection, SSH, scanning, and aggregation.

**Tech Stack:** Rust 1.85 (edition 2024), tokio, axum, serde, clap, ratatui, reqwest

---

## Codebase Map

```
apple-smi/
├── Cargo.toml                          # workspace root, binary crate
├── src/
│   └── main.rs                         # 2138 lines — CLI, TUI, HTTP daemon, daemon mgmt (NEEDS SPLIT)
├── crates/cluster-monitor/
│   ├── Cargo.toml                      # asmi-core library crate
│   └── src/
│       ├── lib.rs                      # public exports
│       ├── types.rs                    # NodeSnapshot, ProcessInfo, RdmaStatus, etc.
│       ├── collector.rs                # collect_node_metrics(), SSH commands, parsers
│       ├── config.rs                   # ClusterConfig, NodeMap
│       ├── monitor.rs                  # ClusterMonitor background polling
│       ├── scanner.rs                  # node discovery + probing (TB, ARP, Tailscale)
│       ├── ssh.rs                      # ssh_run(), local_run()
│       └── aggregator.rs              # ClusterState, ClusterAggregates
├── docs/plans/                         # implementation plans
└── ROADMAP.md                          # feature roadmap (v0.1–v0.6)
```

### Current HTTP Endpoints (daemon mode)
| Route | Method | Purpose |
|-------|--------|---------|
| `/metrics` | GET | Full `NodeSnapshot` JSON |
| `/health` | GET | `{ok, hostname, uptime_secs}` |
| `/processes` | GET | `{hostname, processes}` |
| `/stream` | GET | SSE push of snapshot every poll tick |
| `/cluster` | GET | `Vec<NodeSnapshot>` (hub mode only) |
| `/nodes` | GET | Known hostnames (hub mode only) |
| `/jaccl/config` | GET | RDMA hostfile matrix |
| `/jaccl/config` | POST | Generate + write JACCL hostfile |

### Key Consumers (r1o web app)
| r1o function | Current data source | Phase 2 replacement |
|---|---|---|
| `listNodeModels()` | SSH `du` scan (15s) | `GET /models` |
| `mlxDebug()` — log tailing | SSH `tail -50 logfile` | `GET /logs` |
| `mlxDebug()` — runtime info | SSH `python3 -c "import mlx"` | `GET /runtime` |
| setup-checks route | 6 SSH commands | `GET /health/setup` |

---

## Phase 0: Fix Broken Tests & Stabilize

### Task 1: Fix `mock_snapshot` compilation error in aggregator tests

The test fixture `mock_snapshot` in `aggregator.rs:177-195` is missing fields added to `NodeSnapshot`: `chip_model`, `serial_number`, `model_name`, `rdma`, `interface_ips`.

**Files:**
- Modify: `crates/cluster-monitor/src/aggregator.rs:177-195`

**Step 1: Read the current mock and identify missing fields**

The `NodeSnapshot` struct (types.rs) has these fields the mock is missing:
- `chip_model: Option<String>` (added for hardware identity)
- `serial_number: Option<String>`
- `model_name: Option<String>`
- `rdma: Option<RdmaStatus>` (added for RDMA status)
- `interface_ips: BTreeMap<String, Vec<String>>` (added for RDMA IP correlation)

**Step 2: Update mock_snapshot**

```rust
fn mock_snapshot(hostname: &str, online: bool) -> NodeSnapshot {
    NodeSnapshot {
        hostname: hostname.to_string(),
        online,
        timestamp: Utc::now(),
        chip_model: None,
        serial_number: None,
        model_name: None,
        cpu_watts: 5000.0,
        gpu_watts: 8000.0,
        ane_watts: 100.0,
        cpu_percent: 25.0,
        gpu_percent: 60.0,
        ram_used_bytes: 128 * 1024 * 1024 * 1024,
        ram_total_bytes: 512 * 1024 * 1024 * 1024,
        ram_percent: 25.0,
        cpu_temp_c: Some(42.0),
        gpu_temp_c: None,
        processes: vec![],
        top_tasks: vec![],
        rdma: None,
        interface_ips: std::collections::BTreeMap::new(),
    }
}
```

**Step 3: Run tests**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test --workspace`
Expected: All tests pass (currently ~51 tests)

**Step 4: Commit**

```bash
git add crates/cluster-monitor/src/aggregator.rs
git commit -m "fix(tests): add missing fields to mock_snapshot fixture"
```

---

## Phase 1: Split main.rs into Modules

`main.rs` is 2138 lines containing 5 distinct concerns. Split it before adding new endpoints.

### Task 2: Extract HTTP daemon into `src/daemon.rs`

The HTTP server logic (routes, handlers, AppState, SSE streaming) is ~300 lines embedded in `run_serve()`. Extract it into a dedicated module.

**Files:**
- Create: `src/daemon.rs`
- Modify: `src/main.rs`

**Step 1: Create `src/daemon.rs` with the axum router**

Move from `main.rs:1880-2138` into `daemon.rs`:
- `AppState` struct
- All handler functions: `metrics_handler`, `health_handler`, `processes_handler`, `cluster_handler`, `nodes_handler`, `stream_handler`, `jaccl_config_handler`, `jaccl_generate_handler`
- The `build_router()` function that wires them together

```rust
// src/daemon.rs
use axum::{extract::State, response::Json, routing::{get, post}, Router};
use std::sync::Arc;
use tokio::sync::RwLock;

#[derive(Clone)]
pub struct AppState {
    pub snapshot: Arc<RwLock<Option<asmi_core::NodeSnapshot>>>,
    pub cluster_state: Option<Arc<RwLock<asmi_core::ClusterState>>>,
    pub node_map: Arc<RwLock<asmi_core::NodeMap>>,
    pub hostname: String,
    pub started_at: std::time::Instant,
    pub metrics_tx: tokio::sync::broadcast::Sender<String>,
}

pub fn build_router(state: AppState) -> Router {
    Router::new()
        .route("/metrics", get(metrics_handler))
        .route("/health", get(health_handler))
        .route("/processes", get(processes_handler))
        .route("/stream", get(stream_handler))
        .route("/cluster", get(cluster_handler))
        .route("/nodes", get(nodes_handler))
        .route("/jaccl/config", get(jaccl_config_handler).post(jaccl_generate_handler))
        .with_state(state)
}

// ... move all handler functions here unchanged ...
```

**Step 2: Update main.rs to use the module**

In `main.rs`:
```rust
mod daemon;

// In run_serve(), replace inline router construction with:
let app = daemon::build_router(daemon::AppState {
    snapshot: Arc::clone(&snapshot),
    cluster_state: cluster_state.clone(),
    node_map,
    hostname: hostname.clone(),
    started_at,
    metrics_tx: metrics_tx.clone(),
});
```

**Step 3: Run tests + cargo check**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check && cargo test --workspace`
Expected: Compiles, all tests pass

**Step 4: Commit**

```bash
git add src/daemon.rs src/main.rs
git commit -m "refactor: extract HTTP daemon into src/daemon.rs"
```

---

### Task 3: Extract daemon management into `src/daemon_mgmt.rs`

The `daemon` subcommand handler (`daemon status|start|stop|restart|deploy|logs`) is ~200 lines. Extract it.

**Files:**
- Create: `src/daemon_mgmt.rs`
- Modify: `src/main.rs`

**Step 1: Create `src/daemon_mgmt.rs`**

Move the daemon subcommand match arms (status, start, stop, restart, deploy, logs) into:
```rust
// src/daemon_mgmt.rs
pub async fn handle_daemon_command(action: &str, target: Option<&str>) -> anyhow::Result<()> {
    // ... existing daemon management code ...
}
```

**Step 2: Update main.rs**

Replace the inline match with:
```rust
mod daemon_mgmt;

// In main():
"daemon" => daemon_mgmt::handle_daemon_command(&action, target.as_deref()).await?,
```

**Step 3: Run cargo check**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check`
Expected: Compiles clean

**Step 4: Commit**

```bash
git add src/daemon_mgmt.rs src/main.rs
git commit -m "refactor: extract daemon management into src/daemon_mgmt.rs"
```

---

## Phase 2: New API Endpoints

### Task 4: `GET /models` — Local model file enumeration

Replaces `listNodeModels()` in r1o's `cluster-core.ts` which runs a 15-second SSH `du` scan. The daemon scans model directories on startup and caches the result, refreshing every 60s.

**Files:**
- Create: `crates/cluster-monitor/src/models.rs` (model discovery logic)
- Modify: `crates/cluster-monitor/src/lib.rs` (export new module)
- Modify: `src/daemon.rs` (add route + handler)

**Step 1: Write the failing test**

```rust
// crates/cluster-monitor/src/models.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_model_dir_name() {
        // HuggingFace convention: "org--model-name" on disk → "org/model-name" display
        assert_eq!(
            parse_model_name("Qwen--Qwen3-32B-4bit"),
            "Qwen/Qwen3-32B-4bit"
        );
    }

    #[test]
    fn test_parse_model_dir_no_separator() {
        assert_eq!(
            parse_model_name("my-local-model"),
            "my-local-model"
        );
    }
}
```

**Step 2: Run test to verify it fails**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test -p asmi-core test_parse_model`
Expected: FAIL — module doesn't exist

**Step 3: Implement model discovery**

```rust
// crates/cluster-monitor/src/models.rs
//! Local model file discovery — scans known directories for downloaded models.

use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use tracing::debug;

/// A model found on the local filesystem.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LocalModel {
    /// Display name: "org/model-name" (HuggingFace convention)
    pub name: String,
    /// Absolute path on disk
    pub path: PathBuf,
    /// Size in bytes (sum of all files in the directory)
    pub size_bytes: u64,
}

/// Response for GET /models
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelsResponse {
    pub models: Vec<LocalModel>,
    /// How old the cached scan is, in seconds
    pub scan_age_seconds: u64,
}

/// Default directories to scan for models (macOS).
/// Users typically store models in ~/Models or the HuggingFace cache.
pub fn default_model_dirs() -> Vec<PathBuf> {
    let home = dirs::home_dir().unwrap_or_else(|| PathBuf::from("/tmp"));
    vec![
        home.join("Models"),
        home.join(".cache/huggingface/hub"),
    ]
}

/// Parse a model directory name into a display name.
/// HuggingFace convention: "Qwen--Qwen3-32B-4bit" → "Qwen/Qwen3-32B-4bit"
pub fn parse_model_name(dir_name: &str) -> String {
    dir_name.replacen("--", "/", 1)
}

/// Scan directories for model folders.
/// A "model" is a directory containing at least one `.safetensors` or `.gguf` file.
pub fn scan_models(dirs: &[PathBuf]) -> Vec<LocalModel> {
    let mut models = Vec::new();

    for dir in dirs {
        if !dir.is_dir() {
            continue;
        }
        debug!(dir = %dir.display(), "scanning for models");

        let entries = match std::fs::read_dir(dir) {
            Ok(e) => e,
            Err(_) => continue,
        };

        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() {
                continue;
            }

            // Check if this directory contains model files
            let has_model_files = std::fs::read_dir(&path)
                .map(|entries| {
                    entries.flatten().any(|e| {
                        let name = e.file_name();
                        let name = name.to_string_lossy();
                        name.ends_with(".safetensors")
                            || name.ends_with(".gguf")
                            || name == "config.json"
                    })
                })
                .unwrap_or(false);

            if !has_model_files {
                continue;
            }

            let dir_name = path.file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_default();

            // Calculate total size (non-recursive, just top-level files)
            let size_bytes: u64 = std::fs::read_dir(&path)
                .map(|entries| {
                    entries.flatten()
                        .filter_map(|e| e.metadata().ok())
                        .filter(|m| m.is_file())
                        .map(|m| m.len())
                        .sum()
                })
                .unwrap_or(0);

            models.push(LocalModel {
                name: parse_model_name(&dir_name),
                path,
                size_bytes,
            });
        }
    }

    models.sort_by(|a, b| a.name.cmp(&b.name));
    models
}
```

**Step 4: Register in lib.rs**

Add to `crates/cluster-monitor/src/lib.rs`:
```rust
pub mod models;
pub use models::{LocalModel, ModelsResponse, scan_models, default_model_dirs};
```

**Step 5: Add the route handler in daemon.rs**

```rust
// In daemon.rs, add to AppState:
pub model_cache: Arc<RwLock<Option<(Vec<asmi_core::LocalModel>, std::time::Instant)>>>,

// Handler:
async fn models_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let cache = state.model_cache.read().await;
    let (models, scanned_at) = match cache.as_ref() {
        Some((m, t)) => (m.clone(), t.elapsed().as_secs()),
        None => (vec![], 0),
    };
    Json(serde_json::json!({
        "models": models,
        "scan_age_seconds": scanned_at,
    }))
}

// Add to router:
.route("/models", get(models_handler))
```

**Step 6: Add background model scan loop in run_serve()**

In `main.rs` `run_serve()`, after the metrics polling loop, add:
```rust
// Background model scan — refresh every 60s
{
    let model_cache = Arc::clone(&model_cache);
    tokio::spawn(async move {
        let dirs = asmi_core::default_model_dirs();
        loop {
            let models = asmi_core::scan_models(&dirs);
            tracing::info!(count = models.len(), "model scan complete");
            *model_cache.write().await = Some((models, std::time::Instant::now()));
            tokio::time::sleep(Duration::from_secs(60)).await;
        }
    });
}
```

**Step 7: Run tests**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test --workspace`
Expected: All tests pass including new model tests

**Step 8: Commit**

```bash
git add crates/cluster-monitor/src/models.rs crates/cluster-monitor/src/lib.rs src/daemon.rs src/main.rs
git commit -m "feat(daemon): GET /models endpoint — local model file enumeration"
```

---

### Task 5: `GET /logs` — Server log tailing

Replaces SSH-based log tailing in `mlxDebug()`. Returns the last N lines from a named log file.

**Files:**
- Modify: `src/daemon.rs` (add route + handler)

**Step 1: Implement the handler**

```rust
/// GET /logs?name=mlx-server&lines=50
/// Tails log files from known locations.
async fn logs_handler(
    State(state): State<AppState>,
    axum::extract::Query(params): axum::extract::Query<std::collections::HashMap<String, String>>,
) -> Json<serde_json::Value> {
    let name = params.get("name").cloned().unwrap_or_else(|| "mlx-server".to_string());
    let lines: usize = params.get("lines")
        .and_then(|l| l.parse().ok())
        .unwrap_or(50)
        .min(500); // cap at 500 lines

    // Map log names to file paths
    let log_path = match name.as_str() {
        "mlx-server" | "mlx_lm" => "/tmp/r1o-mlx_lm-server.log",
        "mlx-vlm" | "mlx_vlm" => "/tmp/r1o-mlx_vlm-server.log",
        "vllm" | "vllm_mlx" => "/tmp/r1o-vllm_mlx-server.log",
        "asmi" | "daemon" => "~/Library/Logs/asmi-daemon.log",
        _ => {
            return Json(serde_json::json!({
                "error": format!("unknown log name: {name}"),
                "known_names": ["mlx-server", "mlx-vlm", "vllm", "asmi"],
            }));
        }
    };

    // Expand ~ to HOME
    let expanded = log_path.replace('~', &std::env::var("HOME").unwrap_or_default());

    match std::fs::read_to_string(&expanded) {
        Ok(content) => {
            let all_lines: Vec<&str> = content.lines().collect();
            let start = all_lines.len().saturating_sub(lines);
            let tail: Vec<&str> = all_lines[start..].to_vec();
            Json(serde_json::json!({
                "name": name,
                "path": expanded,
                "lines": tail,
                "total_lines": all_lines.len(),
            }))
        }
        Err(e) => Json(serde_json::json!({
            "name": name,
            "path": expanded,
            "error": format!("could not read log: {e}"),
        })),
    }
}
```

**Step 2: Add route**

```rust
.route("/logs", get(logs_handler))
```

**Step 3: Run cargo check**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check`
Expected: Compiles clean

**Step 4: Commit**

```bash
git add src/daemon.rs
git commit -m "feat(daemon): GET /logs endpoint — server log tailing"
```

---

### Task 6: `GET /runtime` — Python/MLX version info

Replaces SSH-based `python3 -c "import mlx.core"` in `mlxDebug()`. Discovers and caches Python/MLX/vLLM versions on startup.

**Files:**
- Modify: `src/daemon.rs` (add route + handler)
- Modify: `src/main.rs` (add runtime probe at startup)

**Step 1: Add runtime info collection**

```rust
// In daemon.rs or a new src/runtime.rs:

#[derive(Clone, Serialize, Deserialize)]
pub struct RuntimeInfo {
    pub python_version: Option<String>,
    pub mlx_version: Option<String>,
    pub mlx_device: Option<String>,
    pub vllm_version: Option<String>,
    pub macos_version: Option<String>,
}

/// Probe the local Python environment for ML framework versions.
/// Runs shell commands — call once at startup, cache result.
pub async fn probe_runtime() -> RuntimeInfo {
    use tokio::process::Command;

    let python = Command::new("python3")
        .args(["-c", "import sys; print(sys.version.split()[0])"])
        .output().await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    let mlx = Command::new("python3")
        .args(["-c", "import mlx.core as mx; print(mx.__version__); print(mx.default_device())"])
        .output().await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    let (mlx_version, mlx_device) = match mlx {
        Some(output) => {
            let mut lines = output.lines();
            (
                lines.next().map(|s| s.to_string()),
                lines.next().map(|s| s.to_string()),
            )
        }
        None => (None, None),
    };

    let vllm = Command::new("python3")
        .args(["-c", "import vllm; print(vllm.__version__)"])
        .output().await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    let macos = Command::new("sw_vers")
        .args(["-productVersion"])
        .output().await
        .ok()
        .filter(|o| o.status.success())
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());

    RuntimeInfo {
        python_version: python,
        mlx_version,
        mlx_device,
        vllm_version: vllm,
        macos_version: macos,
    }
}
```

**Step 2: Add to AppState and handler**

```rust
// In AppState:
pub runtime: Arc<RuntimeInfo>,

// Handler:
async fn runtime_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    Json(serde_json::to_value(state.runtime.as_ref())
        .unwrap_or(serde_json::json!({"error": "no runtime info"})))
}

// Route:
.route("/runtime", get(runtime_handler))
```

**Step 3: Probe at startup in run_serve()**

```rust
// Before building AppState:
let runtime = Arc::new(probe_runtime().await);
tracing::info!(
    python = runtime.python_version.as_deref().unwrap_or("none"),
    mlx = runtime.mlx_version.as_deref().unwrap_or("none"),
    "runtime probed"
);
```

**Step 4: Run cargo check**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check`
Expected: Compiles clean

**Step 5: Commit**

```bash
git add src/daemon.rs src/main.rs
git commit -m "feat(daemon): GET /runtime endpoint — Python/MLX/macOS version info"
```

---

### Task 7: `GET /health/setup` — Setup validation checks

Replaces 6 SSH commands in the r1o web app's setup-checks route. Runs a battery of local checks and returns structured results.

**Files:**
- Create: `crates/cluster-monitor/src/health.rs` (setup check logic)
- Modify: `crates/cluster-monitor/src/lib.rs` (export)
- Modify: `src/daemon.rs` (add route)

**Step 1: Write the test**

```rust
// crates/cluster-monitor/src/health.rs
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_result_pass() {
        let c = CheckResult { id: "test".into(), pass: true, detail: Some("ok".into()) };
        assert!(c.pass);
    }

    #[test]
    fn test_check_result_fail() {
        let c = CheckResult { id: "test".into(), pass: false, detail: Some("missing".into()) };
        assert!(!c.pass);
    }
}
```

**Step 2: Implement health checks**

```rust
// crates/cluster-monitor/src/health.rs
//! Node health/setup validation checks.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CheckResult {
    pub id: String,
    pub pass: bool,
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SetupChecks {
    pub checks: Vec<CheckResult>,
    pub all_pass: bool,
}

/// Run all setup validation checks. Blocking I/O — run in spawn_blocking or async context.
pub async fn run_setup_checks() -> SetupChecks {
    let mut checks = Vec::new();

    // 1. Python + MLX importable
    checks.push(check_python_mlx().await);

    // 2. RDMA devices present
    checks.push(check_rdma().await);

    // 3. Disk space sufficient (>50GB free on /)
    checks.push(check_disk_space().await);

    // 4. SSH keys present
    checks.push(check_ssh_keys().await);

    // 5. zshenv configured (PATH includes .cargo/bin and conda)
    checks.push(check_zshenv().await);

    let all_pass = checks.iter().all(|c| c.pass);
    SetupChecks { checks, all_pass }
}

async fn check_python_mlx() -> CheckResult {
    let output = tokio::process::Command::new("python3")
        .args(["-c", "import mlx.core as mx; print(f'MLX {mx.__version__}')"])
        .output().await;
    match output {
        Ok(o) if o.status.success() => CheckResult {
            id: "python-mlx".into(),
            pass: true,
            detail: Some(String::from_utf8_lossy(&o.stdout).trim().to_string()),
        },
        Ok(o) => CheckResult {
            id: "python-mlx".into(),
            pass: false,
            detail: Some(String::from_utf8_lossy(&o.stderr).trim().to_string()),
        },
        Err(e) => CheckResult {
            id: "python-mlx".into(),
            pass: false,
            detail: Some(format!("python3 not found: {e}")),
        },
    }
}

async fn check_rdma() -> CheckResult {
    let output = tokio::process::Command::new("sh")
        .args(["-c", "ibv_devices 2>/dev/null | grep -c rdma"])
        .output().await;
    match output {
        Ok(o) if o.status.success() => {
            let count: i32 = String::from_utf8_lossy(&o.stdout).trim().parse().unwrap_or(0);
            CheckResult {
                id: "rdma".into(),
                pass: count > 0,
                detail: Some(format!("{count} devices")),
            }
        }
        _ => CheckResult {
            id: "rdma".into(),
            pass: false,
            detail: Some("ibv_devices not available".into()),
        },
    }
}

async fn check_disk_space() -> CheckResult {
    let output = tokio::process::Command::new("sh")
        .args(["-c", "df -g / | awk 'NR==2{print $4}'"])
        .output().await;
    match output {
        Ok(o) if o.status.success() => {
            let gb: u64 = String::from_utf8_lossy(&o.stdout).trim().parse().unwrap_or(0);
            CheckResult {
                id: "disk-space".into(),
                pass: gb >= 50,
                detail: Some(format!("{gb}GB free")),
            }
        }
        _ => CheckResult {
            id: "disk-space".into(),
            pass: false,
            detail: Some("could not check disk space".into()),
        },
    }
}

async fn check_ssh_keys() -> CheckResult {
    let home = dirs::home_dir().unwrap_or_default();
    let ed25519 = home.join(".ssh/id_ed25519");
    let rsa = home.join(".ssh/id_rsa");
    let pass = ed25519.exists() || rsa.exists();
    CheckResult {
        id: "ssh-keys".into(),
        pass,
        detail: if pass { Some("found".into()) } else { Some("no SSH keys in ~/.ssh/".into()) },
    }
}

async fn check_zshenv() -> CheckResult {
    let home = dirs::home_dir().unwrap_or_default();
    let zshenv = home.join(".zshenv");
    match std::fs::read_to_string(&zshenv) {
        Ok(content) => {
            let has_cargo = content.contains(".cargo/bin");
            CheckResult {
                id: "zshenv".into(),
                pass: has_cargo,
                detail: if has_cargo { Some("configured".into()) } else { Some("missing .cargo/bin in PATH".into()) },
            }
        }
        Err(_) => CheckResult {
            id: "zshenv".into(),
            pass: false,
            detail: Some("~/.zshenv not found".into()),
        },
    }
}
```

**Step 3: Register in lib.rs + add route**

In `lib.rs`:
```rust
pub mod health;
pub use health::{CheckResult, SetupChecks, run_setup_checks};
```

In `daemon.rs`:
```rust
async fn setup_handler(State(_state): State<AppState>) -> Json<serde_json::Value> {
    let checks = asmi_core::run_setup_checks().await;
    Json(serde_json::to_value(&checks)
        .unwrap_or(serde_json::json!({"error": "check failed"})))
}

// Route:
.route("/health/setup", get(setup_handler))
```

**Step 4: Run tests**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test --workspace`
Expected: All tests pass

**Step 5: Commit**

```bash
git add crates/cluster-monitor/src/health.rs crates/cluster-monitor/src/lib.rs src/daemon.rs
git commit -m "feat(daemon): GET /health/setup endpoint — node setup validation"
```

---

## Phase 3: Harden Cluster Hub Mode

### Task 8: Add `--models-dir` CLI flag for custom model paths

**Files:**
- Modify: `src/main.rs` (add CLI arg)
- Modify: `src/daemon.rs` (pass model dirs into AppState)

**Step 1: Add CLI argument**

```rust
// In Cli struct:
/// Directories to scan for models (comma-separated). Defaults to ~/Models.
#[arg(long, value_delimiter = ',')]
models_dir: Vec<String>,
```

**Step 2: Wire into run_serve**

```rust
let model_dirs = if cli.models_dir.is_empty() {
    asmi_core::default_model_dirs()
} else {
    cli.models_dir.iter().map(PathBuf::from).collect()
};
```

**Step 3: Run cargo check + commit**

```bash
git add src/main.rs src/daemon.rs
git commit -m "feat(cli): --models-dir flag for custom model scan directories"
```

---

### Task 9: Add uptime and version to `/health` response

The web app currently shows "unknown" for uptime when using asmi. Add it to the health endpoint.

**Files:**
- Modify: `src/daemon.rs` (extend health_handler)

**Step 1: Extend health response**

```rust
async fn health_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let has_data = state.snapshot.read().await.is_some();
    let snap = state.snapshot.read().await;
    let process_count = snap.as_ref().map(|s| s.processes.len()).unwrap_or(0);
    Json(serde_json::json!({
        "ok": has_data,
        "hostname": state.hostname,
        "uptime_secs": state.started_at.elapsed().as_secs(),
        "version": env!("CARGO_PKG_VERSION"),
        "process_count": process_count,
    }))
}
```

**Step 2: Commit**

```bash
git add src/daemon.rs
git commit -m "feat(daemon): add version and process_count to /health response"
```

---

### Task 10: Add integration test for all endpoints

**Files:**
- Create: `tests/daemon_endpoints.rs`

**Step 1: Write the integration test**

This test starts a daemon on a random port and hits all endpoints:

```rust
// tests/daemon_endpoints.rs
//! Integration test: start daemon, hit all endpoints.
//!
//! This test requires the local machine to have powermetrics access (sudo).
//! Skip in CI with: cargo test --workspace -- --skip daemon_endpoints

use std::time::Duration;

#[tokio::test]
async fn test_all_endpoints_respond() {
    // Start daemon on a random port
    let port = 19090 + (std::process::id() % 1000) as u16;
    let child = tokio::process::Command::new(env!("CARGO_BIN_EXE_asmi"))
        .args(["--serve", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn();

    let mut child = match child {
        Ok(c) => c,
        Err(_) => {
            eprintln!("could not spawn asmi, skipping");
            return;
        }
    };

    // Wait for server to start
    tokio::time::sleep(Duration::from_secs(5)).await;

    let client = reqwest::Client::new();
    let base = format!("http://localhost:{port}");

    // Test all endpoints return 200
    for path in &["/health", "/metrics", "/processes", "/models", "/logs?name=asmi", "/runtime", "/health/setup"] {
        let url = format!("{base}{path}");
        let resp = client.get(&url)
            .timeout(Duration::from_secs(5))
            .send().await;
        match resp {
            Ok(r) => assert!(r.status().is_success(), "{path} returned {}", r.status()),
            Err(e) => panic!("{path} failed: {e}"),
        }
    }

    child.kill().await.ok();
}
```

**Step 2: Run the test**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test --workspace test_all_endpoints_respond -- --nocapture`
Expected: All endpoints return 200

**Step 3: Commit**

```bash
git add tests/daemon_endpoints.rs
git commit -m "test: integration test for all daemon HTTP endpoints"
```

---

## Phase 4: Build, Deploy, Verify

### Task 11: Build, deploy, and verify on cluster

**Step 1: Build release**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo build --release`

**Step 2: Deploy to all nodes**

```bash
ASMI_BIN=target/release/asmi
for h in m3u1 m3u2 m3u3 m4m1; do
  scp "$ASMI_BIN" "$h:~/.cargo/bin/asmi"
  ssh "$h" "codesign --force --sign - ~/.cargo/bin/asmi"
  ssh "$h" "launchctl bootout gui/\$(id -u)/com.asmi.daemon 2>/dev/null; sleep 1; launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/com.asmi.daemon.plist"
done
```

**Step 3: Verify new endpoints on all nodes**

```bash
for h in m3u2.local m3u1.local m3u3.local m4m1.local; do
  echo "=== $h ==="
  curl -s "http://$h:9090/health" | python3 -m json.tool
  curl -s "http://$h:9090/models" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'models: {len(d[\"models\"])}')"
  curl -s "http://$h:9090/runtime" | python3 -m json.tool
  curl -s "http://$h:9090/health/setup" | python3 -c "import json,sys; d=json.load(sys.stdin); print(f'checks: {len(d[\"checks\"])} all_pass: {d[\"all_pass\"]}')"
done
```

**Step 4: Update r1o web app's `asmi-client.ts`**

After verifying the daemon endpoints work, update the web app's shared client to add fetch functions for the new endpoints:
- `fetchAsmiModels(hostname)` → calls `GET /models`
- `fetchAsmiLogs(hostname, name, lines)` → calls `GET /logs`
- `fetchAsmiRuntime(hostname)` → calls `GET /runtime`
- `fetchAsmiSetup(hostname)` → calls `GET /health/setup`

These replace the SSH-based implementations in `cluster-core.ts`.

---

## Summary

| Task | Files | Lines (est.) | Description |
|------|-------|:---:|---|
| 1 | aggregator.rs | +8 | Fix broken mock_snapshot test fixture |
| 2 | daemon.rs (new), main.rs | ~300 moved | Extract HTTP daemon module |
| 3 | daemon_mgmt.rs (new), main.rs | ~200 moved | Extract daemon management module |
| 4 | models.rs (new), daemon.rs, lib.rs | +120 | GET /models — model file discovery |
| 5 | daemon.rs | +50 | GET /logs — log tailing |
| 6 | daemon.rs, main.rs | +80 | GET /runtime — version info |
| 7 | health.rs (new), daemon.rs, lib.rs | +130 | GET /health/setup — setup checks |
| 8 | main.rs, daemon.rs | +10 | --models-dir CLI flag |
| 9 | daemon.rs | +5 | Extend /health response |
| 10 | tests/daemon_endpoints.rs (new) | +40 | Integration test |
| 11 | deploy scripts | — | Build, deploy, verify |

**Execution order:**
```
Task 1 (fix tests) → Task 2-3 (split main.rs, parallel) → Task 4-7 (new endpoints, parallel) → Task 8-9 (polish) → Task 10 (integration test) → Task 11 (deploy)
```

**Net effect:** main.rs shrinks from 2138 → ~1600 lines. 4 new daemon endpoints eliminate 9 SSH calls from the r1o web app per node.
