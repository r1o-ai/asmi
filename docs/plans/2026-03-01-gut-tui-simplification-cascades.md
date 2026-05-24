# asmi Refactor: Gut TUI + Simplification Cascades

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Remove the dead TUI, split the monolithic main.rs, unify duplicated process managers, add typed HTTP errors, and convert all regexes to statics — cleaning asmi for the r1o v1 release.

**Architecture:** 5 sequential cascades, each compiling and passing tests before the next. The TUI removal unlocks the module split, which produces the clean file layout for the remaining 3 independent cascades (serve.rs, daemon.rs, cluster-monitor crate).

**Tech Stack:** Rust (edition 2024, rust-version 1.85), axum 0.8, tokio, regex, std::sync::LazyLock

**Status:** COMPLETED (commit `d115892`)

---

## Pre-Refactor State

| Metric | Value |
|--------|-------|
| `src/main.rs` | 2,011 lines |
| `src/daemon_mgmt.rs` | 194 lines |
| Total binary (`src/`) | 3,763 lines |
| Total all (`src/` + `crates/`) | ~10,000 lines |
| Dependencies | 16 (includes ratatui, crossterm) |
| Duplicated `resolve_python()` | 3 copies |
| Ad-hoc JSON errors | ~16 return points |
| Regex compilations per call | 22 |

## Post-Refactor State

| Metric | Value |
|--------|-------|
| `src/main.rs` | 136 lines |
| Total binary (`src/`) | 2,490 lines |
| Total all (`src/` + `crates/`) | 8,294 lines |
| Dependencies | 14 (-ratatui, -crossterm) |
| Duplicated `resolve_python()` | 0 (1 canonical in asmi-core) |
| Ad-hoc JSON errors | 0 (all use `ApiError` enum) |
| Regex compilations per call | 0 (all `LazyLock<Regex>`) |
| **Net lines deleted** | **~3,046** |

---

## Cascade 1: Gut the TUI (~1,450 lines deleted)

**Files:**
- Modify: `Cargo.toml` — remove `ratatui = "0.30"`, `crossterm = "0.29"`
- Modify: `src/main.rs` — delete TUI code, keep CLI + daemon + formatters

**What was deleted:**
- crossterm imports (lines 7-11)
- ratatui imports (lines 15-22)
- `ActivityLog` struct + `Phase` enum + all methods (lines 157-312)
- `MergeMode` struct (lines 315-321)
- Interactive TUI event loop in `main()` (lines 563-675)
- All `render_*()` functions (~930 lines): `render()`, `render_progress()`, `render_activity()`, `render_header()`, `render_nodes()`, `render_node_summary()`, `render_node_detail()`, `render_process_table()`, `render_footer()`
- `Format::Tui` variant from the enum

**What was simplified:**
- Format detection: removed `if stdout().is_terminal() → Tui`, now defaults to `Table`
- `main()` flow: subcommand → serve → one-shot/watch (no TUI branch)
- `usage_color()` and `gpu_color()` changed from ratatui `Color` return to ANSI escape strings

**What was kept intact:**
- `bin_name()` helper
- `print_table()`, `print_json()` output formatters
- `run_serve()` daemon startup
- One-shot mode (scan → print → exit)
- Streaming watch mode (`--watch --format table`)
- All daemon management subcommands

**Verify:** `cargo build && cargo test` — 0 errors, 12/12 tests pass

---

## Cascade 2: Split main.rs into modules

**Files:**
- Create: `src/cli.rs` — output formatters, one-shot/watch monitor, daemon management
- Create: `src/daemon_startup.rs` — `run_serve()` HTTP daemon startup + background loops
- Rewrite: `src/main.rs` — slim CLI struct + dispatch (~136 lines)
- Delete: `src/daemon_mgmt.rs` — merged into `src/cli.rs`

**Module layout after split:**

| File | Contents | Lines |
|------|----------|-------|
| `src/main.rs` | `Cli` struct, `main()` dispatch (parse → route) | 136 |
| `src/cli.rs` | `run_monitor()`, `run_daemon()`, `print_table()`, `print_json()`, `format_backend()`, `spawn_node_map_updater()` | 513 |
| `src/daemon_startup.rs` | `run_serve()`, `collect_hardware_identity()`, background polling loops | 260 |
| `src/daemon.rs` | HTTP handlers, `AppState`, `build_router()` | 679 |
| `src/serve.rs` | `ProcessManager<R>`, `ServeManager`, `ShareManager` | 902 |

**Key decision:** `daemon_mgmt.rs` was CLI subcommand handling (status/start/stop/deploy/logs), not daemon code. Merging it into `cli.rs` puts all CLI-facing code in one module.

**Verify:** `cargo build && cargo test` — 0 errors, 12/12 tests pass

---

## Cascade 3: Extract `ProcessManager<R: ReadinessCheck>`

**Files:**
- Modify: `src/serve.rs` — unify `ServeManager` + `ShareManager`

**Before:** Two near-identical managers (80% copy-paste):
- `ServeManager` (lines 1-523) — HTTP health polling readiness
- `ShareManager` (lines 525-840) — log file monitoring readiness
- Duplicated: `ManagedServer`/`ManagedShare`, `kill_child`/`kill_share_child`, `persist_state`/`persist_share_state`

**After:** Single generic `ProcessManager<R>` with trait-based readiness:

```rust
trait ReadinessCheck: Send + Sync + 'static {
    async fn poll_ready(&self, child: &mut Child, timeout_secs: u64) -> Result<bool, String>;
}

struct HttpHealth { port: u16, endpoints: Vec<&'static str> }
struct LogMonitor { log_path: String, ready_markers: Vec<&'static str>, error_markers: Vec<&'static str> }

pub type ServeManager = ProcessManager<HttpHealth>;
pub type ShareManager = ProcessManager<LogMonitor>;
```

**Unified:**
- `ManagedServer` + `ManagedShare` → `ManagedProcess` (with `port: Option<u16>`)
- `kill_child()` + `kill_share_child()` → single `kill_child()`
- `persist_state()` + `persist_share_state()` → single `persist_state()` (branches on `port`)
- `stop()` shared in `ProcessManager<R>`

**Public API preserved exactly** — type aliases ensure `daemon.rs` and `daemon_startup.rs` see the same `ServeManager`/`ShareManager` types.

**Verify:** `cargo build && cargo test` — 12/12 tests pass

---

## Cascade 4: Typed HTTP responses with `ApiError`

**Files:**
- Modify: `src/daemon.rs` — add `ApiError` enum, convert handlers

**Added:**
```rust
enum ApiError {
    BadRequest(String),   // 400
    NotFound(String),     // 404
    Internal(String),     // 500
}
impl axum::response::IntoResponse for ApiError { ... }
```

**Converted 14 handlers** from `-> Json<serde_json::Value>` to `-> Result<Json<...>, ApiError>`:

| Handler | Error categories |
|---------|-----------------|
| `metrics_handler` | NotFound, Internal |
| `cluster_handler` | BadRequest, Internal |
| `jaccl_config_handler` | NotFound, Internal |
| `jaccl_generate_handler` | NotFound, Internal |
| `logs_handler` | BadRequest, Internal |
| `serve_status_handler` | NotFound, Internal |
| `serve_load_handler` | BadRequest, NotFound |
| `serve_stop_handler` | NotFound |
| `serve_reload_handler` | NotFound |
| `serve_share_handler` | BadRequest |
| `runtime_handler` | Internal |
| `setup_handler` | Internal |
| `network_health_handler` | Internal |
| `network_fix_handler` | Internal |

**Left 6+ handlers unchanged** (always succeed): `health_handler`, `processes_handler`, `nodes_handler`, `models_handler`, `volumes_handler`, `thunderbolt_handler`, `arp_handler`, `stream_handler`, `serve_share_stop_handler`

**Web layer impact:** Safe — `asmi-client.ts` already checks `!res.ok`. Same `{"error": "msg"}` JSON body format preserved for backward compat.

**Verify:** `cargo build && cargo test` — 12/12 tests pass

---

## Cascade 5: Static regexes + `resolve_python()` dedup

**Files:**
- Modify: `crates/cluster-monitor/src/lib.rs` — add canonical `resolve_python()`
- Modify: `crates/cluster-monitor/src/health.rs` — use `crate::resolve_python`
- Modify: `src/daemon.rs` — `pub use asmi_core::resolve_python;` (re-export for serve.rs)
- Modify: `crates/cluster-monitor/src/collector.rs` — 8 static regexes
- Modify: `crates/cluster-monitor/src/scanner.rs` — 12 static regexes (14 calls, 2 shared)

**Part A — `resolve_python()` dedup:**

| Location | Before | After |
|----------|--------|-------|
| `src/daemon.rs:35` | Full function definition | `pub use asmi_core::resolve_python;` |
| `crates/.../health.rs:7` | Full function definition | `use crate::resolve_python;` |
| `crates/.../lib.rs` | Not present | Canonical definition + re-export |

**Part B — Static regexes (22 → 20 statics, 2 deduped):**

`collector.rs` — 8 statics:
- `POWER_RE`, `GPU_ACTIVE_RE`, `GPU_IDLE_RE`, `CPU_ACTIVE_RE`
- `PAGE_SIZE_RE`, `PAGE_RE`
- `PS_RE`, `FOOTPRINT_RE`

`scanner.rs` — 12 statics (from 14 calls, 2 shared):
- `IFACE_RE` — shared by `parse_ifconfig_bridges()` and `parse_ifconfig_all_ips()`
- `INET_LINK_LOCAL_RE`, `INET_RE`
- `ARP_RE` — shared by `parse_arp_table()` and `discover_thunderbolt_bridge()`
- `DEVICE_RE`, `STATE_RE`
- `SP_DEVICE_RE`, `SP_SPEED_RE`, `SP_STATUS_RE`
- `MLX_PORT_RE`, `MLX_MODEL_RE`
- `PING_LATENCY_RE`

Pattern used:
```rust
use std::sync::LazyLock;
static POWER_RE: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"(?m)^(CPU|GPU|ANE) Power:\s+([\d.]+)\s+mW").unwrap()
});
```

**Verify:** `cargo build && cargo test` — 70/70 asmi-core tests pass, 12/12 serve_share tests pass

---

## Follow-Up: MetricsHistory Per-Model Enrichment

**Status:** NOT STARTED — planned as separate enhancement

**Goal:** Record per-model CPU/GPU utilization in history, so the web topology page can show inference load per model over time.

**Current state** (`crates/cluster-monitor/src/types.rs:429`):
```rust
pub struct MetricsHistory {
    capacity: usize,
    pub cpu: VecDeque<f64>,
    pub gpu: VecDeque<f64>,
    pub memory: VecDeque<f64>,
    pub power: VecDeque<f64>,
}
```

Only records node-level aggregates. No per-process breakdown.

**Proposed change:**

### Task 1: Add ProcessMetricsHistory type

**Files:**
- Modify: `crates/cluster-monitor/src/types.rs`

```rust
/// Per-process metrics history (keyed by model name in MetricsHistory).
#[derive(Debug, Clone)]
pub struct ProcessMetricsHistory {
    pub cpu: VecDeque<f64>,
    pub gpu_footprint_mb: VecDeque<f64>,
    capacity: usize,
}

impl ProcessMetricsHistory {
    pub fn new(capacity: usize) -> Self {
        Self {
            cpu: VecDeque::with_capacity(capacity),
            gpu_footprint_mb: VecDeque::with_capacity(capacity),
            capacity,
        }
    }

    pub fn push(&mut self, cpu: f64, gpu_footprint_mb: f64) {
        if self.cpu.len() >= self.capacity {
            self.cpu.pop_front();
            self.gpu_footprint_mb.pop_front();
        }
        self.cpu.push_back(cpu);
        self.gpu_footprint_mb.push_back(gpu_footprint_mb);
    }
}
```

Add to `MetricsHistory`:
```rust
pub struct MetricsHistory {
    // ... existing fields ...
    /// Per-model metrics history, keyed by model name (last path component).
    pub models: HashMap<String, ProcessMetricsHistory>,
}
```

### Task 2: Push per-model data in aggregator

**Files:**
- Modify: `crates/cluster-monitor/src/aggregator.rs:46-62`

In `update_node()`, after pushing node-level history, iterate `snapshot.processes` and push per-model data:

```rust
for proc in &snapshot.processes {
    let model_key = proc.server_models.first()
        .map(|m| m.id.rsplit('/').next().unwrap_or(&m.id).to_string())
        .or_else(|| proc.model.clone())
        .unwrap_or_else(|| format!("pid-{}", proc.pid));
    let proc_hist = history.models
        .entry(model_key)
        .or_insert_with(|| ProcessMetricsHistory::new(self.history_capacity));
    proc_hist.push(proc.cpu_percent, proc.footprint_mb.unwrap_or(0.0));
}
```

### Task 3: Expose via /metrics endpoint

**Files:**
- No changes needed — `MetricsHistory` is already serialized by `NodeSnapshot` via the daemon. The web layer reads `processes[]` from `/metrics` which already has `cpu_percent` and `footprint_mb`.

The history enrichment is for the **ClusterState** (hub mode `/cluster` endpoint and web SSE stream), not the per-node `/metrics`. The web topology page would read from `ClusterState.histories[hostname].models` to render per-model sparklines.

### Task 4: Clean up stale model entries

Models that stop running should have their history entries cleaned after N polls with no data. Add to `update_node()`:

```rust
// Remove history for models no longer running
let active_models: HashSet<String> = /* collect from snapshot.processes */;
history.models.retain(|k, _| active_models.contains(k));
```

**Test plan:**
- Unit test: push model data, verify history accumulates
- Unit test: model stops running, verify entry is cleaned after retention
- Integration: web app renders per-model CPU sparklines in NodeDetailPanel

---

## Verification Checklist (All Completed)

- [x] `cargo build` — compiles clean
- [x] `cargo test` — 82/82 tests pass (70 asmi-core + 12 serve_share)
- [x] `cargo build --release` — release binary builds
- [x] `src/daemon_mgmt.rs` deleted (merged into `cli.rs`)
- [x] ratatui + crossterm removed from `Cargo.toml`
- [x] No `Format::Tui` variant anywhere
- [x] Zero duplicated `resolve_python()` definitions
- [x] Zero `Regex::new()` calls in hot paths
- [x] All error returns use proper HTTP status codes
