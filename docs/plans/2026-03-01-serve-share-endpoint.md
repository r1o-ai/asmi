# Asmi: /serve/share Endpoint for Distributed Model Sharing

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Add `POST /serve/share` and `GET /serve/share/status` endpoints to asmi so the web UI (and MCP tools) can start/stop `mlx_lm.share` distributed model sharing sessions — the same way `/serve/load` manages `mlx_lm.server`.

**Architecture:** `mlx_lm.share` is fundamentally different from `mlx_lm.server`:
- **share** doesn't start an HTTP server — it's a headless process that joins a JACCL ring and exposes model shards via RDMA
- It has no health endpoint to poll — readiness is determined by log output ("Starting endpoint" or "Connected to coordinator")
- It's always distributed (requires `--backend jaccl` + `--hostfile`)
- Only one share session runs at a time (not per-port)

**Tech Stack:** Rust, Axum, Tokio async, `tokio::process::Child`

**Reference command:**
```bash
# Single-node share (coordinator runs this)
python -m mlx_lm.share --model mlx-community/Kimi-K2.5-8bit

# Distributed share (JACCL wraps it)
python -m mlx.launch --hostfile ~/hostfile.json --backend jaccl -- \
  python -m mlx_lm.share --model mlx-community/Kimi-K2.5-8bit
```

---

## Tasks

### Task 1: Add `MlxLmShare` variant to `ServeEngine`

**File:** `crates/cluster-monitor/src/types.rs`

Add `MlxLmShare` to the `ServeEngine` enum (alongside `MlxLm`, `MlxVlm`, `VllmMlx`):
```rust
#[serde(rename = "mlx_lm_share")]
MlxLmShare,
```

Add to `Display` impl:
```rust
Self::MlxLmShare => write!(f, "mlx_lm_share"),
```

Add `EngineConfig` for share:
```rust
Self::MlxLmShare => EngineConfig {
    binary: "mlx_lm.share",
    binary_args: &[],
    uvicorn_app: None,
    model_flag: Some("--model"),
    // share has no HTTP server — health checked via log output
    health_endpoints: &[],
},
```

**Tests:** Existing serialization tests should still pass. No new tests needed for this task.

---

### Task 2: Add `ShareManager` to `serve.rs`

**File:** `src/serve.rs`

Create a new `ShareManager` struct (separate from `ServeManager` since share has different lifecycle):

```rust
/// Internal state for the share session.
struct ManagedShare {
    state: ServeState,       // Idle | Loading | Ready | Error
    model: Option<String>,
    backend: ServeBackend,   // Always Jaccl for share
    child: Option<tokio::process::Child>,
    pid: Option<u32>,
    load_started: Option<std::time::Instant>,
    error: Option<String>,
}

#[derive(Clone)]
pub struct ShareManager {
    inner: Arc<RwLock<ManagedShare>>,
}
```

Methods:
- `new() -> Self` — starts Idle
- `start(req: ShareRequest) -> ()` — spawns background task, transitions Loading → Ready/Error
- `stop() -> ()` — kills child, transitions to Idle
- `status() -> ShareStatus` — read-only snapshot

**ShareRequest** (add to `types.rs`):
```rust
pub struct ShareRequest {
    pub model_path: String,       // required (no bare share)
    pub backend: String,          // "auto" | "jaccl" (defaults to auto)
    pub hostfile: Option<String>, // JACCL hostfile path
}
```

**ShareStatus** (add to `types.rs`):
```rust
pub struct ShareStatus {
    pub state: ServeState,
    pub model: Option<String>,
    pub backend: ServeBackend,
    pub pid: Option<u32>,
    pub elapsed_ms: u64,
    pub error: Option<String>,
}
```

**Key difference from ServeManager:** Instead of polling HTTP health endpoints, the background `do_share()` task should:
1. Spawn the process with stdout/stderr → `/tmp/r1o-mlx-share.log`
2. Read the log file in a loop (every 500ms) looking for readiness markers:
   - `"Starting endpoint"` → Ready
   - `"Connected to"` → Ready
   - `"Error"` or `"Exception"` → Error (with log tail)
3. Race against `child.wait()` for early crash detection (same pattern as `do_load_inner`)
4. 120s timeout (share startup is slower than server — needs to sync shards)

**Crash recovery:** Persist to `~/.r1o/share-state.json`. On daemon restart, restore like `ServeManager::restore()`.

---

### Task 3: Wire `ShareManager` into `AppState`

**File:** `src/daemon.rs`, `src/main.rs`

Add to `AppState`:
```rust
pub share_manager: crate::serve::ShareManager,
```

Initialize in `main.rs` alongside the serve managers:
```rust
let share_manager = crate::serve::ShareManager::restore().await;
```

---

### Task 4: Add HTTP endpoints

**File:** `src/daemon.rs`

Add two new endpoints:

#### `POST /serve/share` — start a share session
```rust
async fn serve_share_handler(
    State(state): State<AppState>,
    Json(req): Json<ShareRequest>,
) -> Json<serde_json::Value>
```
- Validates `model_path` is non-empty
- Calls `state.share_manager.start(req).await`
- Returns `{"ok": true, "state": "loading"}`

#### `GET /serve/share/status` — share session status
```rust
async fn serve_share_status_handler(
    State(state): State<AppState>,
) -> Json<serde_json::Value>
```
- Returns `ShareStatus` JSON

#### `POST /serve/share/stop` — stop share session
```rust
async fn serve_share_stop_handler(
    State(state): State<AppState>,
) -> Json<serde_json::Value>
```
- Calls `state.share_manager.stop().await`

Register in `build_router()`:
```rust
.route("/serve/share", post(serve_share_handler))
.route("/serve/share/status", get(serve_share_status_handler))
.route("/serve/share/stop", post(serve_share_stop_handler))
```

---

### Task 5: Update startup banner

**File:** `src/main.rs`

Add share endpoints to the daemon startup printout:
```
  POST /serve/share       Start distributed share session
  GET  /serve/share/status Share session status
  POST /serve/share/stop  Stop share session
```

---

### Task 6: Integration test

**File:** `tests/serve_share.rs`

Write a test that:
1. Creates a `ShareManager::new()`
2. Verifies initial status is `Idle`
3. Starts a share with a non-existent model path
4. Waits for error state (process should fail fast)
5. Verifies error message contains useful info
6. Stops → verifies back to Idle

---

## Design Decisions

1. **Separate ShareManager vs extending ServeManager:** Share has no port, no HTTP health check, and only one instance. Extending ServeManager would require many conditional branches. A dedicated type is cleaner.

2. **Log-based readiness instead of health polling:** `mlx_lm.share` doesn't expose an HTTP API. We scrape the log file for readiness markers — same approach the MLX team uses in their test harness.

3. **Single share session:** Unlike serve (which manages 2 ports), share runs one session. The coordinator orchestrates which nodes participate via the hostfile.

4. **120s timeout:** Share startup involves RDMA handshake + model shard distribution across nodes. 60s (the server timeout) is too tight for large models like Kimi-K2.5 (612GB).

---

## Out of Scope (future work)

- **Multi-node orchestration:** This plan only wires the local `mlx_lm.share` command. Orchestrating share across all nodes (POST to each node's `/serve/share`) is a web-layer concern.
- **Live log streaming:** SSE streaming of share logs (useful for debugging RDMA handshake issues). Could be added as `GET /serve/share/logs` later.
- **Integration with `/serve/status`:** Could add share status to the combined status response — deferred for simplicity.
