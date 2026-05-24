# Remove Hardcoded Paths & Enrich /v1/models Probing

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Make asmi a proper independent package by removing all hardcoded cluster hostnames, file paths, and dev machine paths, while enriching the `/v1/models` probe to include `context_length` and `max_tokens` metadata.

**Architecture:** Two independent workstreams: (1) Replace the `DAEMON_NODES` constant and hardcoded paths in `src/main.rs` with dynamic lookups from the existing `NodeMap` and platform-standard directories. (2) Enrich `ProcessInfo` in `asmi-core` with model server metadata from `/v1/models` probing, and extend the existing `parse_v1_models_response` to extract full model metadata. Both workstreams touch different files and can be committed independently.

**Tech Stack:** Rust 1.85, asmi-core (cluster-monitor crate), serde, tokio, dirs, clap

---

## Phase 1: Remove Hardcoded Paths from Binary

### Task 1: Replace hardcoded log paths with platform-standard directories

**Files:**
- Modify: `src/main.rs:339` (log file creation)
- Modify: `src/main.rs:1648` (daemon log tail)

**Step 1: Write the failing test**

No unit test needed — this is a path construction change. We verify by compilation + manual run.

**Step 2: Replace `/tmp/asmi.log` with `dirs::data_local_dir()`**

In `src/main.rs`, replace line 339:

```rust
// OLD:
let log_file = std::fs::File::create("/tmp/asmi.log")?;

// NEW:
let log_dir = dirs::data_local_dir()
    .unwrap_or_else(|| std::path::PathBuf::from("/tmp"))
    .join("asmi");
std::fs::create_dir_all(&log_dir)?;
let log_file = std::fs::File::create(log_dir.join("asmi.log"))?;
```

On macOS this resolves to `~/Library/Application Support/asmi/asmi.log`.

**Step 3: Replace `/tmp/asmi-daemon.log` in the daemon logs command**

In `src/main.rs`, around line 1648, replace the hardcoded path:

```rust
// OLD:
let cmd = "tail -50 /tmp/asmi-daemon.log";

// NEW:
// The daemon writes to ~/Library/Application Support/asmi/asmi.log on macOS.
// On remote nodes, use the same XDG-style path the daemon itself uses.
let cmd = format!(
    "tail -50 \"$(echo ~/Library/Application\\ Support/asmi/asmi.log 2>/dev/null || echo /tmp/asmi.log)\""
);
```

Actually, since the daemon itself will also be updated to use the new path, and all nodes run macOS, simplify:

```rust
let log_path = "~/Library/Application\\ Support/asmi/asmi.log";
let cmd = format!("tail -50 {}", log_path);
```

**Step 4: Verify it compiles**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check`
Expected: compiles with no errors

**Step 5: Commit**

```bash
git add src/main.rs
git commit -m "fix: use platform-standard log directory instead of /tmp"
```

---

### Task 2: Remove `DAEMON_NODES` constant — use `NodeMap` for daemon management

This is the critical change. The daemon subcommand (`asmi daemon status/start/stop/restart/deploy/logs`) currently iterates over a hardcoded `DAEMON_NODES` array. It should use `NodeMap.nodes` instead.

**Files:**
- Modify: `src/main.rs:1517-1522` (remove `DAEMON_NODES` const)
- Modify: `src/main.rs:1525-1660` (daemon subcommand handler)

**Step 1: Delete the `DAEMON_NODES` constant (lines 1517-1522)**

Remove:
```rust
const DAEMON_NODES: &[(&str, &str)] = &[
    ("m3u2", "m3u2.local"),
    ("m3u1", "m3u1.local"),
    ("m3u3", "m3u3.local"),
    ("m4m1", "m4m1.local"),
];
```

**Step 2: Load `NodeMap` at the start of the daemon handler**

At the top of the daemon match arm (around line 1525), load the node list:

```rust
let node_map = asmi_core::NodeMap::load();
let known_nodes: Vec<String> = if node_map.nodes.is_empty() {
    eprintln!("No known nodes in NodeMap. Run `asmi` first to discover cluster nodes,");
    eprintln!("or add seed hosts: `asmi --hosts m3u1,m3u2,m3u3`");
    std::process::exit(1);
} else {
    node_map.nodes.clone()
};
```

**Step 3: Replace all `DAEMON_NODES` iterations**

Every occurrence of `for &(name, _) in DAEMON_NODES` becomes:

```rust
for name in &known_nodes {
    let name = name.as_str();
```

The second tuple element (`addr`) was only used for display — the actual SSH always used `name` directly. So no functionality is lost.

**Step 4: Replace the `target == "m3u2"` hardcoded local check (line 1649)**

```rust
// OLD:
if target == local_hostname || target == "m3u2" {

// NEW:
if target == local_hostname {
```

The `"m3u2"` check was a workaround for hostname mismatch. With `NodeMap` aliases, this is no longer needed.

**Step 5: Verify it compiles**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check`
Expected: compiles with no errors

**Step 6: Commit**

```bash
git add src/main.rs
git commit -m "fix: replace hardcoded DAEMON_NODES with dynamic NodeMap lookup"
```

---

### Task 3: Remove hardcoded dev machine binary path

**Files:**
- Modify: `src/main.rs:1610-1617` (binary path resolution for deploy)

**Step 1: Replace the fallback path**

```rust
// OLD (lines 1610-1617):
// Falls back to ~/Projects/Personal/apple-smi/target/release/asmi

// NEW: Use current_exe() as primary, which asmi as final fallback
let bin = std::env::current_exe()
    .ok()
    .filter(|p| p.exists())
    .or_else(|| {
        // Check if a release build exists adjacent to the workspace
        let workspace = std::path::Path::new(env!("CARGO_MANIFEST_DIR"));
        let release = workspace.join("target/release/asmi");
        if release.exists() { Some(release) } else { None }
    })
    .or_else(|| {
        // which asmi — find it on PATH
        std::process::Command::new("which")
            .arg("asmi")
            .output()
            .ok()
            .filter(|o| o.status.success())
            .and_then(|o| {
                let path = String::from_utf8_lossy(&o.stdout).trim().to_string();
                let p = std::path::PathBuf::from(&path);
                if p.exists() { Some(p) } else { None }
            })
    })
    .unwrap_or_else(|| {
        dirs::home_dir()
            .unwrap_or_default()
            .join(".cargo/bin/asmi")
    });
```

This resolution chain: `current_exe()` → `target/release/asmi` → `which asmi` → `~/.cargo/bin/asmi`.

**Step 2: Verify it compiles**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check`
Expected: compiles with no errors

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "fix: resolve deploy binary via current_exe/which instead of hardcoded path"
```

---

### Task 4: Make `DAEMON_PLIST` name generic

**Files:**
- Modify: `src/main.rs:1523` (DAEMON_PLIST constant)

**Step 1: Remove the r1o-specific plist name**

```rust
// OLD:
const DAEMON_PLIST: &str = "com.r1o.asmi-daemon";

// NEW:
const DAEMON_PLIST: &str = "com.asmi.daemon";
```

**Step 2: Verify it compiles**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check`
Expected: compiles with no errors

**Step 3: Commit**

```bash
git add src/main.rs
git commit -m "fix: rename daemon plist from r1o-specific to generic asmi name"
```

---

### Task 5: Make `NodeMap::config_path()` respect XDG_CONFIG_HOME

**Files:**
- Modify: `crates/cluster-monitor/src/config.rs:150-157`

**Step 1: Write the failing test**

```rust
#[test]
fn test_config_path_ends_with_asmi() {
    let path = NodeMap::config_path();
    assert!(path.ends_with("asmi/config.json"),
        "config path should end with asmi/config.json, got: {}", path.display());
}
```

Add this in the existing `#[cfg(test)] mod tests` block in `config.rs` (after line 465).

**Step 2: Run test to verify it passes (it should already pass)**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test -p asmi-core test_config_path_ends_with_asmi -- --nocapture`
Expected: PASS (current implementation already ends with `asmi/config.json`)

**Step 3: Update `config_path()` to respect XDG_CONFIG_HOME**

```rust
/// Path to the persistent config file.
/// Respects `XDG_CONFIG_HOME` if set, otherwise `~/.config/asmi/config.json`.
pub fn config_path() -> PathBuf {
    if let Ok(xdg) = std::env::var("XDG_CONFIG_HOME") {
        let p = PathBuf::from(xdg);
        if p.is_absolute() {
            return p.join("asmi").join("config.json");
        }
    }
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("/tmp"))
        .join(".config")
        .join("asmi")
        .join("config.json")
}
```

**Step 4: Run all config tests**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test -p asmi-core -- config`
Expected: all config tests pass

**Step 5: Commit**

```bash
git add crates/cluster-monitor/src/config.rs
git commit -m "feat: respect XDG_CONFIG_HOME for config file location"
```

---

## Phase 2: Enrich `/v1/models` Probing with Full Metadata

### Task 6: Add `ModelServerMetadata` struct to types.rs

**Files:**
- Modify: `crates/cluster-monitor/src/types.rs` (add new struct after `ProcessInfo`)

**Step 1: Add the new types**

After the `ProcessInfo` struct (around line 80), add:

```rust
/// Metadata about a model served by an MLX/vllm endpoint, as reported by `/v1/models`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelServerMetadata {
    /// Full model ID from the server (e.g. "mlx-community/Qwen3-32B-4bit")
    pub id: String,
    /// Context window size in tokens, if reported by the server.
    pub context_length: Option<u64>,
    /// Maximum output tokens, if reported by the server.
    pub max_tokens: Option<u64>,
}
```

**Step 2: Add `server_models` field to `ProcessInfo`**

Extend `ProcessInfo` (lines 68-80) with a new field:

```rust
pub struct ProcessInfo {
    pub pid: u32,
    pub framework: ProcessFramework,
    pub model: Option<String>,
    pub port: Option<u16>,
    pub cpu_percent: f64,
    pub mem_percent: f64,
    pub footprint_mb: Option<f64>,
    pub distributed: Option<DistributedBackend>,
    /// Model metadata from probing the server's `/v1/models` endpoint.
    /// Empty if the server is not reachable or has no models loaded.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub server_models: Vec<ModelServerMetadata>,
}
```

**Step 3: Add `Default` for `server_models` in all `ProcessInfo` construction sites**

Every place that constructs a `ProcessInfo` needs `server_models: Vec::new()`. There are two sites:
- `collector.rs` in `parse_ps_mlx` (around line 512)
- `scanner.rs` in `scan_node_inner` (around line 250)

In `collector.rs`, the `ProcessInfo` construction (around line 512):
```rust
procs.push(ProcessInfo {
    pid,
    framework,
    model,
    port,
    cpu_percent,
    mem_percent,
    footprint_mb: None,
    distributed,
    server_models: Vec::new(), // populated later by endpoint probing
});
```

In `scanner.rs`, the same pattern applies wherever `ProcessInfo` is built.

**Step 4: Verify it compiles**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check`
Expected: compiles with no errors

**Step 5: Run all tests**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test`
Expected: all 51 tests pass

**Step 6: Commit**

```bash
git add crates/cluster-monitor/src/types.rs crates/cluster-monitor/src/collector.rs crates/cluster-monitor/src/scanner.rs
git commit -m "feat: add ModelServerMetadata struct for /v1/models enrichment"
```

---

### Task 7: Extend `parse_v1_models_response` to extract full metadata

**Files:**
- Modify: `crates/cluster-monitor/src/scanner.rs:969-985` (existing parser)
- Modify: `crates/cluster-monitor/testdata/` (add test fixture)

**Step 1: Create a realistic test fixture**

Create file: `crates/cluster-monitor/testdata/v1-models-response.json`

```json
{
  "object": "list",
  "data": [
    {
      "id": "mlx-community/Qwen3-32B-4bit",
      "object": "model",
      "created": 1706745938,
      "owned_by": "system",
      "context_length": 131072,
      "max_tokens": 16384
    }
  ]
}
```

**Step 2: Write the failing test**

Add to the `#[cfg(test)] mod tests` block in `scanner.rs`:

```rust
#[test]
fn test_parse_v1_models_metadata() {
    let json = std::fs::read_to_string("testdata/v1-models-response.json").unwrap();
    let models = parse_v1_models_metadata(&json);
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "mlx-community/Qwen3-32B-4bit");
    assert_eq!(models[0].context_length, Some(131072));
    assert_eq!(models[0].max_tokens, Some(16384));
}

#[test]
fn test_parse_v1_models_metadata_no_context() {
    // Some servers don't report context_length
    let json = r#"{"data":[{"id":"test-model","object":"model"}]}"#;
    let models = parse_v1_models_metadata(json);
    assert_eq!(models.len(), 1);
    assert_eq!(models[0].id, "test-model");
    assert_eq!(models[0].context_length, None);
    assert_eq!(models[0].max_tokens, None);
}

#[test]
fn test_parse_v1_models_metadata_empty() {
    let models = parse_v1_models_metadata("{}");
    assert!(models.is_empty());
}

#[test]
fn test_parse_v1_models_metadata_invalid_json() {
    let models = parse_v1_models_metadata("not json");
    assert!(models.is_empty());
}
```

**Step 3: Run tests to verify they fail**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test -p asmi-core test_parse_v1_models_metadata`
Expected: FAIL — `parse_v1_models_metadata` not found

**Step 4: Implement `parse_v1_models_metadata`**

Add new function near the existing `parse_v1_models_response` (around line 985):

```rust
/// Parse a `/v1/models` JSON response into full model metadata.
///
/// Extracts `id`, `context_length`, and `max_tokens` from each model entry.
/// Returns an empty vec on invalid JSON or missing `data` array.
pub(crate) fn parse_v1_models_metadata(json: &str) -> Vec<ModelServerMetadata> {
    let value: serde_json::Value = match serde_json::from_str(json) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };

    value
        .get("data")
        .and_then(|d| d.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| {
                    let id = item.get("id")?.as_str()?.to_string();
                    Some(ModelServerMetadata {
                        id,
                        context_length: item
                            .get("context_length")
                            .and_then(|v| v.as_u64()),
                        max_tokens: item
                            .get("max_tokens")
                            .and_then(|v| v.as_u64()),
                    })
                })
                .collect()
        })
        .unwrap_or_default()
}
```

Add the import at the top of `scanner.rs`:
```rust
use crate::types::ModelServerMetadata;
```

**Step 5: Run tests to verify they pass**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test -p asmi-core test_parse_v1_models_metadata`
Expected: all 4 tests pass

**Step 6: Commit**

```bash
git add crates/cluster-monitor/src/scanner.rs crates/cluster-monitor/testdata/v1-models-response.json
git commit -m "feat: parse context_length and max_tokens from /v1/models response"
```

---

### Task 8: Wire `/v1/models` metadata into scanner's `scan_node_inner`

**Files:**
- Modify: `crates/cluster-monitor/src/scanner.rs:228-243` (existing /v1/models probe)

**Step 1: Update the existing probe to use the new parser**

The scanner already probes `/v1/models` per process (lines 228-243). Update it to use `parse_v1_models_metadata` instead of `parse_v1_models_response`, and attach the results to `ProcessInfo`:

```rust
// Replace existing probe block (lines 228-243):
let curl_cmd = format!(
    "curl -s --connect-timeout 2 http://127.0.0.1:{}/v1/models 2>/dev/null",
    port
);
let server_models = match ssh_run(hostname, &curl_cmd, config).await {
    Ok(ref r) if r.has_output() => parse_v1_models_metadata(&r.stdout),
    _ => Vec::new(),
};

// Use server-reported model ID if ps didn't capture one
let model_from_server = server_models.first().map(|m| m.id.clone());
```

Then when constructing `ProcessInfo`, set:
```rust
server_models,
```

And use `model_from_server` as a fallback for the `model` field if `extract_flag_value` didn't find one.

**Step 2: Keep backward compat — old `parse_v1_models_response` still needed for `MlxServerInfo`**

The existing `parse_v1_models_response` is used to build `MlxServerInfo.models` (a `Vec<String>`). Keep it, but implement it as a thin wrapper:

```rust
fn parse_v1_models_response(json: &str) -> Vec<String> {
    parse_v1_models_metadata(json)
        .into_iter()
        .map(|m| m.id)
        .collect()
}
```

**Step 3: Verify it compiles**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check`
Expected: compiles with no errors

**Step 4: Run all tests**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test`
Expected: all tests pass (existing `parse_v1_models_response` tests still work via wrapper)

**Step 5: Commit**

```bash
git add crates/cluster-monitor/src/scanner.rs
git commit -m "feat: attach model server metadata to ProcessInfo during node scan"
```

---

### Task 9: Add `/v1/models` probing to collector.rs

The collector runs every 2s for metrics but currently doesn't probe HTTP endpoints. Add an optional probe step that enriches `ProcessInfo` for processes with known ports.

**Files:**
- Modify: `crates/cluster-monitor/src/collector.rs` (add probe after process parsing)

**Step 1: Add a `probe_model_endpoints` function**

This is an async function that takes a hostname, config, and list of `ProcessInfo`, and enriches each one that has a port:

```rust
use crate::types::ModelServerMetadata;

/// Probe `/v1/models` on each process that has a port, enriching with server metadata.
/// This is fire-and-forget: failures leave `server_models` empty (not an error).
pub(crate) async fn probe_model_endpoints(
    hostname: &str,
    config: &crate::config::ClusterConfig,
    processes: &mut [ProcessInfo],
) {
    use futures::future::join_all;

    let probes: Vec<_> = processes
        .iter()
        .enumerate()
        .filter_map(|(i, p)| p.port.map(|port| (i, port)))
        .map(|(i, port)| {
            let hostname = hostname.to_string();
            let config = config.clone();
            async move {
                let curl_cmd = format!(
                    "curl -s --connect-timeout 2 http://127.0.0.1:{}/v1/models 2>/dev/null",
                    port
                );
                let models = match crate::ssh::ssh_run(&hostname, &curl_cmd, &config).await {
                    Ok(ref r) if r.has_output() => {
                        crate::scanner::parse_v1_models_metadata(&r.stdout)
                    }
                    _ => Vec::new(),
                };
                (i, models)
            }
        })
        .collect();

    for (i, models) in join_all(probes).await {
        processes[i].server_models = models;
    }
}
```

**Important:** This requires making `parse_v1_models_metadata` in `scanner.rs` `pub(crate)` (it already is from Task 7).

**Step 2: Wire into `collect_node_metrics`**

In the main `collect_node_metrics` function (around line 244-268), after assembling `processes` and before returning the `NodeSnapshot`, add:

```rust
// Enrich processes with /v1/models metadata (non-blocking, best-effort)
probe_model_endpoints(hostname, config, &mut processes).await;
```

**Step 3: Verify it compiles**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check`
Expected: compiles with no errors

**Step 4: Run all tests**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test`
Expected: all tests pass

**Step 5: Commit**

```bash
git add crates/cluster-monitor/src/collector.rs
git commit -m "feat: probe /v1/models during metrics collection for model enrichment"
```

---

### Task 10: Add `models_loaded` enrichment to aggregates

**Files:**
- Modify: `crates/cluster-monitor/src/aggregator.rs` (update ClusterAggregates)
- Modify: `crates/cluster-monitor/src/types.rs` (if ClusterAggregates is there)

**Step 1: Find where `ClusterAggregates.models_loaded` is computed**

Check `aggregator.rs` — the `models_loaded` field in aggregates should include the full model IDs from `server_models` (not just the `--model` flag parse). Update the aggregation logic:

```rust
// When building models_loaded, prefer server-reported model IDs:
let models_loaded: Vec<String> = snapshots
    .iter()
    .filter(|s| s.online)
    .flat_map(|s| s.processes.iter())
    .flat_map(|p| {
        // Prefer server-reported IDs, fall back to ps-parsed model name
        if !p.server_models.is_empty() {
            p.server_models.iter().map(|m| m.id.clone()).collect::<Vec<_>>()
        } else if let Some(ref model) = p.model {
            vec![model.clone()]
        } else {
            vec![]
        }
    })
    .collect::<std::collections::HashSet<_>>()
    .into_iter()
    .collect();
```

**Step 2: Run all tests**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test`
Expected: all tests pass

**Step 3: Commit**

```bash
git add crates/cluster-monitor/src/aggregator.rs
git commit -m "feat: prefer server-reported model IDs in cluster aggregates"
```

---

### Task 11: Re-export `parse_v1_models_metadata` from lib.rs

**Files:**
- Modify: `crates/cluster-monitor/src/lib.rs`
- Modify: `crates/cluster-monitor/src/scanner.rs` (make function pub)

**Step 1: Make the function public**

In `scanner.rs`, change:
```rust
pub(crate) fn parse_v1_models_metadata(json: &str) -> Vec<ModelServerMetadata> {
```
to:
```rust
pub fn parse_v1_models_metadata(json: &str) -> Vec<ModelServerMetadata> {
```

**Step 2: Re-export from lib.rs**

Add to `crates/cluster-monitor/src/lib.rs`:
```rust
pub use scanner::parse_v1_models_metadata;
pub use types::ModelServerMetadata;
```

This allows consumers (r1o-tui, web app) to call `asmi_core::parse_v1_models_metadata()` directly if they need to parse a `/v1/models` response themselves.

**Step 3: Verify it compiles**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check`
Expected: compiles with no errors

**Step 4: Commit**

```bash
git add crates/cluster-monitor/src/lib.rs crates/cluster-monitor/src/scanner.rs
git commit -m "feat: export parse_v1_models_metadata and ModelServerMetadata from asmi-core"
```

---

## Phase 3: Verification

### Task 12: Run full test suite and verify JSON output

**Step 1: Run all tests**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo test`
Expected: all tests pass (51 existing + 5 new = ~56)

**Step 2: Verify JSON output includes new fields**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo run -- -f json 2>/dev/null | python3 -m json.tool | head -40`

Verify that:
- `processes` array entries now include `server_models` (when a server is running)
- `aggregates.models_loaded` shows model IDs
- No hardcoded hostnames appear in the output

**Step 3: Verify daemon subcommand reads from NodeMap**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo run -- daemon status`

Expected: reads node list from `~/.config/asmi/config.json` (or exits with a helpful message if empty)

**Step 4: Verify log file location**

Run: `ls ~/Library/Application\ Support/asmi/asmi.log`
Expected: file exists after running asmi

**Step 5: Final commit (if any fixups needed)**

```bash
git add -A
git commit -m "chore: final verification pass"
```

---

## Summary of Changes

| File | Change | Phase |
|------|--------|-------|
| `src/main.rs:339` | `/tmp/asmi.log` → `dirs::data_local_dir()/asmi/asmi.log` | 1 |
| `src/main.rs:1517-1522` | Delete `DAEMON_NODES` const | 1 |
| `src/main.rs:1523` | `com.r1o.asmi-daemon` → `com.asmi.daemon` | 1 |
| `src/main.rs:1525-1660` | Daemon handler reads `NodeMap.nodes` | 1 |
| `src/main.rs:1610-1617` | Binary path: `current_exe()` → `which` → `~/.cargo/bin/asmi` | 1 |
| `src/main.rs:1648-1649` | Daemon log path + remove `"m3u2"` check | 1 |
| `crates/.../config.rs:150-157` | Respect `XDG_CONFIG_HOME` | 1 |
| `crates/.../types.rs` | Add `ModelServerMetadata`, extend `ProcessInfo` | 2 |
| `crates/.../scanner.rs` | New `parse_v1_models_metadata`, wire into scan | 2 |
| `crates/.../collector.rs` | Add `probe_model_endpoints`, wire into collect | 2 |
| `crates/.../aggregator.rs` | Prefer server-reported model IDs | 2 |
| `crates/.../lib.rs` | Export new types and parser | 2 |
| `crates/.../testdata/v1-models-response.json` | New test fixture | 2 |

**New tests:** 5 (parse_v1_models_metadata × 4, config_path × 1)
**Expected total test count:** ~56
