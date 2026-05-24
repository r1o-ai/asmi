# asmi ↔ Standalone JACCL / mlx.distributed_config Integration

**Date:** 2026-04-26
**Author:** MA + Claude
**Status:** Draft — ready to execute
**Tracks:** [r1o-ai/asmi#1](https://github.com/r1o-ai/asmi/issues/1)

---

## Problem

`src/rdma_autosetup.rs:450-461` deterministically assigns `169.254.{iface_index}.1` link-local IPs as a fallback. The interface index is **local** to each node, so two nodes with `enX` at the same index get the same IP — verified live 2026-04-26: hub:en5 and m3u3:en5 both got `169.254.5.1`. This silently breaks any 3+ node JACCL mesh.

Two related bugs surfaced during the same investigation:
- `src/serve.rs:865` sets `MLX_METAL_FAST_SYNCH=1` unconditionally — known to cause non-deterministic GPU lock with JACCL ([mlx#3142](https://github.com/ml-explore/mlx/issues/3142)).
- `/serve/status` reports `state=ready` on HTTP listener bind, before mlx_lm.server's lazy weight load completes — broken models silently appear healthy.

## Approach

**Stop reimplementing what already exists.** Apple ships `mlx.distributed_config` (binary, in `mlx_lm` package) which handles cross-node IP assignment, cable-pair symmetry, and hostfile generation correctly. The standalone JACCL library (extracted from MLX 2026-04-22) provides a public C++ API and a documented hostfile schema.

Strategy: replace asmi's home-grown autosetup with a thin shell-out to the official tool. Keep asmi's role as the *daemon* (state, metrics, serve lifecycle); delegate the *cluster topology* problem to Apple's tool. Phase 0 establishes the shell-out wrapper; later phases retire each piece of the broken reimplementation.

## Architecture

```
┌──────────────────────────────────────────────┐
│  asmi daemon (Rust, port 9090)              │
│                                              │
│  EXISTING:                                   │
│   /metrics, /processes, /serve/{load,stop}   │
│                                              │
│  REPLACED:                                   │
│   /rdma/setup    → invokes mlx_dist_config   │
│   /jaccl/config  → reads tool output         │
│                                              │
│  NEW:                                        │
│   /serve/status  → 1-token inference probe   │
│   /jaccl/probe   → all-reduce smoke test     │
│                  (via standalone lib FFI)    │
└────────────────┬─────────────────────────────┘
                 │ shell out
                 ▼
┌──────────────────────────────────────────────┐
│  mlx.distributed_config (Apple, in mlx_lm)   │
│   --auto-setup → assigns IPs, writes hostfile│
│   --dot        → cable graph                 │
│   --hostfile   → consume existing            │
└──────────────────────────────────────────────┘
                 │ at launch time
                 ▼
┌──────────────────────────────────────────────┐
│  Standalone libjaccl (CMake FetchContent)   │
│  - jaccl::init() reads JACCL_RANK/COORD/IBV │
│  - all_sum, all_gather, send/recv           │
└──────────────────────────────────────────────┘
```

## Tech Stack

- Rust 1.85 (asmi workspace edition 2024)
- `tokio::process::Command` for shell-outs (already used in rdma_autosetup.rs)
- `serde_json` for parsing `mlx.distributed_config` output
- Optional later: `cmake` build dep + bindgen for direct FFI to libjaccl
- Test fixtures: r1o-ai/asmi cluster — hub + m3u1 + m3u3 (live 3-node TB5)

## Constraints

- **Minimal surface change** — keep existing endpoint paths and JSON shapes stable so downstream consumers (sweep.py, web app, cluster-config-assistant agent) don't break.
- **Graceful degradation** — if `mlx.distributed_config` isn't installed (older mlx_lm), fall back to the current logic with a clear `source: "legacy_fallback"` tag and a deprecation warning in tracing.
- **No sudo regression** — current asmi already invokes `sudo ifconfig` via spawn; the new path can require `sudo` for the same operations but must not require *passwordless* sudo unless explicitly opted in.
- **Tests run on live cluster, not mocks** — per repeated lessons, mocked RDMA tests pass while production RDMA fails. Integration tests assert against real `/rdma/check` output.

## Out of Scope

- Direct Rust FFI to libjaccl (deferred — can use existing `tools/r1o-tui/crates/jaccl-sys`)
- Removing the local mlx_lm `_mlx_backend_fix.pth` patch (separate cleanup, blocked on this work landing)
- Web app `/api/cluster/hostfile/generate` changes (will simplify after asmi is fixed)

---

## Phase 0 — Prereqs (verify before coding)

### Task 0.1 — Confirm `mlx.distributed_config` available on all 3 nodes (2 min)

Each node must have the binary in PATH. mlx_lm 0.31.3 ships it.

```bash
for node in hub m3u1 m3u3; do
  if [ "$node" = "hub" ]; then
    which mlx.distributed_config
  else
    ssh "$node" 'which mlx.distributed_config'
  fi
done
```

**Expected:** `/opt/homebrew/bin/mlx.distributed_config` on each node (or equivalent).

**If missing:** `pip3 install --upgrade mlx_lm` on the offending node.

### Task 0.2 — Capture baseline error count (2 min)

This is iteration 0 of the refine-skill Loop Mode style probe. Numbers we'll re-measure after each phase to confirm forward progress.

```bash
# 1. asmi /rdma/setup IP collisions
for node in hub m3u1 m3u3; do
  curl -s -X POST http://$node:9090/rdma/setup | jq '.ips[] | select(.source=="manual") | .ip'
done | sort | uniq -c | awk '$1 > 1 { print "COLLISION:", $0 }'
# Expected today: COLLISION: 2 169.254.5.1
# Target: 0 collisions

# 2. asmi /serve/status lenient health
curl -s -X POST http://localhost:9090/serve/load -H 'Content-Type: application/json' \
  -d '{"model_path":"/nonexistent","engine":"mlx_lm","backend":"single","port":29999}'
sleep 2
curl -s 'http://localhost:9090/serve/status?port=29999' | jq '.state'
# Expected today: "ready" (false positive)
# Target: "error" or "loading", never "ready" without a real model
```

Record numbers in `docs/runbooks/jaccl-integration-baseline.md`.

### Task 0.3 — Branch (1 min)

```bash
cd ~/Projects/Personal/apple-smi
git checkout -b fix/jaccl-standalone-integration
```

Commit checkpoint: nothing yet (just branch).

---

## Phase 1 — Wrapper module (TDD)

### Task 1.1 — Write failing test for the wrapper (3 min)

New file: `src/mlx_distributed_config.rs`

```rust
//! Thin wrapper around Apple's mlx.distributed_config binary.

use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Hostfile {
    pub backend: String,
    pub envs: Vec<String>,
    pub hosts: Vec<HostEntry>,
}

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct HostEntry {
    pub ssh: String,
    pub ips: Vec<String>,
    pub rdma: Vec<Option<String>>,
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("mlx.distributed_config binary not found in PATH")]
    BinaryNotFound,
    #[error("mlx.distributed_config exited with status {0}: {1}")]
    NonZeroExit(i32, String),
    #[error("invalid JSON output: {0}")]
    JsonParse(#[from] serde_json::Error),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

pub async fn auto_setup(
    hosts: &[String],
    backend: &str,
) -> Result<Hostfile, ConfigError> {
    todo!("Phase 1.2")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn auto_setup_against_live_cluster() {
        // Skip in CI; live test only
        if std::env::var("ASMI_LIVE_CLUSTER_TEST").is_err() {
            eprintln!("skipping — set ASMI_LIVE_CLUSTER_TEST=1 to run");
            return;
        }
        let result = auto_setup(
            &["hub".into(), "m3u1".into(), "m3u3".into()],
            "jaccl-ring",
        )
        .await
        .expect("auto_setup should succeed");
        assert_eq!(result.hosts.len(), 3, "should produce 3 hosts");
        assert!(
            result.hosts.iter().all(|h| !h.ips.is_empty()),
            "every host must have at least one IP"
        );

        // No two hosts share an IP — the bug we're fixing
        let mut all_ips: Vec<&str> = result
            .hosts
            .iter()
            .flat_map(|h| h.ips.iter().map(String::as_str))
            .collect();
        all_ips.sort();
        let before = all_ips.len();
        all_ips.dedup();
        assert_eq!(before, all_ips.len(), "duplicate IPs across hosts");
    }
}
```

Add to `src/main.rs`: `mod mlx_distributed_config;`

```bash
cargo test mlx_distributed_config 2>&1 | tail -5
# Expected: test fails because auto_setup is unimplemented (todo!())
```

**Commit:** `test(mlx-config): failing test for live-cluster auto-setup wrapper`

### Task 1.2 — Implement the wrapper (4 min)

Replace `todo!()` with:

```rust
pub async fn auto_setup(
    hosts: &[String],
    backend: &str,
) -> Result<Hostfile, ConfigError> {
    let tmp = tempfile::NamedTempFile::new()?;
    let tmp_path = tmp.path().to_path_buf();

    let output = tokio::process::Command::new("mlx.distributed_config")
        .args([
            "--verbose",
            "--hosts",
            &hosts.join(","),
            "--over",
            "thunderbolt",
            "--backend",
            backend,
            "--auto-setup",
            "--output-hostfile",
            tmp_path.to_str().unwrap(),
        ])
        .output()
        .await
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ConfigError::BinaryNotFound
            } else {
                ConfigError::Io(e)
            }
        })?;

    if !output.status.success() {
        return Err(ConfigError::NonZeroExit(
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }

    let body = tokio::fs::read_to_string(&tmp_path).await?;
    let hostfile: Hostfile = serde_json::from_str(&body)?;
    Ok(hostfile)
}

pub async fn dot_topology(hosts: &[String]) -> Result<String, ConfigError> {
    let output = tokio::process::Command::new("mlx.distributed_config")
        .args([
            "--hosts",
            &hosts.join(","),
            "--over",
            "thunderbolt",
            "--dot",
        ])
        .output()
        .await
        .map_err(|e| {
            if e.kind() == std::io::ErrorKind::NotFound {
                ConfigError::BinaryNotFound
            } else {
                ConfigError::Io(e)
            }
        })?;

    if !output.status.success() {
        return Err(ConfigError::NonZeroExit(
            output.status.code().unwrap_or(-1),
            String::from_utf8_lossy(&output.stderr).into_owned(),
        ));
    }
    Ok(String::from_utf8_lossy(&output.stdout).into_owned())
}
```

Add `tempfile` and `thiserror` to `Cargo.toml` if not already there.

```bash
ASMI_LIVE_CLUSTER_TEST=1 cargo test mlx_distributed_config -- --nocapture 2>&1 | tail -20
# Expected: passes — 3 hosts, unique IPs
```

**Commit:** `feat(mlx-config): wrapper for auto_setup and dot_topology`

---

## Phase 2 — Replace `rdma_autosetup` IP fallback

### Task 2.1 — Failing test for `/rdma/setup` no-collision invariant (3 min)

Add to `src/rdma_autosetup.rs` test module:

```rust
#[tokio::test]
async fn no_ip_collisions_across_cluster() {
    if std::env::var("ASMI_LIVE_CLUSTER_TEST").is_err() {
        return;
    }
    let nodes = ["hub", "m3u1", "m3u3"];
    let mut all_ips: Vec<String> = vec![];
    for node in nodes {
        let url = if node == "hub" {
            "http://localhost:9090/rdma/setup".to_string()
        } else {
            format!("http://{}:9090/rdma/setup", node)
        };
        let resp: serde_json::Value = reqwest::Client::new()
            .post(&url)
            .send()
            .await
            .unwrap()
            .json()
            .await
            .unwrap();
        for ip_entry in resp["ips"].as_array().unwrap() {
            all_ips.push(ip_entry["ip"].as_str().unwrap().to_string());
        }
    }
    let total = all_ips.len();
    all_ips.sort();
    all_ips.dedup();
    assert_eq!(total, all_ips.len(), "found duplicate IPs across nodes");
}
```

Run pre-fix to confirm it fails with current behavior:

```bash
ASMI_LIVE_CLUSTER_TEST=1 cargo test no_ip_collisions 2>&1 | tail -5
# Expected: FAIL — duplicate 169.254.5.1
```

**Commit:** `test(rdma): collision-detection invariant for live cluster`

### Task 2.2 — Replace fallback with mlx_distributed_config (3 min)

In `src/rdma_autosetup.rs`, delete lines 450-461 (the deterministic IP fallback). Replace with a call to `mlx_distributed_config::auto_setup`:

```rust
// At top of autosetup() — try the official tool first
match crate::mlx_distributed_config::auto_setup(
    &node_map.read().await.hostnames(),
    "jaccl-ring",
).await {
    Ok(hostfile) => {
        // Use the official assignment
        for host_entry in &hostfile.hosts {
            for (idx, ip) in host_entry.ips.iter().enumerate() {
                results.push(InterfaceIp {
                    iface: format!("(via mlx_distributed_config #{idx})"),
                    ip: ip.clone(),
                    source: "mlx_distributed_config".into(),
                });
            }
        }
        // Also stash the hostfile for /jaccl/config to read
        let path = std::path::PathBuf::from(
            std::env::var("HOME").unwrap_or_default()
        )
        .join(".r1o/hostfiles/asmi-auto.json");
        let _ = tokio::fs::create_dir_all(path.parent().unwrap()).await;
        let _ = tokio::fs::write(&path, serde_json::to_string_pretty(&hostfile)?).await;
    }
    Err(crate::mlx_distributed_config::ConfigError::BinaryNotFound) => {
        tracing::warn!(
            "mlx.distributed_config not installed; falling back to legacy autosetup. \
             This path has known IP-collision bugs (see r1o-ai/asmi#1)."
        );
        legacy_assign_ips(&iface, &mut results).await;
    }
    Err(e) => {
        tracing::error!("mlx.distributed_config failed: {e}; falling back");
        legacy_assign_ips(&iface, &mut results).await;
    }
}
```

Move the deleted code into `legacy_assign_ips()` private fn so it's still reachable as a fallback.

```bash
ASMI_LIVE_CLUSTER_TEST=1 cargo test no_ip_collisions 2>&1 | tail -5
# Expected: PASS
```

**Commit:** `fix(rdma): use mlx.distributed_config for IP assignment, fallback to legacy`

### Task 2.3 — Re-run baseline probe (2 min)

```bash
# Re-deploy asmi to all 3 nodes (memory: must codesign after scp)
cargo build --release
for node in hub m3u1 m3u3; do
  if [ "$node" != "hub" ]; then
    scp target/release/asmi $node:/opt/homebrew/bin/asmi
    ssh $node 'codesign -f -s - /opt/homebrew/bin/asmi'
    ssh $node "launchctl kickstart -k gui/$(id -u)/com.asmi.daemon"
  fi
done
# Then re-run Task 0.2 probe
```

**Expected:** 0 collisions across nodes.

**Commit:** none (verification only).

---

## Phase 3 — `/jaccl/config` reads from auto-setup output

### Task 3.1 — Update `jaccl_config_handler` (3 min)

In `src/daemon.rs`, change `jaccl_config_handler` (the GET) to read from `~/.r1o/hostfiles/asmi-auto.json` (written by Phase 2.2). If absent, regenerate via `mlx_distributed_config::auto_setup`. If that fails, fall through to the existing topology synthesis.

```rust
async fn jaccl_config_handler(
    State(state): State<AppState>,
) -> Result<Json<serde_json::Value>, AxumError> {
    let path = std::path::PathBuf::from(
        std::env::var("HOME").unwrap_or_default()
    )
    .join(".r1o/hostfiles/asmi-auto.json");

    if let Ok(body) = tokio::fs::read_to_string(&path).await {
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(&body) {
            return Ok(Json(json));
        }
    }

    // Regenerate
    let hostnames = state.node_map.read().await.hostnames();
    match crate::mlx_distributed_config::auto_setup(&hostnames, "jaccl-ring").await {
        Ok(hostfile) => Ok(Json(serde_json::to_value(hostfile).unwrap())),
        Err(_) => {
            // Fall back to existing topology synthesis
            legacy_jaccl_config(&state).await
        }
    }
}
```

**Commit:** `feat(jaccl): /jaccl/config now reads mlx.distributed_config output`

### Task 3.2 — Smoke-test `/jaccl/config` returns valid hostfile (2 min)

```bash
curl -s http://localhost:9090/jaccl/config | jq '.hosts[].ips | length'
# Expected: 1 1 1 (each host has at least one IP)

curl -s http://localhost:9090/jaccl/config > /tmp/check.json
mlx.launch --hostfile /tmp/check.json --backend jaccl-ring -- echo ok
# Expected: prints "ok" 3 times (one per rank)
```

If the second command works end-to-end, **issue #1 is fixed**.

**Commit:** none (verification).

---

## Phase 4 — `/serve/status` inference smoke probe

### Task 4.1 — Add `verified_inference` field (3 min)

In `src/serve.rs`, extend `ServeStatus` struct:

```rust
#[derive(Serialize, Deserialize)]
pub struct ServeStatus {
    // ... existing fields ...
    pub verified_inference: Option<bool>,
    pub verified_at_ms: Option<u64>,
}
```

State remains `loading` until the smoke probe succeeds. New helper:

```rust
async fn smoke_probe_inference(
    port: u16,
    model_path: &str,
    timeout: Duration,
) -> Result<bool, ProbeError> {
    let client = reqwest::Client::builder()
        .timeout(timeout)
        .build()?;
    let resp = client
        .post(format!("http://localhost:{port}/v1/chat/completions"))
        .json(&serde_json::json!({
            "model": model_path,
            "messages": [{"role": "user", "content": "hi"}],
            "max_tokens": 1,
            "stream": false,
        }))
        .send()
        .await?;
    let body: serde_json::Value = resp.json().await?;
    Ok(body["choices"][0]["message"].is_object())
}
```

Trigger the probe from `wait_for_ready` after the HTTP listener binds — only flip `state=ready` when the probe returns true.

**Commit:** `feat(serve): /serve/status flips ready only after inference smoke probe`

### Task 4.2 — Test against the broken model (3 min)

```bash
curl -s -X POST http://localhost:9090/serve/load -H 'Content-Type: application/json' \
  -d '{"model_path":"/Users/ma/Models/Qwen3.5-122B-A10B-mlx-nvfp4","engine":"mlx_lm","backend":"single","port":19080}'

# Watch state transitions for ~2 minutes
for i in 1 2 3 4 5; do
  sleep 20
  curl -s 'http://localhost:9090/serve/status?port=19080' | jq '{state, verified_inference}'
done
# Expected: state=loading throughout, eventually state=error (because the model
# itself fails to load — see r1o-ai/asmi#? for the Qwen3.5-122B-A10B-nvfp4 weight
# naming bug). NEVER state=ready, since inference probe fails.
```

**Commit:** none (verification).

---

## Phase 5 — `MLX_METAL_FAST_SYNCH` opt-in

### Task 5.1 — Make the env var conditional (2 min)

In `src/serve.rs:865`, change the unconditional `cmd.env("MLX_METAL_FAST_SYNCH", "1")` to:

```rust
// Only set MLX_METAL_FAST_SYNCH for non-distributed serves.
// JACCL has a known GPU-lock bug with FAST_SYNCH (mlx#3142).
if matches!(backend, ServeBackend::Single) {
    cmd.env("MLX_METAL_FAST_SYNCH", "1");
}
```

For `Jaccl` / `JacclRing` / `Ring`, leave it unset.

**Commit:** `fix(serve): MLX_METAL_FAST_SYNCH only for single-node (jaccl GPU-lock bug)`

### Task 5.2 — File issue close-out (1 min)

```bash
gh issue comment r1o-ai/asmi#1 --body-file - <<'EOF'
Fixed in branch fix/jaccl-standalone-integration. Phase 2.2 replaces the
deterministic-index IP fallback with a shell-out to mlx.distributed_config
(Apple's official tool), which handles cross-node deconfliction correctly.
Phase 5 also fixes the related serve.rs:865 unconditional MLX_METAL_FAST_SYNCH
bug for distributed backends. Live test confirms 0 IP collisions across hub/m3u1/m3u3.
EOF
```

(Use `NO_COLOR=1 TERM=dumb gh ...` per `feedback-external-repo-pr-guard`. Use `--body-file -` to take the body from stdin without invoking `cat`/`bat`.)

**Commit:** none.

---

## Phase 6 — Verify end-to-end on live cluster

### Task 6.1 — Re-run sweep harness with 3N JACCL topology (4 min)

Add to `tools/inference-autoresearch/configs/sweep-foundation-2026-04-26.json`:

```json
"topologies": [
  {"id": "1N-hub", "nodes": ["hub"], "backend": "single"},
  {"id": "3N-jaccl-ring", "nodes": ["hub", "m3u1", "m3u3"], "backend": "jaccl-ring",
   "hostfile": "~/.r1o/hostfiles/asmi-auto.json"}
]
```

Re-run for the 35B model (already on hub, must `mlx_lm.share` to m3u1 + m3u3 first):

```bash
asmi-share Qwen3.5-35B-A3B-4bit --to m3u1,m3u3
python3 tools/inference-autoresearch/sweep.py \
  --matrix tools/inference-autoresearch/configs/sweep-foundation-2026-04-26.json \
  --topology 3N-jaccl-ring \
  --model Qwen3.5-35B-A3B-4bit
```

**Expected:** 8 cells complete, agg tok/s shows scaling vs the 1N baseline (per memory: 2-node JACCL on 27B was 35.3 vs 25.5 = +38%; 3N should be similar or better at small batch).

**Commit:** `chore(sweep): add 3N-jaccl-ring topology to foundation matrix`

### Task 6.2 — Open PR (2 min)

```bash
git push -u origin fix/jaccl-standalone-integration
NO_COLOR=1 TERM=dumb gh pr create --repo r1o-ai/asmi \
  --title "fix(rdma): integrate mlx.distributed_config + standalone JACCL lib (closes #1)" \
  --body-file - <<'EOF'
Replaces home-grown rdma_autosetup IP assignment with shell-out to Apple's
official mlx.distributed_config tool. Fixes the deterministic-index collision
bug from #1. Adds inference smoke probe to /serve/status (closes silent
"ready-but-broken" state). Makes MLX_METAL_FAST_SYNCH opt-in for distributed
backends (mlx#3142).

Verified live on hub + m3u1 + m3u3 (TB5 mesh):
- 0 IP collisions across cluster
- /jaccl/config produces a hostfile that mlx.launch accepts
- 3N JACCL-ring sweep on Qwen3.5-35B-A3B-4bit completes 8 cells

Closes #1.
EOF
```

**Commit:** none (PR is the artifact).

---

## Migrations / Side Effects

- **`~/.r1o/hostfiles/asmi-auto.json` is now generated by asmi.** Web app and TUI consumers should read this path. If they currently read `auto.json`, leave a symlink: `ln -sf asmi-auto.json ~/.r1o/hostfiles/auto.json` (during Phase 2.2).
- **r1o-tui's `_mlx_backend_fix.pth` becomes obsolete** — the standalone JACCL lib's clean init path doesn't have the multi-backend race. Schedule removal in a follow-up after this lands. Do NOT remove in this PR.
- **Web app `/api/cluster/hostfile/generate`** can be simplified to `curl localhost:9090/jaccl/config` after this lands. Out of scope here.

## Deliverables

- [ ] `src/mlx_distributed_config.rs` — new wrapper module
- [ ] `src/rdma_autosetup.rs` — `auto_setup` calls wrapper, falls back to `legacy_assign_ips`
- [ ] `src/daemon.rs` — `jaccl_config_handler` reads `asmi-auto.json`
- [ ] `src/serve.rs` — `verified_inference` field + smoke probe; `MLX_METAL_FAST_SYNCH` opt-in
- [ ] Tests: `no_ip_collisions_across_cluster`, `auto_setup_against_live_cluster`
- [ ] PR opened against r1o-ai/asmi
- [ ] Issue #1 referenced + closed in PR description
- [ ] r1o sweep harness `sweep-foundation-2026-04-26.json` extended with 3N-jaccl-ring topology
- [ ] Baseline probe re-run confirms 0 collisions, no false-positive `ready` states

## Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| `mlx.distributed_config --auto-setup` requires passwordless sudo on remotes | High | Test detects this via stderr; fallback path documented in tracing log |
| Phase 4 smoke probe times out on cold-load of large models (96GB+) | Medium | Probe timeout configurable per `/serve/load` request (default 5min for >50GB models) |
| Hostfile schema drift between asmi version and consumer expectations | Low | Pin schema version; add `version: "1"` field to JSON output |
| Live cluster tests flaky due to TB5 cable/thermal issues | Medium | Tests gated by `ASMI_LIVE_CLUSTER_TEST=1` — never run in CI without dedicated hardware |

## Total Time Estimate

- Phase 0: 5 min (verify prereqs)
- Phase 1: 7 min (wrapper + tests)
- Phase 2: 8 min (replace fallback, redeploy, verify)
- Phase 3: 5 min (jaccl_config handler)
- Phase 4: 6 min (smoke probe)
- Phase 5: 3 min (METAL_FAST_SYNCH opt-in + issue comment)
- Phase 6: 6 min (live verify + PR)

**Total active engineering: ~40 min** (excluding model sync + 3N sweep run, which are async)
