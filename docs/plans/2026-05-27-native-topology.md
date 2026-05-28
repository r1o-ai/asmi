# Implementation plan — Native TB UUID topology discovery (v3)

**Date:** 2026-05-27
**Companion research:** Conversation evidence (wiki + live SPThunderboltDataType probes on all 4 nodes)
**Companion eval:** ARDD critic REJECT v1 (7 findings), ACCEPT_WITH_RESERVATIONS v2 (2 findings). v3 simplifies HTTP to `curl` subprocess — eliminates both v2 blockers.
**Working branch:** `feat/native-topology`
**Repo:** `/Users/ma/Projects/Personal/apple-smi/`

---

## Working protocol

Apply dependency-scanner framework before every multi-file edit. See Iron Laws for the non-negotiables this work obeys.

## Key simplification (v2 → v3)

v2 added `reqwest::blocking` feature, which caused two critic blockers: panic inside `#[tokio::main]` CLI path, and `Client` has no `Default` impl. 

v3 eliminates both by using `Command::new("curl")` — the same subprocess pattern used everywhere else in the codebase (`system_profiler`, `networksetup`, `ssh`, `ifconfig`, `ibv_devinfo`). Fully sync, zero dependency changes, no Cargo.toml edit, works in any context (async or sync).

## Architecture

```
Each node:  system_profiler SPThunderboltDataType → extract domain_uuid per bus + peer domain_uuid
            networksetup -listallhardwareports → bus_map (bus_index → enN)
               ↓
            GET /thunderbolt returns both UUID data and bus_map per node
               ↓
Hub:        curl http://<node>:9090/thunderbolt (5s timeout, parallel via thread::scope)
               ↓
            UUID cross-match: nodeA.peer_domain == nodeB.local_domain (both directions)
               ↓
            Bus-to-interface: use EACH NODE'S own bus_map (not hardcoded)
               ↓
            TopologyReport { nodes, links, raw_dot, ... } — identical output format
```

No Python. No MLX. No new dependencies. `curl` replaces SSH for data collection.

## Evidence citations

| Fact | Source |
|---|---|
| `domain_uuid_key` is firmware-derived, stable across boots | Wiki: `tb-cable-uuid-identity` |
| Cable identified when `nodeA.peer_uuid == nodeB.local_uuid` AND vice versa | Wiki: `tb-cable-uuid-identity` |
| `SPThunderboltDataType` has `domain_uuid_key` at bus level, peer `domain_uuid_key` in `_items[]` | Live probe on hub, m3u2, m3u3, m3u4 (this session) |
| All 6 links discoverable via UUID cross-match across 4 nodes | Live cross-reference of all node SPThunderboltDataType output (this session) |
| Codebase uses `Command::new` for all external calls (system_profiler, networksetup, ssh, ifconfig) | grep of `src/topology.rs`, `src/daemon.rs`, `src/rdma_autosetup.rs` |
| `discover_topology` is sync, called via `spawn_blocking` from 2 async sites + 1 CLI site | `daemon_startup.rs:291`, `rdma_autosetup.rs:227`, `main.rs:199` |
| `/jaccl/config` handler at `daemon.rs:200` builds response from live topology cache | `daemon.rs:200-320` |
| `/topology/dot` handler at `daemon.rs:941` returns `report.raw_dot.clone()` | `daemon.rs:941-947` |
| `is_host_reachable` at `topology.rs:111` uses SSH | `topology.rs:111-125` |
| `networksetup -listallhardwareports` gives "Thunderbolt N → Device: enX" uniformly | Live probe on hub + m3u3 |
| `reqwest` 0.12 in Cargo.toml lacks `blocking` feature; all usage is async | `Cargo.toml:45`, `grep -rn reqwest src/` |

## Tech stack

- Rust (existing asmi binary crate)
- `serde_json` for SPThunderboltDataType parsing (already a dep)
- `curl` via `std::process::Command` for HTTP fan-out (already on all nodes)
- No new dependencies, no Cargo.toml changes

## Tasks

### Phase A — Extend /thunderbolt with UUIDs and bus map

#### Task A1 — Add UUID fields + bus_map to scan_thunderbolt() — 15 min

**Pre-flight:** `src/daemon.rs:808-901`. Consumers: background loop (daemon_startup.rs:242), `/thunderbolt` handler (daemon.rs:904). Both consume `serde_json::Value` — additive fields are non-breaking.

**Edit:** `src/daemon.rs` — `scan_thunderbolt()` function

1. Extract `domain_uuid_key` and `switch_uid_key` per bus (from bus-level JSON).
2. Extract peer `domain_uuid_key` from first `_items[]` entry (connected peer).
3. Parse `_name` field to extract bus index (`"thunderboltusb4_bus_3"` → `3`).
4. Add to each port's JSON: `domain_uuid`, `switch_uid`, `peer_domain_uuid`, `bus_index`.
5. Build `bus_map` by parsing `networksetup -listallhardwareports` inline. Add to response root: `"bus_map": {"0": "en2", "1": "en3", ...}`.

```rust
// Bus-to-interface mapping from networksetup
fn build_bus_map() -> serde_json::Value {
    let output = std::process::Command::new("networksetup")
        .args(["-listallhardwareports"])
        .output()
        .ok();
    let mut map = serde_json::Map::new();
    if let Some(out) = output {
        let text = String::from_utf8_lossy(&out.stdout);
        let lines: Vec<&str> = text.lines().collect();
        for (i, line) in lines.iter().enumerate() {
            // "Hardware Port: Thunderbolt N" (skip "Thunderbolt Bridge")
            if line.contains("Thunderbolt") && !line.contains("Bridge") {
                if let Some(n) = line.split("Thunderbolt ").nth(1).and_then(|s| s.trim().parse::<u8>().ok()) {
                    if let Some(dev_line) = lines.get(i + 1) {
                        if let Some(dev) = dev_line.strip_prefix("Device: ") {
                            map.insert((n - 1).to_string(), serde_json::json!(dev.trim()));
                        }
                    }
                }
            }
        }
    }
    serde_json::Value::Object(map)
}
```

**Verify:** `cargo check` + restart asmi + verify:
```bash
curl -s http://localhost:9090/thunderbolt | python3 -c "
import json, sys; d=json.load(sys.stdin)
print('bus_map:', d.get('bus_map', {}))
for p in d['ports']:
    print(f'  bus {p.get(\"bus_index\")}: domain={p.get(\"domain_uuid\",\"\")[:8]}... peer={p.get(\"peer_domain_uuid\",\"\")[:8]}...')
"
```

**Commit:** `[native-topo] add domain_uuid, peer_domain_uuid, bus_map to /thunderbolt`

### Phase B — Native topology discovery

#### Task B1 — Add discover_via_native() — 20 min

**Pre-flight:** New function in `src/topology.rs`. Returns `Result<TopologyReport>`. Must populate ALL fields including `raw_dot` (consumed by `topology_dot_handler` at daemon.rs:941 and CLI `--format dot`). Uses `Command::new("curl")` for HTTP — fully sync, works in `spawn_blocking` and CLI context alike.

**Edit:** `src/topology.rs` — add new function + helpers

```rust
/// Fetch a node's /thunderbolt data via curl (sync, 5s timeout).
fn fetch_thunderbolt(host: &str) -> Option<serde_json::Value> {
    let url = format!("http://{}:9090/thunderbolt", host);
    let output = Command::new("curl")
        .args(["-s", "--connect-timeout", "5", "--max-time", "5", &url])
        .output()
        .ok()?;
    if !output.status.success() { return None; }
    serde_json::from_slice(&output.stdout).ok()
}

/// HTTP health check via curl (3s timeout). Replaces SSH is_host_reachable.
fn is_host_reachable_http(host: &str) -> bool {
    let url = format!("http://{}:9090/health", host);
    Command::new("curl")
        .args(["-s", "--connect-timeout", "3", "--max-time", "3", "-o", "/dev/null", "-w", "%{http_code}", &url])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).starts_with("200"))
        .unwrap_or(false)
}

#[derive(Debug)]
struct NativeBusInfo {
    bus_index: u8,
    domain_uuid: String,
    peer_domain_uuid: Option<String>,
    interface: String,
}

/// Native topology: curl to each node's /thunderbolt, cross-match domain_uuids.
fn discover_via_native(hosts: &[String]) -> Result<TopologyReport> {
    // 1. Parallel HTTP fan-out via thread::scope (one thread per host, 5s timeout each)
    let node_data: Vec<(String, serde_json::Value)> = std::thread::scope(|s| {
        let handles: Vec<_> = hosts.iter().map(|h| {
            let h = h.clone();
            s.spawn(move || fetch_thunderbolt(&h).map(|v| (h, v)))
        }).collect();
        handles.into_iter().filter_map(|h| h.join().unwrap()).collect()
    });

    if node_data.len() < 2 {
        bail!("need at least 2 reachable nodes, got {}", node_data.len());
    }

    // 2. Parse bus info from each node's /thunderbolt response
    //    Extract: hostname, bus_map, Vec<NativeBusInfo>

    // 3. UUID cross-match: for each bus with peer_domain_uuid,
    //    find the remote node whose bus has matching domain_uuid.
    //    Both directions must reciprocate.

    // 4. Create TopologyLink using each node's own bus_map for interface resolution.
    //    device = format!("rdma_{}", bus_map[bus_index])

    // 5. Assemble report
    let nodes: Vec<String> = hosts.to_vec();
    let raw_dot = generate_dot(&nodes, &links);  // populate for /topology/dot
    let missing = compute_missing_links(&nodes, &links);
    let mesh_complete = missing.is_empty();
    let n = nodes.len();
    let jaccl_ready = mesh_complete && links.len() >= n * (n - 1) / 2;
    let jaccl_ready_subsets = find_jaccl_subsets_from_links(&nodes, &links);

    Ok(TopologyReport { nodes, links, mesh_complete, missing_links: missing, jaccl_ready, jaccl_ready_subsets, raw_dot })
}
```

**Verify:** `cargo check`. Then save MLX baseline and compare:
```bash
curl -s http://localhost:9090/topology > /tmp/topo-mlx-baseline.json
# (after wiring native path in B2)
curl -s http://localhost:9090/topology > /tmp/topo-native.json
python3 -c "
import json
a = json.load(open('/tmp/topo-mlx-baseline.json'))
b = json.load(open('/tmp/topo-native.json'))
al = sorted([(l['node_a'],l['device_a'],l['node_b'],l['device_b']) for l in a['links']])
bl = sorted([(l['node_a'],l['device_a'],l['node_b'],l['device_b']) for l in b['links']])
assert al == bl, f'MISMATCH:\n  MLX:    {al}\n  Native: {bl}'
print(f'MATCH: {len(al)} links identical')
assert len(b.get('raw_dot','')) > 0, 'raw_dot is empty!'
print('raw_dot: OK')
"
```

**Commit:** `[native-topo] native UUID-based topology discovery via curl`

#### Task B2 — Wire native as primary, keep mlx as permanent fallback — 10 min

**Pre-flight:** `discover_topology()` at topology.rs:64. Three call sites: `daemon_startup.rs:292`, `rdma_autosetup.rs:228`, `main.rs:199`. All call `discover_topology` — none call internals directly.

**Edit:** `src/topology.rs` — replace `discover_topology()`

```rust
pub fn discover_topology(hosts: &[String], backend: &str) -> Result<TopologyReport> {
    // Primary: native HTTP + UUID cross-match
    let http_reachable: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = hosts.iter().map(|h| {
            let h = h.clone();
            s.spawn(move || if is_host_reachable_http(&h) { Some(h) } else { None })
        }).collect();
        handles.into_iter().filter_map(|h| h.join().unwrap()).collect()
    });

    for h in hosts {
        if !http_reachable.contains(h) {
            eprintln!("topology: {h} not reachable via HTTP");
        }
    }

    if http_reachable.len() >= 2 {
        match discover_via_native(&http_reachable) {
            Ok(mut report) => {
                for h in hosts {
                    if !report.nodes.contains(h) { report.nodes.push(h.clone()); }
                }
                let missing = compute_missing_links(&report.nodes, &report.links);
                report.mesh_complete = missing.is_empty();
                report.missing_links = missing;
                let n = report.nodes.len();
                report.jaccl_ready = report.mesh_complete && report.links.len() >= n * (n - 1) / 2;
                report.jaccl_ready_subsets = find_jaccl_subsets_from_links(&report.nodes, &report.links);
                return Ok(report);
            }
            Err(e) => eprintln!("native topology failed ({e:#}), trying mlx fallback"),
        }
    }

    // Fallback: mlx.distributed_config (kept permanently for nodes without asmi)
    let ssh_reachable: Vec<String> = std::thread::scope(|s| {
        let handles: Vec<_> = hosts.iter().map(|h| {
            let h = h.clone();
            s.spawn(move || if is_host_reachable(&h) { Some(h) } else { None })
        }).collect();
        handles.into_iter().filter_map(|h| h.join().unwrap()).collect()
    });

    if ssh_reachable.len() >= 2 {
        match discover_via_mlx(&ssh_reachable, backend) {
            Ok(mut report) => { /* same post-processing */ return Ok(report); }
            Err(e) => eprintln!("mlx fallback failed ({e:#}), trying ARP"),
        }
    }

    discover_via_arp(hosts)
}
```

Rename existing `is_host_reachable` → `is_host_reachable_ssh` for clarity.

**Verify:** Restart asmi. Check logs for "native topology" path (not "mlx.distributed_config"). Verify `/topology` returns 6 links. Verify `/topology/dot` returns non-empty DOT.

**Commit:** `[native-topo] native primary, mlx permanent fallback, HTTP reachability`

### Phase C — Remove mlx from autosetup path

#### Task C1 — Generate hostfile from topology, delete run_mlx_distributed_config — 15 min

**Pre-flight:** `run_mlx_distributed_config()` at rdma_autosetup.rs:305 called only from `autosetup()` Step 3. The MLX-generated hostfile schema (consumed by `mlx.launch --hostfile` via serve.rs):
```json
{
    "backend": "jaccl",
    "hosts": [
        { "ssh": "127.0.0.1", "ips": ["10.1.10.70"], "rdma": [null, "rdma_en4", "rdma_en3", "rdma_en5"] },
        { "ssh": "m3u2", "ips": ["10.1.10.50"], "rdma": ["rdma_en4", null, "rdma_en5", "rdma_en3"] }
    ]
}
```

**Edit:** `src/rdma_autosetup.rs`

Replace Step 3 block with `write_jaccl_hostfile_from_topology()` that produces the exact MLX-compatible schema above. Key fields:
- `"backend": "jaccl"` at root
- `ssh`: `"127.0.0.1"` for local (rank 0), hostname for remote
- `ips`: array with one LAN IP (from Tailscale or en0/en1, fetched via `curl http://<host>:9090/health` which returns hostname, or from known config)
- `rdma`: N-element array — `rdma[j]` = device connecting this host to rank j (from topology links), null for self

Delete `run_mlx_distributed_config()` (dead code).

**Verify:** `cargo check` + restart asmi. Save MLX-generated baseline first:
```bash
cp ~/.r1o/hostfiles/auto.json /tmp/hostfile-mlx-baseline.json
# (after restart with native path)
python3 -c "
import json
a = json.load(open('/tmp/hostfile-mlx-baseline.json'))
b = json.load(open('/Users/ma/.r1o/hostfiles/auto.json'))
assert 'backend' in b, 'missing backend key'
assert all('ssh' in h and 'ips' in h and 'rdma' in h for h in b['hosts']), 'missing host keys'
print(f'Schema OK: {len(b[\"hosts\"])} hosts')
"
```

**Commit:** `[native-topo] generate JACCL hostfile from topology, delete run_mlx_distributed_config`

### Phase D — Tests and verification

#### Task D1 — Unit tests — 10 min

**Edit:** `src/topology.rs` — add tests

```rust
#[test]
fn test_bus_name_parsing() {
    // "thunderboltusb4_bus_3" → bus_index 3
}

#[test]
fn test_uuid_cross_match() {
    // Two nodes with reciprocal domain_uuids → correct TopologyLink
}

#[test]
fn test_bus_map_parsing() {
    // Canned networksetup output for M3 Ultra → {0:"en2", 1:"en3", ...}
}
```

**Verify:** `cargo test`

**Commit:** `[native-topo] unit tests for native topology`

#### Task D2 — End-to-end verification — 5 min

No commit — verification only:
```bash
# Compare native vs MLX topology links (identical)
# Compare native vs MLX hostfile schema (compatible)
# Scramble hub en4, restart asmi, verify correct IP assignment
# Verify /topology/dot returns valid DOT graph
```

## File touch matrix

| File | Lines added | Lines removed | Notes |
|---|---|---|---|
| `src/daemon.rs` | ~30 | 0 | UUID fields + bus_map in scan_thunderbolt() |
| `src/topology.rs` | ~130 | ~10 | discover_via_native, fetch_thunderbolt, is_host_reachable_http, tests. Keep discover_via_mlx. |
| `src/rdma_autosetup.rs` | ~60 | ~100 | write_jaccl_hostfile_from_topology, delete run_mlx_distributed_config |

**Total:** ~220 LOC added, ~110 removed. Net: +110 LOC. Zero Cargo.toml changes.

## Risk register

| Risk | Mitigation |
|---|---|
| Bus-to-interface differs by Mac model | Each node's `/thunderbolt` includes its own `bus_map` from `networksetup`. No hardcoded offset. |
| Remote asmi not running | Fallback chain: native (curl) → mlx (SSH+Python) → ARP. Permanent — mlx never deleted. |
| curl timeout on hung peer | `--connect-timeout 5 --max-time 5`. Thread-per-host via `thread::scope` — one hung node doesn't block others. |
| Dock/hub connected | UUID cross-match requires both directions. Non-cluster devices won't reciprocate. |
| Hostfile schema drift | Verify against MLX baseline in Task C1. Produce exact same keys: `backend`, `ssh`, `ips`, `rdma`. |
| `raw_dot` empty | `generate_dot()` called explicitly in `discover_via_native`. Verified in D2. |

## Rollback strategy

| Failure point | Action |
|---|---|
| Native discovery wrong | `discover_via_mlx` fires automatically as fallback. |
| Hostfile generation wrong | Revert C1 — `run_mlx_distributed_config` in git history. |
| Full revert | `git revert` the branch. Zero schema/dep changes. |

## Acceptance criteria

1. `curl http://localhost:9090/topology` returns 6 links identical to MLX baseline (diff verified)
2. `curl http://localhost:9090/topology/dot` returns non-empty DOT graph
3. `asmi topology --hosts hub,m3u2,m3u3,m3u4` succeeds without Python/MLX (native path)
4. IP assignment via `assign_topology_ips()` assigns correct /30 IPs after restart
5. JACCL hostfile at `~/.r1o/hostfiles/auto.json` has MLX-compatible schema (`backend`/`ssh`/`ips`/`rdma`)
6. `cargo test` passes (including new native topology tests)
7. `discover_via_mlx` retained as permanent fallback (grep confirms)
8. HTTP fan-out has 5s per-node timeout (curl `--max-time 5`)
9. Zero Cargo.toml changes

## Out of scope

- Dedicated `GET /links` endpoint (follow-up — `/thunderbolt` + `bus_map` sufficient)
- JACCL standalone env var migration (`JACCL_*` vs `MLX_*`) — separate task
- Removing Python/MLX from nodes — other features use it
- Deleting `discover_via_mlx` — kept permanently
