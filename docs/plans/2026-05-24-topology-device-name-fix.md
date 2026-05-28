# Fix Topology Device Name Mismatch

**Date:** 2026-05-24  
**Scope:** `apple-smi/src/topology.rs`  
**Parent plan:** `2026-05-24-jaccl-topology-device-selection.md` (Phase 6 — device selection code done, blocked by wrong names)

---

## Problem

The topology ARP-based fallback generates RDMA device names by prepending `rdma_` to **network interface** names (e.g., `en15` → `rdma_en15`). But macOS has two separate interface layers per TB5 port:

| Layer | Examples | Purpose |
|-------|----------|---------|
| RDMA parent | `en2`-`en7` | Kernel-level, hosts `rdma_enX` devices |
| IP network | `en10`, `en15`, `en20` | Carries TCP/IP traffic, has link-local + /30 IPs |

`ibv_devices` shows `rdma_en2`-`rdma_en7`. The topology reports `rdma_en15` — which doesn't exist in ibverbs, causing `jaccl_init_mesh` to hang.

`mlx.distributed_config` uses the correct names (`en5/en4` → `rdma_en5/rdma_en4`) but fails when ANY host in the list is unreachable (e.g., marmac offline → SSH timeout → "no topology output" → ARP fallback).

## Evidence

```
# mlx.distributed_config with only online hosts → CORRECT names
$ mlx.distributed_config --hosts hub,m3u4 --over thunderbolt --backend jaccl --dot
  a -- b [label="en5/en4"]    ← rdma_en5/rdma_en4 ← VALID ibverbs devices

# ARP fallback → WRONG names
topology link: rdma_en15/rdma_en10  ← NOT ibverbs devices, hang on init

# ibverbs truth
$ ibv_devices
  rdma_en2 (Down), rdma_en3 (Active), rdma_en4 (Active),
  rdma_en5 (Active), rdma_en6 (Down), rdma_en7 (Down)
```

## Architecture

```
Current:   discover_topology(all_hosts)
           → discover_via_mlx(all_hosts)  ← FAILS (SSH to offline hosts)
           → discover_via_arp(all_hosts)  ← WRONG device names

Fixed:     discover_topology(all_hosts)
           → filter_reachable(all_hosts)  ← NEW: pre-filter
           → discover_via_mlx(online_only) ← WORKS with correct names
           → discover_via_arp(online_only) ← fallback, NOW with device mapping
```

## Tasks

### Task 1: Filter unreachable hosts before calling mlx.distributed_config (5 min)

**File:** `src/topology.rs`

`discover_topology()` currently passes ALL configured hosts (including offline ones like `marmac`) to `mlx.distributed_config`. When any host is unreachable, the tool fails entirely. Filter first.

```rust
pub fn discover_topology(hosts: &[String], backend: &str) -> Result<TopologyReport> {
    // Pre-filter: only pass hosts we can SSH to (5s timeout)
    let reachable: Vec<String> = hosts
        .iter()
        .filter(|h| {
            let ok = is_host_reachable(h);
            if !ok {
                eprintln!("topology: skipping unreachable host {h}");
            }
            ok
        })
        .cloned()
        .collect();

    if reachable.len() < 2 {
        bail!("need at least 2 reachable hosts for topology, got {}", reachable.len());
    }

    match discover_via_mlx(&reachable, backend) {
        Ok(mut report) => {
            // Re-add unreachable hosts as nodes with no links
            for h in hosts {
                if !report.nodes.contains(h) {
                    report.nodes.push(h.clone());
                }
            }
            // Recalculate missing links against full host set
            report.missing_links = compute_missing_links(&report.nodes, &report.links);
            report.mesh_complete = report.missing_links.is_empty();
            report.jaccl_ready = report.mesh_complete;
            Ok(report)
        }
        Err(e) => {
            eprintln!("mlx.distributed_config failed ({e:#}), falling back to ARP-based discovery");
            discover_via_arp(hosts)
        }
    }
}
```

Add SSH reachability check (fast, uses TCP connect only):

```rust
fn is_host_reachable(host: &str) -> bool {
    let user = std::env::var("USER").unwrap_or_else(|_| "root".to_string());
    Command::new("ssh")
        .args([
            "-o", "ConnectTimeout=3",
            "-o", "StrictHostKeyChecking=no",
            "-o", "BatchMode=yes",
            "-o", "LogLevel=ERROR",
            &format!("{user}@{host}"),
            "true",
        ])
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false)
}
```

**Verify:** `cargo check --features jaccl`

**Commit:** `fix(topology): filter unreachable hosts before mlx.distributed_config`

---

### Task 2: Map network interfaces to RDMA devices in ARP fallback (5 min)

**File:** `src/topology.rs`

When the ARP fallback runs (e.g., `mlx.distributed_config` binary not installed), it discovers links via network interfaces (`en15`) but reports them as RDMA devices (`rdma_en15`). Fix by building the actual mapping from IORegistry.

Add `collect_rdma_interface_map()`:

```rust
/// Build mapping from network interface to RDMA device via IORegistry.
/// e.g., "en5" → "rdma_en5", "en3" → "rdma_en3"
/// Only includes interfaces that have a child AppleThunderboltRDMAInterface.
fn collect_rdma_interface_map(host: &str) -> HashMap<String, String> {
    // ioreg -l output: rdma_enX appears as child of its parent IOEthernetInterface (enX)
    // We track the last seen "BSD Name" and map it when we see rdma_enX
    let output = ssh_cmd(host, "ioreg -l 2>/dev/null | grep -E '\"BSD Name\"|rdma_en'");
    let mut result = HashMap::new();
    let mut parent_bsd = String::new();

    for line in output.lines() {
        if let Some(caps) = line.find("\"BSD Name\" = \"") {
            let rest = &line[caps + 14..];
            if let Some(end) = rest.find('"') {
                parent_bsd = rest[..end].to_string();
            }
        }
        if let Some(caps) = line.find("rdma_en") {
            // Extract rdma_enN from: +-o rdma_en5  <class ...>
            let rest = &line[caps..];
            let rdma_name: String = rest.chars().take_while(|c| c.is_alphanumeric() || *c == '_').collect();
            if !parent_bsd.is_empty() && rdma_name.starts_with("rdma_en") {
                result.insert(parent_bsd.clone(), rdma_name);
                parent_bsd.clear();
            }
        }
    }
    result
}
```

Then in `discover_via_arp()`, also collect the RDMA device maps per node and use them when building TopologyLink:

```rust
// In discover_via_arp, after collecting ARP data:

// Collect RDMA device maps per node (ioreg-based)
let rdma_maps: HashMap<String, HashMap<String, String>> = hosts.iter()
    .map(|h| (h.clone(), collect_rdma_interface_map(h)))
    .collect();

// When building links, look up the real RDMA device:
let (dev_a, dev_b) = if host < remote_host {
    let real_a = rdma_maps.get(host)
        .and_then(|m| m.get(local_iface).cloned())
        .unwrap_or_else(|| format!("rdma_{local_iface}"));
    let real_b = rdma_maps.get(remote_host)
        .and_then(|m| m.get(remote_iface).cloned())
        .unwrap_or_else(|| format!("rdma_{remote_iface}"));
    (real_a, real_b)
} else {
    // ... same with swapped roles
};
```

**Problem:** The ARP fallback discovers links via high-numbered interfaces (en15, en20), but the RDMA map maps low-numbered interfaces (en3→rdma_en3). We'd need an additional mapping: en15 → en3 (same TB port, different layer).

**Simpler alternative:** Since Task 1 should make `mlx.distributed_config` work in most cases, the ARP fallback can simply **not prepend `rdma_`** and instead store the raw network interface name. The transfer code's device validation (already implemented) will detect these as invalid and fall back to `new_auto`.

```rust
// In discover_via_arp, change device naming to clearly mark as network interface:
let (dev_a, dev_b) = if host < remote_host {
    (format!("iface:{local_iface}"), format!("iface:{remote_iface}"))
} else {
    (format!("iface:{remote_iface}"), format!("iface:{local_iface}"))
};
```

This makes the fallback honest — it knows the network interface but NOT the RDMA device. Consumers can see `iface:en15` and know it's not a valid RDMA device name.

**Verify:** `cargo check --features jaccl && cargo test -- topology`

**Commit:** `fix(topology): differentiate network interface names from RDMA device names`

---

### Task 3: Handle `iface:` prefix in transfer device resolution (2 min)

**File:** `src/transfer.rs`

Update `resolve_device_for_peer` to reject `iface:` prefixed names (ARP fallback devices) and fall through to `new_auto`:

```rust
// In resolve_device_for_peer, after extracting (local_dev, peer_dev):
if local_dev.starts_with("iface:") || peer_dev.starts_with("iface:") {
    tracing::warn!(
        %local_dev, %peer_dev,
        "topology has interface names not RDMA devices, falling back to auto-discover"
    );
    return None;
}
```

**Verify:** `cargo check --features jaccl`

**Commit:** `feat(transfer): reject non-RDMA device names from topology`

---

### Task 4: Extract `compute_missing_links` helper (2 min)

**File:** `src/topology.rs`

The missing-links calculation is duplicated in `parse_dot()` and `discover_via_arp()`. Extract to a shared helper (used by Task 1 too):

```rust
fn compute_missing_links(nodes: &[String], links: &[TopologyLink]) -> Vec<(String, String)> {
    let connected: HashSet<(String, String)> = links.iter().map(|l| {
        if l.node_a < l.node_b {
            (l.node_a.clone(), l.node_b.clone())
        } else {
            (l.node_b.clone(), l.node_a.clone())
        }
    }).collect();

    let mut missing = Vec::new();
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let pair = (nodes[i].clone(), nodes[j].clone());
            if !connected.contains(&pair) {
                missing.push(pair);
            }
        }
    }
    missing
}
```

**Verify:** `cargo test -- topology`

**Commit:** `refactor(topology): extract compute_missing_links helper`

---

### Task 5: Live test — full flow (5 min)

```bash
# Rebuild and deploy
cd /Users/ma/Projects/Personal/apple-smi
cargo build --release --features jaccl

# Restart hub + m3u4
kill $(lsof -ti :9090); sleep 2
nohup ./target/release/asmi --serve --cluster --bind 0.0.0.0 --port 9090 > /tmp/asmi-new.log 2>&1 &
scp target/release/asmi m3u4:/tmp/asmi-new
ssh m3u4 'kill $(lsof -ti :9090); sleep 2; nohup /tmp/asmi-new --serve --cluster --bind 0.0.0.0 --port 9090 > /tmp/asmi-new.log 2>&1 &'

sleep 30  # wait for topology scan

# Verify topology has correct RDMA device names
curl -sf http://localhost:9090/topology | python3 -c "
import json, sys
d = json.load(sys.stdin)
for l in d['links']:
    if 'm3u4' in [l['node_a'], l['node_b']]:
        print(f\"{l['node_a']}↔{l['node_b']}: {l['device_a']}↔{l['device_b']}\")
        assert l['device_a'].startswith('rdma_en'), f'wrong: {l[\"device_a\"]}'
        # Verify device number is < 10 (real ibverbs devices)
"

# Transfer test
curl -sf http://localhost:9090/transfer \
  -X POST -H 'Content-Type: application/json' \
  -d '{"model_dir":"jaccl-test-model","peer":"m3u4","direction":"send"}' \
  --no-buffer
```

**Expected:**
```
hub↔m3u4: rdma_en5↔rdma_en4   ← correct ibverbs devices, not en15/en10
data: {"type":"done","transport":"jaccl-rdma","durationMs":...}
```

**Victory:** Transfer completes. File verified on m3u4.

**Commit:** `test(transfer): verify RDMA transfer with topology-resolved devices`

---

## Risks

| Risk | Mitigation |
|------|------------|
| `is_host_reachable` adds 3s per offline host | Runs in parallel via thread scope, total ~3s |
| `mlx.distributed_config` still fails with online-only hosts | ARP fallback with `iface:` prefix, transfer uses `new_auto` |
| Tests use hardcoded `rdma_en3/en4` device names | Tests use DOT input which goes through `parse_dot`, not ARP fallback |
| Thread scope in `discover_via_arp` adds overhead for reachability checks | Only needed when `mlx.distributed_config` fails first |

## Total estimate

~20 min implementation, ~5 min live test.
