# Handoff: apple-smi — NodeMap, RDMA IP Discovery, Hierarchical Scanning

## Session Metadata
- Created: 2026-02-21 07:09:45
- Project: /Users/ma/Projects/Personal/apple-smi
- Branch: main
- Session duration: ~90 minutes

### Recent Commits (for context)
  - 80c633c Add node registry: persist discovered hostnames across runs
  - 0442f2c Add config file registry at ~/.config/apple-smi/config.toml
  - e8f38fc Fix dedup: use seed hosts exclusively when provided
  - e9c003f Add loading spinner while cluster data is being fetched
  - 1b0fd6f Fix duplicate nodes from seed hosts + discovery overlap

## Handoff Chain

- **Continues from**: [2026-02-21-060427-mlx-top-activity-log-dedup-jaccl.md](./2026-02-21-060427-mlx-top-activity-log-dedup-jaccl.md)
  - Previous title: mlx-top — Activity Log, Dedup Fix, JACCL Detection
- **Supersedes**: None

## Current State Summary

Implemented a persistent `NodeMap` system that auto-discovers hostname aliases (ARP names like "mac-360" -> canonical SSH names like "m3u2") and Thunderbolt bridge IPs for RDMA/distributed inference. The system uses a three-phase hierarchical scanning approach: (1) fast probe of known seed hosts for instant TUI display, (2) full discovery scan with ARP/Tailscale/Thunderbolt, (3) periodic re-scan. All alias mappings and RDMA IPs persist to `~/.config/mlx-top/config.json`. A TUI merge mode (`m` key) lets users manually map nodes when auto-discovery fails. The `total_nodes` header count now only includes SSH-reachable nodes (was counting all ARP candidates). All changes compile cleanly and pass 46 tests. **Changes are NOT committed yet** — all are in the working tree.

## Codebase Understanding

### Architecture Overview

`mlx-top` is a Rust TUI (ratatui) for monitoring Apple Silicon clusters via SSH. The core library (`mlx-top-core` crate in `crates/cluster-monitor/`) handles discovery, scanning, metrics collection, and state management. The binary (`src/main.rs`) handles CLI parsing, TUI rendering, and persistence.

**Data flow**: Discovery (ARP/Tailscale/TB) -> Scan (SSH probe each peer) -> Metrics (parallel SSH for powermetrics/vmstat/ps) -> ClusterState -> TUI render.

**Key shared state**: `ClusterState` behind `Arc<RwLock<>>`, updated by scan loop and metrics loop, read by TUI render tick. `NodeMap` behind `Arc<RwLock<>>` for alias/RDMA IP persistence.

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `crates/cluster-monitor/src/config.rs` | ClusterConfig + **NodeMap** (aliases, nodes, rdma_ips, hostfile_json generation) | Core of this session's work |
| `crates/cluster-monitor/src/scanner.rs` | Discovery methods + `scan_cluster` + `scan_seeds` — emits AliasDiscovered and RdmaIpsDiscovered events | Alias/RDMA event emission |
| `crates/cluster-monitor/src/monitor.rs` | Background polling loops, `ClusterMonitor` with `node_map: Arc<RwLock<NodeMap>>` | Applies aliases in poll_metrics |
| `crates/cluster-monitor/src/aggregator.rs` | `ClusterState` — `update_scan`, `merge_scan`, `total_nodes` logic | Fixed total_nodes counting |
| `crates/cluster-monitor/src/types.rs` | `ClusterEvent` enum — added `AliasDiscovered`, `RdmaIpsDiscovered` | Event definitions |
| `src/main.rs` | TUI, CLI, event handling, NodeMap load/save, merge mode | Binary entry point |
| `crates/cluster-monitor/src/collector.rs` | Metrics collection via SSH (powermetrics, vmstat, ps, footprint) | Resolves hostname via `hostname -s` |

### Key Patterns Discovered

- **Hostname resolution**: The collector runs `hostname -s` as part of the vmstat command to resolve canonical names. This is how "mac-360" -> "m3u2" mapping happens at the metrics level. However, the first poll cycle can run BEFORE scan completes, using raw seed hostnames.
- **Event-driven architecture**: `ClusterEvent` enum + `EventSink` (broadcast channel) for real-time TUI updates. Events are emitted from scanner/collector tasks, consumed by TUI loop and background handlers.
- **Dedup layers**: (1) `deduplicate_peers` in discovery, (2) `update_scan`/`merge_scan` in aggregator by hostname, (3) NodeMap `resolve_dedup` in poll_metrics, (4) HashMap insert in `update_nodes`.
- **The duplicate node problem**: ARP table returns names like "mac-360", Tailscale returns "m3u2", Thunderbolt returns IPs. Same machine appears as 3+ different identities. NodeMap aliases collapse them.

## Work Completed

### Tasks Finished

- [x] Fix `total_nodes` to only count SSH-reachable nodes (was counting all ARP candidates)
- [x] Add hierarchical scanning: seed probe first, then full discovery
- [x] Add `merge_scan` to ClusterState for non-destructive scan result merging
- [x] Create `NodeMap` with persistent aliases, known nodes, and RDMA IPs
- [x] Auto-discover aliases via `AliasDiscovered` events from scanner
- [x] Auto-discover Thunderbolt bridge IPs via `RdmaIpsDiscovered` events
- [x] Apply NodeMap aliases in poll_metrics to dedup the poll list
- [x] Background event handler to persist NodeMap on alias/RDMA discovery
- [x] TUI merge mode (`m` key) for manual node mapping
- [x] Hostfile JSON generation for mlx.launch (`NodeMap::hostfile_json`)
- [x] Remove old `nodes.json` registry, replaced by `config.json`
- [x] Add `dirs` crate to core library for config path resolution

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `crates/cluster-monitor/src/config.rs` | Added `NodeMap` struct with aliases, nodes, rdma_ips, load/save, resolve, hostfile_json | Persistent alias + RDMA IP map |
| `crates/cluster-monitor/src/types.rs` | Added `AliasDiscovered` and `RdmaIpsDiscovered` to `ClusterEvent` | Event-driven alias/IP discovery |
| `crates/cluster-monitor/src/scanner.rs` | Added `scan_seeds()`, emit alias + RDMA events from `scan_cluster` | Hierarchical scanning + auto-mapping |
| `crates/cluster-monitor/src/monitor.rs` | Added `node_map` field to `ClusterMonitor`, pass to poll_metrics, hierarchical scan phases | Alias resolution in metrics polling |
| `crates/cluster-monitor/src/aggregator.rs` | Changed `total_nodes` to only count `ssh_ok`, added `merge_scan()` | Fix header count, non-destructive merge |
| `crates/cluster-monitor/src/lib.rs` | Export `NodeMap`, `scan_seeds` | Public API |
| `crates/cluster-monitor/Cargo.toml` | Added `dirs = "6"` dependency | Config path resolution in core |
| `src/main.rs` | Rewrote: NodeMap load/save, alias event handler, RDMA event handler, TUI merge mode, removed old registry | Full integration |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| Event-driven alias discovery | (a) Track probed_via in ScanResult, (b) Return alias pairs from scan_cluster, (c) Emit events | Events decouple scanner from persistence. Clean separation — scanner discovers, main.rs persists. |
| NodeMap in config.json not nodes.json | (a) Keep separate files, (b) Single config.json | Single file for all persistent node data. Replaces the old incomplete registry. |
| total_nodes = ssh_ok count only | (a) All scan results, (b) Only reachable | Non-reachable ARP candidates aren't cluster nodes. Header should show "4/4" not "4/14". |
| Hierarchical scan: seeds first | (a) Full scan only, (b) Seeds then full | Seeds appear in TUI in ~2s. Full discovery can take 10-30s (ARP + SSH timeouts). |
| TUI merge via `m` key | (a) Auto-merge by RAM match, (b) Manual merge | Auto-merge is fragile (multiple nodes can have same RAM). Manual is explicit and correct. |
| Store all TB bridge IPs per node | (a) One IP per node, (b) All IPs | User asked about this — nodes can have multiple TB ports. Store all, use first for hostfile. |

## Pending Work

### Immediate Next Steps

1. **Test live on the cluster** — Run `cargo run` and verify: (a) aliases auto-populate in config.json, (b) RDMA IPs captured, (c) duplicates collapse, (d) header shows correct count
2. **Commit all changes** — Large set of uncommitted modifications across 8 files
3. **Verify TUI merge mode** — Select duplicate node, press `m`, navigate to canonical, press `m` again
4. **Check config.json after first run** — Verify aliases and rdma_ips populated correctly

### Blockers/Open Questions

- [ ] The `hostname -s` resolution in the metrics collector may still fail for some ARP names if mDNS is slow. The NodeMap alias fallback should handle this, but needs live testing.
- [ ] User asked about filtering to one RDMA IP per node vs storing all — current implementation stores all, uses first for hostfile generation. May need user preference.
- [ ] Old `nodes.json` registry was deleted. If user has other tools reading it, they'll need updating.

### Deferred Items

- Bonjour discovery (`DiscoveryMethod::Bonjour`) is still unimplemented (TODO in scanner.rs)
- No CLI command to dump hostfile JSON yet (NodeMap::hostfile_json exists but no `--hostfile` flag)
- No CLI command to show/edit aliases (`--show-aliases`, `--add-alias`)

## Context for Resuming Agent

### Important Context

The **root cause** of the 8-node duplicate display was: ARP table returns different hostnames (mac, mac-360, mac-366, mac-368) than SSH `hostname -s` (m3u1, m3u2, m3u3, m4m1). The old code had no persistent alias mapping, so every restart re-discovered the duplicates. The `NodeMap` system fixes this by auto-mapping any discovered name to its SSH canonical name and persisting to `~/.config/mlx-top/config.json`.

`ClusterMonitor::new()` now takes two arguments: `(config, node_map)`. The old single-argument constructor is gone. All test code updated.

The `ActivityLog::handle_event` method now takes `&ClusterEvent` (borrowed) instead of `ClusterEvent` (owned) — this was needed because the event is also consumed by the background alias handler.

The `scan_cluster` function now returns `(DiscoveredPeer, String, ScanResult)` tuples internally from spawned tasks (instead of just `ScanResult`). This preserves the peer info needed to emit alias events. The public API return type is still `Vec<ScanResult>`.

### Assumptions Made

- Thunderbolt bridge IPs are always in the 169.254.x.x range (link-local)
- `hostname -s` returns a consistent, canonical short hostname on all cluster Macs
- The `dirs` crate resolves `config_dir()` to `~/.config` on macOS
- Multiple TB bridge IPs per node is expected (multiple TB ports)

### Potential Gotchas

- The `merge_scan` method in aggregator was introduced to avoid overwriting seed scan results during full scan. If you switch back to `update_scan`, seed-probed nodes will be lost when full scan completes.
- The event broadcast channel has capacity 256. If events aren't drained fast enough (e.g., TUI is slow), older events are dropped. Alias events could be missed. The background handler runs in a dedicated tokio task to minimize this risk.
- The `handle_event` change from owned to borrowed (`&ClusterEvent`) means the event cloning happens at the call site in the TUI loop drain. Watch for lifetime issues if refactoring.

## Environment State

### Tools/Services Used

- Rust toolchain (cargo build/test)
- SSH to cluster nodes (m3u1, m3u2, m3u3, m4m1)
- Tracing logs to `/tmp/mlx-top.log`

### Active Processes

- No active mlx-top process at handoff time

### Environment Variables

- None required — all config via CLI flags and `~/.config/mlx-top/config.json`

## Related Resources

- Config file: `~/.config/mlx-top/config.json` (created on first run with aliases + RDMA IPs)
- Log file: `/tmp/mlx-top.log` (tracing output, useful for debugging scan/alias issues)
- Test data: `crates/cluster-monitor/testdata/` (ARP tables, Tailscale JSON, powermetrics, etc.)

---

**Security Reminder**: Before finalizing, run `validate_handoff.py` to check for accidental secret exposure.
