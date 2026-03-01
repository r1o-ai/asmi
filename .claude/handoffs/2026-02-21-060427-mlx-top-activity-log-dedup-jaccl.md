# Handoff: mlx-top — Activity Log, Dedup Fix, JACCL Detection

## Session Metadata
- Created: 2026-02-21 06:04:27
- Project: /Users/ma/Projects/Personal/apple-smi
- Branch: main
- Session duration: ~3 hours (continued from a previous session)

### Recent Commits (for context)
  - 80c633c Add node registry: persist discovered hostnames across runs
  - e8f38fc Fix dedup: use seed hosts exclusively when provided
  - e9c003f Add loading spinner while cluster data is being fetched
  - 1b0fd6f Fix duplicate nodes from seed hosts + discovery overlap
  - ef8777d Fix false JACCL detection on single-node processes
  - 5c3edeb Fix seed hosts not used for first metrics poll
  - dbd734f Add roadmap
  - 7bb7a21 Initial release

## Handoff Chain

- **Continues from**: None (first handoff, but builds on a previous session that ran out of context)
- **Supersedes**: None

## Current State Summary

**mlx-top** (formerly apple-smi) is an nvidia-smi equivalent for Apple Silicon Mac clusters. It's a Rust workspace with two crates: `mlx-top` (binary with clap CLI + ratatui TUI) and `mlx-top-core` (library for SSH-based metrics collection, node discovery, and cluster monitoring). The project was renamed from `apple-smi` to `mlx-top` this session. Binaries are `mlx-top` (primary) and `asmi` (alias).

This session added: (1) a real-time event system with `ClusterEvent` broadcast channel for live TUI activity logging, (2) progress gauge during scan/probe, (3) parallel discovery methods (was sequential ~26s, now concurrent ~6s), (4) parallel footprint lookups, (5) hostname resolution via SSH to fix IP/hostname dedup issues, (6) removed Tailscale from default discovery. Node deduplication has been a recurring issue across multiple iterations — IPs from different discovery methods (TB 169.254.x, ARP 10.1.x) appearing as separate nodes.

**Two items remain unfinished:**
1. **JACCL detection** — The JACCL/distributed backend detection infrastructure exists (DistributedBackend enum, --backend flag parsing) but the user reports it's not being detected in the TUI. Needs investigation.
2. **Non-cluster nodes showing** — ARP discovery finds ALL Macs on the network (e.g., "msc-360"), not just cluster nodes. Need filtering strategy.

## Codebase Understanding

### Architecture Overview

Rust workspace with two members:
- **`mlx-top`** (root) — clap CLI + ratatui TUI, reads from shared ClusterState
- **`mlx-top-core`** (`crates/cluster-monitor/`) — library crate with:
  - `types.rs` — NodeSnapshot, ProcessInfo, ProcessFramework, DistributedBackend, ClusterEvent, EventSink
  - `scanner.rs` — Node discovery (TB bridge, Tailscale, ARP, SystemProfiler) + SSH probing
  - `collector.rs` — Parallel SSH metrics collection (powermetrics, vm_stat, ps aux, footprint)
  - `monitor.rs` — Background polling loops with shared state via Arc<RwLock<ClusterState>>
  - `aggregator.rs` — ClusterState with snapshots HashMap, scan_results, aggregates
  - `config.rs` — ClusterConfig with discovery methods, seed hosts, intervals
  - `ssh.rs` — SSH/local command execution

Data flow: Discovery finds peers -> scan_cluster probes each via SSH -> poll_metrics collects powermetrics/vmstat/ps in parallel -> ClusterState updated -> TUI renders from state + events.

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `src/main.rs` | CLI + TUI + ActivityLog | Main binary, renders dashboard, subscribes to events |
| `crates/cluster-monitor/src/scanner.rs` | Discovery + probing | Where nodes are found and probed. Concurrent discovery, hostname resolution |
| `crates/cluster-monitor/src/collector.rs` | Metrics collection | Parallel SSH commands, JACCL detection via CMD_JACCL_ENV |
| `crates/cluster-monitor/src/types.rs` | All type definitions | ProcessFramework, DistributedBackend, ClusterEvent, EventSink |
| `crates/cluster-monitor/src/monitor.rs` | Background loops | Scan loop + metrics loop, event broadcast channel |
| `crates/cluster-monitor/src/aggregator.rs` | State management | ClusterState with dedup in update_scan() |
| `crates/cluster-monitor/src/config.rs` | Configuration | Default discovery: [ThunderboltBridge, Arp] |

### Key Patterns Discovered

- **Discovery hostname resolution**: scan_node_inner runs `hostname -s` via SSH to resolve IPs to names. scan_cluster carries the peer's discovery hostname (if not an IP) and overrides the result.
- **Dedup strategy**: deduplicate_peers() merges by hostname.to_lowercase(). update_scan() also deduplicates scan results by hostname. poll_metrics uses seed_hosts exclusively when provided (avoids IP/hostname collision with discovered nodes).
- **JACCL detection**: Two-stage — (1) parse_ps_mlx checks `--backend jaccl|ring` in process args, (2) CMD_JACCL_ENV greps for `mlx.launch.*--backend` processes, then tags child processes. Only active mlx.launch processes count (env vars alone cause false positives).
- **Event system**: ClusterEvent enum + EventSink (wraps Option<broadcast::Sender>). Scanner/monitor emit events, TUI drains via try_recv() each tick.

## Work Completed

### Tasks Finished

- [x] Add real-time event system (ClusterEvent, EventSink, broadcast channel)
- [x] Replace loading spinner with live activity log + progress gauge
- [x] Parallelize discovery methods (TB, Tailscale, ARP run concurrently)
- [x] Parallelize footprint lookups (join_all instead of sequential)
- [x] Remove 1s artificial delay before metrics polling (0ms with seeds)
- [x] Skip ping for seed/known hosts (scan_node_fast)
- [x] Fix hostname resolution (scan_node_inner does `hostname -s` via SSH)
- [x] Fix dedup (peer hostname carried through, update_scan deduplicates)
- [x] Rename project: apple-smi -> mlx-top
- [x] Remove Tailscale from default discovery
- [x] Remove mlx-smi binary alias (only mlx-top + asmi now)
- [x] Fix registry save (was waiting for 3 epochs, now polls until scan_results non-empty)

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `Cargo.toml` | Renamed to mlx-top, removed mlx-smi bin | Project rename |
| `crates/cluster-monitor/Cargo.toml` | Renamed to mlx-top-core | Project rename |
| `src/main.rs` | ActivityLog struct, event subscription, progress gauge, render_activity, renamed refs | Live UX during scan |
| `src/asmi.rs` | Updated to exec mlx-top instead of apple-smi | Rename |
| `crates/cluster-monitor/src/types.rs` | Added ClusterEvent enum + EventSink | Event system |
| `crates/cluster-monitor/src/scanner.rs` | Concurrent discovery, EventSink params, hostname resolution, peer hostname override, scan_node_fast | Speed + dedup |
| `crates/cluster-monitor/src/collector.rs` | Parallel footprint lookups with join_all | Speed |
| `crates/cluster-monitor/src/monitor.rs` | broadcast channel, EventSink, events() method, removed 1s delay | Event system + speed |
| `crates/cluster-monitor/src/aggregator.rs` | Dedup in update_scan() by hostname | Fix duplicates |
| `crates/cluster-monitor/src/config.rs` | Removed Tailscale from defaults | User request |
| `crates/cluster-monitor/src/lib.rs` | Export ClusterEvent, EventSink, scan_node_fast | New public API |
| `src/mlx_smi.rs` | Deleted | No longer needed |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| Concurrent discovery | Sequential (simple) vs tokio::spawn per method | ~20s speedup, methods are independent |
| EventSink pattern | Bare broadcast::Sender vs wrapper with noop() | Clean API, no-op when nobody subscribes |
| Peer hostname override | Always use SSH hostname -s vs prefer discovery name | Discovery name matches SSH alias; hostname -s may differ from what user calls the node |
| Remove Tailscale default | Keep all defaults vs per-user config | Tailscale adds nodes from other networks, causes confusion |
| Rename to mlx-top | apple-smi, asmi, mlx-top, msmi | "mlx-top" communicates purpose (MLX + top/htop), detects MLX models |

## Pending Work

### Immediate Next Steps

1. **Fix JACCL detection in TUI** — User reports JACCL not showing in the Processes column. The infrastructure exists (DistributedBackend enum, --backend parsing in parse_ps_mlx and collector). Debug: run `ps aux | grep mlx.launch` on a node running distributed inference to see what the process looks like. Check if CMD_JACCL_ENV grep pattern matches. The collector tags child processes with the detected backend — verify this propagates to snapshots shown in TUI.

2. **Filter non-cluster nodes from ARP discovery** — ARP finds all Macs on the network (e.g., "msc-360"). Options: (a) only keep nodes that respond to SSH, (b) filter by hostname pattern, (c) only keep nodes with RDMA or TB bridge connection, (d) let user configure an allowlist. Currently scan_node already filters by SSH (ssh_ok=true), but poll_metrics still polls all ssh_ok nodes.

3. **Widen chip column or abbreviate** — Chip shows "Apple M3 U 4K..." (truncated). Strip "Apple " prefix to get "M3 Ultra" or widen Constraint::Length(16) in render_nodes.

### Blockers/Open Questions

- [ ] What does `ps aux | grep mlx.launch` output look like on a node running JACCL distributed inference? Need to verify the grep pattern matches.
- [ ] Should non-cluster ARP nodes be filtered, or should user explicitly allowlist cluster nodes?

### Deferred Items

- ROADMAP.md features (v0.2-v0.5): temperature, clocks, MLX version, CSV format, topo/pmon/dmon subcommands, Metal GPU counters, ANE utilization
- Node registry still not saving (logic was fixed but needs testing)
- Tailscale discovery available via `--scan tailscale` but not default

## Context for Resuming Agent

### Important Context

The project is at `/Users/ma/Projects/Personal/apple-smi` (directory name hasn't changed, only the Cargo package name changed to mlx-top). The user has a 4-node Apple Silicon cluster: m3u2 (M3 Ultra 512GB, primary), m3u1 (M3 Ultra 512GB), m3u3 (M3 Ultra 192GB), m4m1 (M4 Max 128GB). Connected via Thunderbolt 5 with RDMA (JACCL backend). The user frequently runs `mlx_lm.share` and `mlx.launch --backend jaccl` for distributed inference.

Node deduplication has been the #1 recurring bug across this and the previous session. The same machine appears via multiple discovery methods with different IPs (10.1.x LAN, 169.254.x TB bridge, 100.x Tailscale). The fix layers are: (1) deduplicate_peers by hostname, (2) scan_node resolves IP to hostname via SSH, (3) scan_cluster overrides result hostname with peer discovery name, (4) update_scan deduplicates results. If dupes reappear, check which layer failed.

JACCL detection works in two ways: (1) `parse_ps_mlx` in collector.rs checks `--backend jaccl` in the process command line, (2) `CMD_JACCL_ENV` greps for `mlx.launch.*--backend` and tags all child processes on the same node. False positives from env vars (`MLX_JACCL_*`) were fixed by only detecting from active mlx.launch processes.

The user's CLAUDE.md rules: never suggest models, always ask. No hardcoded hostnames. Use Tavily for web fetching. Check `/cluster-info` for fleet details.

### Assumptions Made

- All cluster nodes are reachable via SSH without password (key-based auth)
- `sudo powermetrics` works without password on all nodes (NOPASSWD in sudoers)
- `hostname -s` on each node returns a short, stable hostname
- ARP table contains hostnames for cluster nodes (not just IPs)

### Potential Gotchas

- **hostname -s mismatch**: User reported that `hostname -s` on machines may not match the SSH alias name (e.g., machine hostname is "Mas-Mac-Studio" but SSH alias is "m3u1"). The code now prefers the discovery hostname over SSH-resolved hostname when the discovery name is not an IP.
- **Chip column width**: Only 16 chars in TUI — "Apple M3 Ultra" truncates. Consider stripping "Apple " prefix.
- **Old binaries**: `apple-smi` and `mlx-smi` were removed from ~/.cargo/bin but may still exist in other PATH locations.
- **Config dir changed**: `~/.config/apple-smi/` -> `~/.config/mlx-top/`. Old registry file won't be found. May need to migrate or delete old dir.
- **43 tests all pass** as of the last build. Tests are in the core crate only.

## Environment State

### Tools/Services Used

- Rust 1.85+ (edition 2024)
- cargo install --path . --force (to install binaries)
- ratatui 0.30, crossterm 0.29, clap 4.5, tokio full features
- SSH to cluster nodes (m3u1, m3u2, m3u3, m4m1)

### Active Processes

- No active mlx-top processes (user quit the TUI)
- MLX inference may be running on cluster nodes (mlx_lm.server, mlx_lm.share)

### Environment Variables

- No special env vars required for mlx-top itself
- MLX distributed inference uses: MLX_JACCL_COORDINATOR, MLX_IBV_DEVICES, MLX_RANK, MLX_HOSTFILE, MLX_METAL_FAST_SYNCH

## Related Resources

- `/Users/ma/Projects/Personal/apple-smi/ROADMAP.md` — Feature roadmap v0.2-v0.5
- `/Users/ma/.claude/plugins/cache/local-skills/mlx-cluster/` — JACCL, RDMA, Thunderbolt troubleshooting skills
- `/tmp/mlx-top.log` — Runtime log file for debugging

---

**Security Reminder**: Before finalizing, run `validate_handoff.py` to check for accidental secret exposure.
