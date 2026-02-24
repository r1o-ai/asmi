# Handoff: RDMA Topology Mapping, Config Persistence, TUI RDMA Column

## Session Metadata
- Created: 2026-02-21 07:53:44
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

- **Continues from**: [2026-02-21-070945-apple-smi-nodemap-rdma.md](./2026-02-21-070945-apple-smi-nodemap-rdma.md)
  - Previous title: apple-smi -- NodeMap, RDMA IP Discovery, Hierarchical Scanning
- **Supersedes**: None

## Current State Summary

This session added RDMA topology mapping to `mlx-top` -- the Rust TUI cluster monitor for Apple Silicon. We built the infrastructure to map which local Thunderbolt interface connects to which remote node via RDMA, including device name correlation (`rdma_en3` -> `en3` -> peer m3u3). Config persistence was fixed (it was silently saving to `~/Library/Application Support/` instead of `~/.config/`). A new RDMA column was added to the TUI table showing link state per node. The progress bar now has a braille spinner. 51 tests pass (up from 46). Binary is installed as `mlx-top` and `asmi`.

**LEFT OFF**: User wants to see a **full RDMA link topology view** -- which node links to which via which interface. The current RDMA column shows links TO a node (e.g., `en3↑ en5↑` for m3u1), but doesn't show a full mesh/matrix view. The user said "i need to be able to see which links to which" right before requesting this handoff.

## Codebase Understanding

### Architecture Overview

`mlx-top` is a Rust CLI/TUI for monitoring Apple Silicon clusters. It uses SSH to probe remote nodes for metrics (powermetrics, ps, sysctl, vm_stat) and discovers nodes via TB bridge scanning, ARP, Tailscale, and system_profiler. The core library (`mlx-top-core` in `crates/cluster-monitor/`) is separate from the TUI binary (`src/main.rs`).

Key data flow: `ClusterMonitor::start()` spawns two loops:
1. **Scan loop** -- discovers nodes, probes hardware/RDMA, emits events
2. **Metrics loop** -- polls CPU/GPU/RAM/power from known nodes

Events flow through a `tokio::sync::broadcast` channel. A background task in main.rs handles events to persist NodeMap (aliases, nodes, RDMA links) to `~/.config/mlx-top/config.json`.

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `src/main.rs` | TUI binary, CLI args, render functions, event handler | Main entry point, all UI rendering |
| `crates/cluster-monitor/src/scanner.rs` | Node discovery + scanning | TB bridge discovery, RDMA scanning, `DiscoveredPeer` |
| `crates/cluster-monitor/src/types.rs` | Core types | `RdmaLink`, `RdmaDevice`, `PortState`, `ClusterEvent`, `ScanResult` |
| `crates/cluster-monitor/src/config.rs` | `ClusterConfig` + `NodeMap` | Config persistence, JACCL hostfile gen, alias resolution |
| `crates/cluster-monitor/src/monitor.rs` | Background monitor | Spawns scan + metrics loops, shared state |
| `crates/cluster-monitor/src/aggregator.rs` | `ClusterState` | Aggregates snapshots, histories, scan results |
| `crates/cluster-monitor/src/collector.rs` | Metrics collection | powermetrics parsing, ps parsing, footprint |
| `crates/cluster-monitor/src/lib.rs` | Public exports | Re-exports from all modules |

### Key Patterns Discovered

- **Event-driven persistence**: `ClusterEvent` variants emitted during scan, handled in main.rs to update `NodeMap` and save to disk. BUT events can race with shutdown -- the one-shot mode now does a direct save after scan rather than relying on async events.
- **Broadcast subscriber timing**: Must subscribe to `monitor.events()` BEFORE calling `monitor.start()`, otherwise early events are lost (broadcast channels don't buffer for late subscribers).
- **RDMA device naming**: `rdma_enX` maps directly to interface `enX`. Parse the device name to get the interface.
- **TB bridge detection**: Not all TB interfaces have 169.254 local IPs. en4 had 192.168.60.1 but still carried RDMA traffic. Fixed by scanning ARP for any `en*` interface with 169.254 peers.
- **macOS config path**: `dirs::config_dir()` returns `~/Library/Application Support/` on macOS, NOT `~/.config/`. Fixed to use `dirs::home_dir().join(".config")` for CLI consistency.

## Work Completed

### Tasks Finished

- [x] Fix NodeMap config persistence (register_node method, NodeProbed event handling)
- [x] Add RdmaLink type with local_interface, local_ip, remote_ip, remote_hostname, rdma_device, port_state
- [x] Add RdmaLinkDiscovered + RdmaDeviceCorrelated events
- [x] Fix en4 blind spot (parse_ifconfig_all_ips, peer-driven ARP scanning)
- [x] Add RDMA device name + port state correlation to RdmaLink
- [x] Generate JACCL hostfile with NxN RDMA device matrix (hostfile_jaccl method)
- [x] Add RDMA column to TUI node table with state indicators (↑↓—)
- [x] Fix config path from ~/Library/Application Support/ to ~/.config/
- [x] Add braille spinner to progress bar
- [x] Fix broadcast subscriber timing (subscribe before start)
- [x] Direct config save in one-shot mode (bypass async event race)

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `crates/cluster-monitor/src/types.rs` | Added `RdmaLink` struct, `RdmaLinkDiscovered`/`RdmaDeviceCorrelated` events, `Display` impl for `RdmaLink` | RDMA topology mapping |
| `crates/cluster-monitor/src/config.rs` | Added `register_node()`, `add_rdma_link()`, `rdma_links_to()`, `hostfile_jaccl()`, `rdma_links` field on NodeMap, fixed config_path to ~/.config, added tests | Config persistence + JACCL hostfile |
| `crates/cluster-monitor/src/scanner.rs` | Added `local_ip` to `DiscoveredPeer`, `parse_ifconfig_all_ips()`, peer-driven TB bridge discovery, post-scan RDMA device correlation, `RdmaLinkDiscovered` emission | en4 fix + RDMA topology |
| `crates/cluster-monitor/src/lib.rs` | Export new types/functions | Public API |
| `src/main.rs` | RDMA column in TUI, event handlers for new events, spinner, direct config save, subscribe-before-start fix | TUI display + persistence |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| Store RDMA links in NodeMap (not ScanResult) | NodeMap vs ScanResult vs separate file | NodeMap already persists to disk, links are topology data that should survive restarts |
| Peer-driven TB bridge discovery | Only detect interfaces with 169.254 local IP vs scan ARP for any en* with 169.254 peers | en4 had 192.168 local IP but still carried TB RDMA traffic to m3u1 |
| Direct save in one-shot mode | Event-driven save vs direct save after scan | Events race with shutdown -- broadcast handler may not process before tokio drops tasks |
| RDMA column indicators: ↑ ↓ — | Text labels vs unicode arrows vs color-only | Compact, visually distinct, colorblind-friendly with color backup |
| XDG-style ~/.config path | dirs::config_dir (~/Library/Application Support) vs ~/.config | CLI tools conventionally use ~/.config on all platforms |
| JACCL hostfile uses NxN matrix | Flat IP list vs NxN device matrix | JACCL requires the rdma[i]=device format per the jaccl skill docs |

## Pending Work

### Immediate Next Steps

1. **Full RDMA mesh/topology view** -- User wants to see "which links to which". Options:
   - A dedicated RDMA panel/tab in the TUI showing the full NxN mesh
   - Or expand the RDMA column to show peer names (e.g., "en3↑m3u3 en5↑m3u1")
   - Consider a detail view when a node row is selected (press Enter to expand)
2. **Populate RDMA links during TUI mode** -- Currently links only populate during scan_cluster (TB bridge discovery). In TUI mode with `--hosts`, seeds don't trigger TB discovery unless `--scan thunderbolt` is also passed. Need to ensure RDMA links populate from the full scan, not just seeds.
3. **Install TDD skill** -- User asked to find TDD skills. Top candidates found:
   - `pproenca/dot-skills@rust-testing` (44 installs)
   - `mikeyobrien/ralph-orchestrator@test-driven-development` (8 installs)
   - Not yet installed -- user didn't confirm which.

### Blockers/Open Questions

- The 3-node JACCL kernel panic (AppleThunderboltRDMA 0.0.1 driver bug) means we can only test RDMA with 2 nodes. Ring backend works for 3+.
- JACCL hostfile generator (`hostfile_jaccl`) only knows local node's links. For the full NxN matrix with 3+ nodes, each node would need to report its own links back. Currently we only have the local perspective.
- Ring hostfile needs 192.168.0.x alias IPs (set via `ifconfig alias`), not 169.254 link-local IPs. These alias IPs aren't captured anywhere yet.

### Deferred Items

- Ring hostfile generator (needs 192.168.0.x IPs that aren't tracked yet)
- Bonjour discovery (DiscoveryMethod::Bonjour marked TODO)
- TDD skill installation (found candidates, user hasn't chosen)

## Context for Resuming Agent

### Important Context

The RDMA topology is a **per-node view from the local machine**. Each node only knows its own TB interfaces and which peers it sees on them. The `NodeMap.rdma_links` stores links FROM the local node TO remote peers. To build a full cluster mesh view, we'd need each node to report its links (not implemented -- would require collecting link data via SSH from each node during scan).

The RDMA column currently shows links TO a given node. For m3u1 it might show `en3↑ en5↑` meaning two local interfaces connect to m3u1, both with RDMA active. For m4m1 it shows `en22—` meaning a TB link exists on en22 but no RDMA device (`rdma_en22` doesn't exist, only en2-en7 have RDMA devices).

**192.168 warning**: If a link has a 192.168 local IP (like en4), the RDMA column shows yellow with `!` suffix. This indicates the interface doesn't have a proper 169.254 link-local address and may need cable re-seating.

The `hostfile_jaccl()` method generates a 2-node JACCL hostfile correctly (tested). For 3+ nodes it fills what it knows from local links and puts `null` for unknown inter-node connections.

### Assumptions Made

- RDMA device names always follow the pattern `rdma_enX` where `enX` is the interface name
- Only en* interfaces carry Thunderbolt traffic (bridge* interfaces are VM bridges)
- ARP entries marked "permanent" are local (self) addresses
- The local node runs `rdma_ctl status` and `ibv_devinfo` locally in the post-scan step

### Potential Gotchas

- **Broadcast event timing**: Subscribe to `monitor.events()` BEFORE `monitor.start()`. Events are lost otherwise.
- **Config path was wrong**: Previously saved to `~/Library/Application Support/mlx-top/`, now correctly saves to `~/.config/mlx-top/config.json`. Old configs at the Library path are orphaned.
- **One-shot mode race**: The async event handler may not process all events before `monitor.stop()` kills tasks. That's why we do a direct save of scan results in one-shot mode.
- **en4 has non-169.254 local IP**: The old code missed TB interfaces without 169.254 local addresses. Fixed with `parse_ifconfig_all_ips()` and peer-driven ARP scanning.
- **Test count**: 51 tests (46 original + 5 new: parse_ifconfig_all_ips, register_node, add_rdma_link, hostfile_jaccl_2node, hostfile_jaccl_empty)

## Environment State

### Tools/Services Used

- Rust toolchain (cargo build/test/install)
- `mlx-top` / `asmi` binaries installed to `~/.cargo/bin/`
- SSH for remote node probing (passwordless keys required)
- RDMA tools: `rdma_ctl`, `ibv_devices`, `ibv_devinfo` (macOS 26.2+)
- Sequential Thinking MCP for analysis
- JACCL and RDMA skills consulted for audit

### Active Processes

- `mlx_lm.server` running on m3u2 port 8003, model MiniMax-M2.5-REAP-19-8bit
- Log file at `/tmp/mlx-top.log`

### Environment Variables

- No env vars required for mlx-top itself
- MLX distributed uses: MLX_METAL_FAST_SYNCH, MLX_JACCL_COORDINATOR, MLX_JACCL_DEVICES, MLX_RANK, MLX_HOSTFILE

## Related Resources

- JACCL skill: `/Users/ma/.claude/plugins/cache/local-skills/mlx-cluster/1.0.0/skills/jaccl`
- RDMA skill: `/Users/ma/.claude/plugins/cache/local-skills/mlx-cluster/1.0.0/skills/rdma`
- Config file: `~/.config/mlx-top/config.json`
- Test data: `crates/cluster-monitor/testdata/` (arp-table.txt, ifconfig-bridges.txt, rdma-status.txt, etc.)

---

**Security Reminder**: Before finalizing, run `validate_handoff.py` to check for accidental secret exposure.
