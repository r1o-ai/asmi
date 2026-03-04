# Handoff: asmi Topology Command + Cluster Infrastructure Cleanup

## Session Metadata
- Created: 2026-03-04 00:59:42
- Project: /Users/ma/Projects/Personal/apple-smi
- Branch: main
- Session duration: ~2 hours (across context boundary)

### Recent Commits (for context)
  - 9666be1 feat(serve): VLM lazy-load via warmup, IOReport telemetry, watchdog
  - b4d1aa0 feat(serve): allow idle server start without model_path
  - d115892 refactor(asmi): gut TUI, split modules, unify process managers, typed errors, static regexes

## Handoff Chain

- **Continues from**: [2026-02-21-075344-rdma-topology-tui.md](./2026-02-21-075344-rdma-topology-tui.md)
  - Previous title: RDMA Topology Mapping, Config Persistence, TUI RDMA Column
- **Supersedes**: None

## Current State Summary

Implemented `asmi topology` — a new CLI subcommand that wraps Apple's `mlx.distributed_config --dot` to discover the full TB5/RDMA mesh topology across all cluster nodes. The command parses DOT output into structured JSON, validates mesh completeness for JACCL, and identifies working 3-node subsets when the full mesh is incomplete. The binary is built, tested (4 unit tests passing), and installed to both `~/.cargo/bin/asmi` and `~/.local/bin/asmi`. Also cleaned up stale config data (removed mini2/m4m/mini1 references, renamed to m4mini) and fixed factual errors in the asmi SKILL.md. The session also diagnosed and stopped a broken OpenClaw gateway that was in an error feedback loop, and wrote a lazy-load MLX proxy script. Phase 3 (HTTP daemon `/topology` endpoint) is next.

## Codebase Understanding

### Architecture Overview

apple-smi is a Rust workspace with two crates:
- **asmi** (binary, `src/`) — CLI entrypoint, daemon startup, HTTP serve routes, topology
- **asmi-core** (`crates/cluster-monitor/`) — shared types: NodeMap, collectors, SSH, scanner, health, models

The CLI uses clap with subcommands (`Daemon`, `Topology`). The HTTP daemon (axum) runs via `--serve --port 9090` and optionally polls remote nodes with `--cluster`. Topology wraps an external Python tool rather than reimplementing.

### Critical Files

| File | Purpose | Relevance |
|------|---------|-----------|
| `src/topology.rs` | NEW — mesh discovery via mlx.distributed_config wrapper | Core deliverable |
| `src/main.rs` | CLI entrypoint, subcommand dispatch | Added `Topology` variant |
| `src/daemon.rs` | HTTP route handlers (axum) | Needs `/topology` endpoints (Phase 3) |
| `src/daemon_startup.rs` | Daemon boot, cache setup, route wiring | Needs topology cache + route registration |
| `crates/cluster-monitor/src/config.rs` | NodeMap, RdmaLink, hostfile_jaccl() | Stale rdma_links — topology should replace |
| `~/.config/asmi/config.json` | Persistent node map, aliases, RDMA links | Cleaned up this session |

### Key Patterns Discovered

- HTTP handlers in `daemon.rs` follow: `async fn handler(State(state): State<Arc<AppState>>) -> Result<Json<T>, ApiError>`
- Background cache loops in `daemon_startup.rs` use `Arc<RwLock<Option<(Value, Instant)>>>` with configurable TTL
- `mlx.distributed_config` outputs DOT to stdout, errors to stderr, and exits non-zero on incomplete mesh but STILL outputs the partial DOT — so we parse whatever DOT we get
- JACCL hostfile format: `[{"ssh": "hostname", "ips": [], "rdma": [null, "rdma_en3", ...]}]` where `rdma[i]` = device connecting to node `i`

## Work Completed

### Tasks Finished

- [x] Phase 1: Cleaned `~/.config/asmi/config.json` — removed mini2/m4m/mini1, added m4mini
- [x] Phase 2: Implemented `asmi topology` CLI subcommand with table/json/dot output
- [x] Phase 5: Fixed asmi SKILL.md — m3u2 RAM 192→512GB, total ~1.1→1,408GB, added topology endpoints, added m4mini
- [x] Phase 6 partial: Fixed `~/hostfile-jaccl-4node.json` — mini2→m4m1
- [x] Diagnosed and killed broken OpenClaw gateway (error feedback loop)
- [x] Wrote MLX lazy-load proxy with pressure-based eviction (`~/.openclaw/scripts/mlx-lazy-proxy.py`)

### Files Modified

| File | Changes | Rationale |
|------|---------|-----------|
| `src/topology.rs` | NEW: 280 lines — DOT parser, mesh validator, table formatter | Core topology feature |
| `src/main.rs` | Added `Topology` subcommand, `TopologyFormat` enum, dispatch | Wire topology into CLI |
| `~/.config/asmi/config.json` | Removed stale nodes/aliases, cleaned rdma_links | Config had mini2/m4m references |
| `~/.claude/skills/asmi/SKILL.md` | Fixed RAM values, added topology endpoints/tasks, added m4mini | Factual corrections |
| `~/hostfile-jaccl-4node.json` | Changed `"ssh": "mini2"` → `"ssh": "m4m1"` | Wrong node in JACCL hostfile |
| `~/.openclaw/scripts/mlx-lazy-proxy.py` | NEW: lazy-load HTTP proxy with asmi pressure eviction | OpenClaw model loading |

### Decisions Made

| Decision | Options Considered | Rationale |
|----------|-------------------|-----------|
| Wrap mlx.distributed_config vs reimplement | Wrap (subprocess) vs native Rust SSH discovery | Apple maintains TB5 discovery; avoids duplicating changing macOS internals |
| DOT parsing approach | Regex vs line-by-line string parsing | Simple line parsing works for mlx.distributed_config's consistent output |
| Topology as subcommand vs flag | `asmi topology` vs `asmi --topology` | Subcommand is cleaner — different output format, different purpose than metrics |
| Node naming: mini2 → m4mini | m4mini, relay, gateway | m4mini follows the chip-based naming scheme (m{gen}{variant}) |
| OpenClaw model eviction | Time-based idle timeout vs pressure-based | Pressure-based (>50% cluster RAM) is smarter — keeps model warm when cluster is light |

## Pending Work

### Immediate Next Steps

1. **Phase 3: Add `/topology` HTTP endpoints to daemon** — `GET /topology` (JSON), `GET /topology/dot`, `GET /topology/validate`. Add to `daemon.rs` handlers and `daemon_startup.rs` route registration. Use 60s cache like thunderbolt_cache pattern.
2. **Phase 4: Update `/jaccl/config` to use live topology** — Replace stale `rdma_links` lookup in `jaccl_config_handler` with topology discovery. Return proper error on incomplete mesh.
3. **Physical: Add m3u1↔m3u3 TB5 cable** — Both nodes have 3 empty ports. This is the missing 6th link for full 4-node JACCL.
4. **Commit the topology.rs changes** — `src/topology.rs` and `src/main.rs` changes are uncommitted.
5. **Regenerate JACCL hostfile** — Once topology is wired into `/jaccl/config`, use `POST /jaccl/config` to generate accurate hostfile from live RDMA data.
6. **Re-enable OpenClaw gateway** — Currently disabled (`launchctl disable gui/501/ai.openclaw.gateway`). Needs working model server first. Lazy proxy at `~/.openclaw/scripts/mlx-lazy-proxy.py` is ready but untested.

### Blockers/Open Questions

- [ ] The JACCL hostfile `rdma_en2` entry for m3u2↔m4m1 may be stale — m3u2 rdma_en2 shows Down in asmi. The actual link is m3u2 rdma_en5↔m4m1 rdma_en3 (confirmed by `asmi topology`). Hostfile needs full regeneration.
- [ ] m3u1 Port 3 shows Link Status `0x7` (half state) — could be a loose cable. Worth checking physically.
- [ ] OpenClaw's `openclaw.json` model config points to `127.0.0.1:8090` but no MLX server runs on that port. The mlx-vlm servers are on `:19082` on m3u2 and m3u3. The lazy proxy would bridge this gap.

### Deferred Items

- MLX research agent was launched (Tavily searches for mlx.launch, JACCL, mlx_lm commands) — results may be in `/private/tmp/claude-501/-Users-ma/tasks/a806ef313f0b9a98d.output` but were never fully consumed
- Cluster skill (`mlx-cluster:cluster`) rewrite — audited at 42/100, needs to be rebuilt around asmi
- Duplicate plugin copies: `~/.claude/plugins/cache/local-skills/mlx-cluster/` and `~/.claude/local-marketplace/mlx-cluster/` — need consolidation
- `mlx_lm` syntax changes: `python -m mlx_lm.generate` is deprecated → use `mlx_lm.generate` or `python -m mlx_lm generate`

## Context for Resuming Agent

### Important Context

The cluster has 4 nodes: m3u2 (512GB M3 Ultra, coordinator), m3u1 (512GB M3 Ultra), m3u3 (256GB M3 Ultra), m4m1 (128GB M4 Max). There is also m4mini (16GB M4 base Mac Mini) which is NOT a cluster node. The TB5 mesh has 5 of 6 links — missing m3u1↔m3u3. `asmi topology` now discovers this live by wrapping `mlx.distributed_config --dot`. The topology module (`src/topology.rs`) parses DOT into `TopologyReport` with links, missing links, JACCL readiness, and valid subsets. The daemon endpoints pattern uses axum with `State<Arc<AppState>>` and background cache loops with `Arc<RwLock<Option<(Value, Instant)>>>`. To add `/topology` endpoints, follow the `thunderbolt_cache` pattern in `daemon_startup.rs`.

### Assumptions Made

- `mlx.distributed_config` is stable and its DOT output format won't change significantly
- The RDMA device names in DOT labels (e.g., `en3/en4`) map directly to `rdma_enX` kernel devices
- 60s cache TTL is appropriate for topology (physical cables don't change often)
- The existing `rdma_links` field in config.json can be replaced by live topology — no downstream consumers depend on it

### Potential Gotchas

- `mlx.distributed_config` exits non-zero on incomplete mesh but STILL outputs partial DOT to stdout — the parser handles this
- `mlx.distributed_config` is a Python script (`/opt/homebrew/bin/mlx.distributed_config`) installed via pip under python3.13, but asmi is built with the system Python 3.14 — shouldn't matter since we subprocess it
- The hostfile has m4m1 as SSH target but RDMA devices may be wrong (rdma_en2 was mini2's mapping). Full regen needed via topology.
- OpenClaw gateway is disabled via launchctl. To re-enable: `launchctl enable gui/501/ai.openclaw.gateway && launchctl kickstart gui/501/ai.openclaw.gateway`

## Environment State

### Tools/Services Used

- `asmi` CLI — installed at `~/.local/bin/asmi` and `~/.cargo/bin/asmi` (release build)
- `mlx.distributed_config` — at `/opt/homebrew/bin/mlx.distributed_config`
- asmi daemon — running on m3u2:9090 (but serving stale topology from old config)

### Active Processes

- asmi daemon on m3u2:9090 (needs restart to pick up new config, but doesn't have /topology endpoints yet)
- mlx-vlm on m3u2:19082 (400MB, Qwen3.5-35B-A3B)
- mlx-vlm on m3u3:19082 (32GB, Qwen3.5-35B-A3B)
- OpenClaw gateway is DISABLED (killed and launchctl disabled)

### Environment Variables

- None specific — asmi uses XDG_CONFIG_HOME for config path (defaults to ~/.config/asmi/)

## Related Resources

- Implementation plan: `~/.claude/plans/asmi-topology-enhancement.md`
- asmi skill: `~/.claude/skills/asmi/SKILL.md`
- apple-smi source: `~/Projects/Personal/apple-smi/`
- Lazy proxy: `~/.openclaw/scripts/mlx-lazy-proxy.py`
- OpenClaw config: `~/.openclaw/openclaw.json`
- Cluster topology reference: `~/.claude/plugins/cache/local-skills/mlx-cluster/1.0.0/references/topology.md`

---

**Security Reminder**: Before finalizing, run `validate_handoff.py` to check for accidental secret exposure.
