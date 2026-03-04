# asmi — Apple Silicon Machine Intelligence

## Project Overview
Rust CLI + HTTP daemon for Apple Silicon cluster monitoring and ANE inference orchestration.
5 nodes, 1.5TB unified RAM, 8 ANE dies (~92 TFLOPS), TB5/RDMA mesh.

## Build & Test
```bash
cargo build                    # debug build
cargo build --release          # release build
cargo test                     # all tests
cargo test --lib               # unit tests only
cargo clippy                   # lint
```

## Architecture
- `src/main.rs` — CLI entrypoint (clap subcommands: Daemon, Topology)
- `src/daemon.rs` — HTTP route handlers (axum)
- `src/daemon_startup.rs` — daemon boot, cache loops, route wiring
- `src/topology.rs` — TB5/RDMA mesh discovery via mlx.distributed_config
- `crates/cluster-monitor/` — shared types: NodeMap, collectors, SSH, scanner

## Session Workflow (Harness Pattern)
At the start of every session:
1. Read `.claude/harness/progress.txt` and `git log --oneline -10`
2. Read `.claude/harness/features.json` — pick highest-priority pending feature
3. Work on ONE feature at a time

At the end of every session:
1. Update `.claude/harness/progress.txt` with: what was done, what failed, decisions, what's next
2. Update feature status in `.claude/harness/features.json`
3. Create git commit if meaningful work completed

## Key Research
- `docs/research/ane-deep-research-summary.md` — canonical ANE reference
- `docs/plans/` — implementation plans
- `.claude/handoffs/` — session handoff docs

## Conventions
- HTTP handlers: `async fn handler(State(state): State<Arc<AppState>>) -> Result<Json<T>, ApiError>`
- Background caches: `Arc<RwLock<Option<(Value, Instant)>>>` with TTL
- Feature gates: `#[cfg(feature = "ane")]` for experimental ANE code
- Error types: typed enums, not string errors
