# asmi Reference Index

## Research
- [ANE Deep Research Summary](../../docs/research/ane-deep-research-summary.md) — canonical ANE reference (537 lines)
  - Sections 1-4: Cluster hardware, Bokhan crate, ANE constraints
  - Sections 5-6: Dual ANE on M3 Ultra, speculative decoding
  - Sections 7-10: MoE targets, model landscape, partitioning, implementation status
  - Sections 11-13: **maderix training** (6-kernel backprop, SRAM probing, private API reference)
  - Section 14: Open questions and next steps

## Implementation Plans
- [ANE Monitoring & Compute](../../docs/plans/2026-03-02-ane-monitoring-and-compute.md) — IOReport power + ObjC FFI bridge
- [TUI Simplification Cascades](../../docs/plans/2026-03-01-gut-tui-simplification-cascades.md) — completed refactor

## Session Handoffs
- [2026-03-04 Topology + Cluster](../../.claude/handoffs/2026-03-04-005942-asmi-topology-and-cluster-infra.md) — most recent
- [2026-02-21 RDMA Topology TUI](../../.claude/handoffs/2026-02-21-075344-rdma-topology-tui.md)
- [2026-02-21 Nodemap RDMA](../../.claude/handoffs/2026-02-21-070945-apple-smi-nodemap-rdma.md)
- [2026-02-21 MLX Top Dedup](../../.claude/handoffs/2026-02-21-060427-mlx-top-activity-log-dedup-jaccl.md)

## External Code References
- `/tmp/ane-bokhan/` — Bokhan Rust ANE crate (builds, GPT-2 working at 45.6 tok/s)
- `/tmp/ane-analysis/` — maderix ObjC ANE training (Stories110M, 6-kernel backprop)
  - `training/train_large.m` — main training loop
  - `training/ane_runtime.h` — private API wrapper
  - `inmem_peak.m` — peak TFLOPS measurement
  - `sram_probe.m` / `sram_bench.m` — SRAM capacity probing

## Cluster Quick Ref
| Node | Chip | RAM | ANEs | Role |
|------|------|-----|------|------|
| m3u1 | M3 Ultra | 512 GB | 2 | Compute |
| m3u2 | M3 Ultra | 512 GB | 2 | Coordinator |
| m3u3 | M3 Ultra | 256 GB | 2 | Compute |
| m4m1 | M4 Max | 128 GB | 1 | Compute |
| m4m-b-1 | M4 Max | 128 GB | 1 | Dev/Build |
| **Total** | | **1.5 TB** | **8** | **~92 TFLOPS** |

## Key APIs
- asmi daemon: `http://m3u2:9090` (axum, routes in `src/daemon.rs`)
- mlx-vlm: `m3u2:19082`, `m3u3:19082`
- ANE private API: `_ANEInMemoryModel` → compile → load → eval → unload
