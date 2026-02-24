# asmi Roadmap

## v0.1 — Initial Release (shipped)

- Primary binary: `asmi` (also installs `mlx-top` alias for backward compatibility)
- Clap CLI: `--hosts`, `--scan`, `--format`, `--watch`, `--interval`
- Smart TTY detection (tui if interactive, table if piped)
- Streaming/ephemeral output (`--watch --format table|json`)
- Process detection: mlx_lm.server, mlx_lm.share, mlx_vlm.server, vllm_mlx, mlx.launch
- JACCL/Ring distributed backend detection
- 5 discovery methods (thunderbolt, tailscale, arp, system profiler, bonjour)
- Parallel SSH metrics (powermetrics, vm_stat, ps, footprint)
- RDMA status monitoring
- Cluster aggregates with per-node history ring buffers
- 43 tests with real captured testdata

## v0.2 — Metrics Parity

Close the remaining gaps with nvidia-smi's core metrics.

- [ ] **Temperature** — populate `cpu_temp_c`/`gpu_temp_c` from IOKit SMC reads or `powermetrics --samplers smc` (`Tc0p`, `Tg0p` keys)
- [ ] **Clock frequencies** — GPU and CPU cluster frequencies from powermetrics DVFS section
- [ ] **Version header** — MLX version (`python -c "import mlx; print(mlx.__version__)"` via SSH) + macOS version (`sw_vers`) in TUI header and table output
- [ ] **TB link speed** — show Thunderbolt link speed from `system_profiler` in TUI node table (data already in `DiscoveredPeer.link_speed`)
- [ ] **RDMA status column** — show RDMA active device count per node in TUI (data already in `ScanResult.rdma`)
- [ ] **Performance state** — map powermetrics frequency/residency states to a simplified P-state indicator

## v0.3 — Output Formats

Scripting and pipeline support matching nvidia-smi's `--query-gpu` flexibility.

- [ ] **CSV format** — `--format csv` output mode
- [ ] **Field selection** — `--query node,gpu_percent,ram_used,power,model` to select specific columns
- [ ] **No-header/no-units** — `--no-header` and `--no-units` flags for csv/table
- [ ] **`--list-nodes`** — dump scan results as a simple list (like `nvidia-smi -L`): hostname, chip, RAM, GPU cores, RDMA status
- [ ] **Verbose query** — `--format verbose` dumping all collected data per node in structured text (like `nvidia-smi -q`)
- [ ] **`--filename`** — native file output flag (currently works via pipe)

## v0.4 — Subcommands

Dedicated monitoring views matching nvidia-smi's `dmon`, `pmon`, `topo`.

- [ ] **`asmi topo`** — NxN node connectivity matrix showing interconnect type per pair:
  - `TB5` = direct Thunderbolt 5 (120 Gbps)
  - `TB5-D` = daisy-chained Thunderbolt
  - `TS` = Tailscale (WireGuard overlay)
  - `LAN` = local network (1 Gbps ethernet)
  - `RDMA` = JACCL RDMA active on link
  - Display CPU affinity equivalent (chip type, core count)
- [ ] **`asmi pmon`** — per-process rolling monitor across all nodes:
  - Columns: node, pid, framework, model, cpu%, mem%, footprint, port, distributed
  - Scrolling output like `nvidia-smi pmon`
  - `-d` delay, `-c` count flags
- [ ] **`asmi dmon`** — per-node rolling device monitor:
  - Selectable metric groups (`-s`): p=power/temp, u=utilization, m=memory, r=rdma, t=throughput
  - Columns: node, power, temp, cpu%, gpu%, ram, tb-rx, tb-tx
- [ ] **TB throughput** — rx/tx bytes/sec from `netstat -I enN` or `nettop` on bridge interfaces

## v0.5 — Metal Integration

GPU-level metrics via Metal APIs and IOKit for deeper profiling.

- [ ] **Metal performance counters** — read GPU counters via `MTLDevice.counterSets` (shader utilization, occupancy, bandwidth)
- [ ] **Per-process GPU time** — investigate IOKit GPU time accounting for running processes
- [ ] **ANE utilization** — Apple Neural Engine utilization % (may require private API or `powermetrics` ANE sampler)
- [ ] **Memory bandwidth** — GPU memory bandwidth utilization via Metal counters or `powermetrics`
- [ ] **Shader core occupancy** — thread execution width, active warps equivalent
- [ ] **GPU power breakdown** — separate GPU cluster power if available from Metal 4 APIs

## Feature Mapping Reference

Coverage of nvidia-smi features (35 applicable, excluding 11 N/A):

| Status | Count | % |
|---|---|---|
| Implemented | 21 | 60% |
| Partial | 8 | 23% |
| Feasible (planned) | 6 | 17% |

Features asmi has that nvidia-smi lacks:
- Cluster-native multi-node monitoring (nvidia-smi is single-host only)
- Dynamic node discovery (5 methods)
- ML framework-aware process detection (model name, serving port)
- Distributed inference tracking (JACCL/Ring)
- Interactive TUI with keyboard navigation
- Smart TTY detection for output format

Features that don't apply (N/A):
- Persistence mode, compute mode, ECC toggle, MIG, clock locking, power limit setting, fan speed, PCI bus ID, VBIOS — none of these exist on Apple Silicon
