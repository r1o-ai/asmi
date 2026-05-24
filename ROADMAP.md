# asmi Roadmap

## v0.1 — Core Metrics + CLI (shipped)

- [x] Single binary: `asmi`
- [x] Clap CLI: `--hosts`, `--scan`, `--format`, `--watch`, `--interval`
- [x] Smart TTY detection (table if piped, watch if interactive)
- [x] Streaming/ephemeral output (`--watch --format table|json`)
- [x] Process detection: mlx_lm.server, mlx_lm.share, mlx_vlm.server, vllm_mlx, mlx.launch
- [x] JACCL/Ring distributed backend detection
- [x] 5 discovery methods (thunderbolt, tailscale, arp, system profiler, bonjour)
- [x] Parallel SSH metrics (powermetrics, vm_stat, ps, footprint)
- [x] RDMA status monitoring
- [x] Cluster aggregates with per-node history ring buffers
- [x] NodeMap persistence (auto-discovers + saves nodes, aliases, RDMA links/IPs)

## v0.2 — HTTP Daemon (shipped)

Axum-based daemon (`--serve`) running on each node via launchd. The data layer for the entire r1o ecosystem.

- [x] **22 HTTP endpoints** on port 9090 (metrics, health, processes, models, volumes, logs, runtime, stream, serve lifecycle, JACCL, thunderbolt, ARP, cluster hub) — expanded to **26 endpoints** in v0.5
- [x] **SSE streaming** — `/stream` pushes NodeSnapshot JSON on every poll tick
- [x] **Runtime probing** — Python, MLX, vLLM, macOS versions cached at startup (`/runtime`)
- [x] **Model inventory** — `/models` scans `~/Models/`, HF cache, external volumes; 60s cache refresh
- [x] **Volume discovery** — `/volumes` enumerates mounted external drives
- [x] **Thunderbolt device tree** — `/thunderbolt` via `system_profiler`, cached 60s
- [x] **ARP topology** — `/arp` for TB link correlation across nodes
- [x] **Setup validation** — `/health/setup` runs MLX/RDMA/SSH/disk checks
- [x] **Network health** — `/health/network` validates TB service names; `/health/network/fix` auto-repairs
- [x] **Log tailing** — `/logs?name=mlx-server&lines=50` for mlx-server, mlx-vlm, vllm, asmi logs
- [x] **Cluster hub mode** — `--cluster` polls all remote nodes; `/cluster` and `/nodes` endpoints
- [x] **JACCL hostfile** — `/jaccl/config` generates hostfile from discovered RDMA topology

## v0.3 — MLX Server Lifecycle (shipped)

Rust port of mlx_daemon.py. Manages per-port MLX server subprocesses with crash recovery.

- [x] **ServeManager** — per-port process manager with state machine (idle → loading → ready → error)
- [x] **Multi-engine** — managed ports: 19080 (mlx_lm), 19082 (mlx_vlm)
- [x] **Load/stop/reload** — `/serve/load`, `/serve/stop`, `/serve/reload` with `?port=N`
- [x] **Crash recovery** — per-port state files (`~/.r1o/serve-state-{port}.json`), early crash detection
- [x] **Health polling** — detects process death, captures error from log tail
- [x] **Distributed share** — `/serve/share` starts `mlx_lm.share` with JACCL/Ring backend
- [x] **Share lifecycle** — `/serve/share/status`, `/serve/share/stop`
- [x] **Backend resolution** — "auto" resolves to jaccl (if hostfile exists) or single

## v0.4 — Daemon Management (shipped)

- [x] **`asmi daemon status`** — health check all known nodes (online/offline + uptime)
- [x] **`asmi daemon start/stop/restart`** — launchd bootstrap/bootout on target or all nodes
- [x] **`asmi daemon deploy`** — scp binary + plist to remote nodes
- [x] **`asmi daemon logs`** — tail daemon log on local or remote node

## v0.5 — Metrics Parity + Process Management (shipped)

Close remaining gaps with nvidia-smi and add process management.

- [x] **CPU cluster breakdown** — per-cluster E/P frequency, residency, and per-core detail from powermetrics
- [x] **GPU frequency** — GPU HW active frequency from powermetrics
- [x] **Disk I/O** — `GET /disk` endpoint with per-device iostat metrics (KB/t, tps, MB/s)
- [x] **Network throughput** — `GET /network` endpoint with per-interface bytes/sec and Mbps (netstat -ib delta)
- [x] **Process tree** — `GET /processes/tree` with parent-child hierarchy, CPU/mem filtering
- [x] **Process kill** — `POST /processes/:pid/kill` with signal selection, remote SSH kill, safety guards
- [ ] **Temperature** — `cpu_temp_c`/`gpu_temp_c` from IOKit SMC reads or `powermetrics --samplers smc` (`Tc0p`, `Tg0p` keys)
- [ ] **Performance state** — map powermetrics frequency/residency states to a simplified P-state indicator
- [x] ~~Version header~~ — done in v0.2 (`/runtime` endpoint)
- [x] ~~TB link speed~~ — done, shown in table output from scan results
- [x] ~~RDMA status column~~ — done, active/total shown in table

## v0.6 — Output Formats

Scripting and pipeline support matching nvidia-smi's `--query-gpu` flexibility.

- [ ] **CSV format** — `--format csv` output mode
- [ ] **Field selection** — `--query node,gpu_percent,ram_used,power,model` to select specific columns
- [ ] **No-header/no-units** — `--no-header` and `--no-units` flags for csv/table
- [ ] **`--list-nodes`** — dump scan results as a simple list (like `nvidia-smi -L`): hostname, chip, RAM, GPU cores, RDMA status
- [ ] **Verbose query** — `--format verbose` dumping all collected data per node in structured text (like `nvidia-smi -q`)
- [ ] **`--filename`** — native file output flag (currently works via pipe)

## v0.7 — Monitoring Subcommands

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

## v0.8 — Storage & Device Discovery

Full cluster device inventory — not just compute nodes, but all attached storage, docks, and NAS devices.

- [x] ~~Volume discovery~~ — done in v0.2 (`/volumes` endpoint)
- [x] ~~Model location awareness~~ — done in v0.2 (`/models` with volume + storage tier)
- [x] ~~Dock topology~~ — done in v0.2 (`/thunderbolt` parses SPThunderboltDataType downstream devices)
- [ ] **Storage tier display** — show hot (SSD), warm (NAS), cold (archive) tiers per node in table output
- [ ] **NAS health** — poll NAS usage, active transfer status, SMART data
- [ ] **Network storage** — discover NAS via `dns-sd -B _smb._tcp` + probe for health APIs
- [ ] **`asmi storage`** — subcommand showing per-node storage summary: volumes, capacity, usage, tier, models stored
- [ ] **Transfer monitoring** — detect active rsync/scp processes and show progress (source, dest, speed, ETA)

## v0.9 — Metal Integration

GPU-level metrics via Metal APIs and IOKit for deeper profiling.

- [ ] **Metal performance counters** — read GPU counters via `MTLDevice.counterSets` (shader utilization, occupancy, bandwidth)
- [ ] **Per-process GPU time** — investigate IOKit GPU time accounting for running processes
- [ ] **ANE utilization** — extend v0.10 ANE integration with Metal Performance Counters correlation
- [ ] **Memory bandwidth** — GPU memory bandwidth utilization via Metal counters or `powermetrics`
- [ ] **Shader core occupancy** — thread execution width, active warps equivalent
- [ ] **GPU power breakdown** — separate GPU cluster power if available from Metal 4 APIs

## v0.10 — ANE Integration (in progress)

Direct Apple Neural Engine compute via private APIs + IOSurface I/O.

- [x] **ANE power (sudoless)** — IOReport `"Energy Model"` channel for ANE power without sudo
- [x] **GET /ane** — dedicated ANE metrics endpoint (power, active status, IOReport source)
- [x] **`ProcessFramework::AneNative`** — process detection for ANE workloads
- [x] **`ane-runtime` crate** — Rust FFI wrappers for `_ANEInMemoryModel` lifecycle (8000 LOC)
- [x] **GPT-2 forward pass** — working ANE inference at 31.1 tok/s (M3 Ultra)
- [ ] **`--experimental-ane`** — CLI flag + feature gate for ANE compute endpoints
- [ ] **GET /ane/compute** — ANE compute subsystem status (compile budget, availability)
- [ ] **POST /ane/eval** — submit MIL program to ANE via HTTP (scaffolded)
- [ ] **GET /ane/probe** — IOSurface memory layout profiling for RDMA research
- [ ] **ANE-RDMA bridge** — cross-node activation transfer via Thunderbolt 5 RDMA
- [ ] **Pipeline parallelism** — distribute transformer layers across nodes
- [ ] **MoE expert placement** — route MoE experts to individual ANE dies across cluster

## Feature Mapping Reference

Coverage of nvidia-smi features (35 applicable, excluding 11 N/A):

| Status | Count | % |
|---|---|---|
| Implemented | 27 | 77% |
| Partial | 5 | 14% |
| Feasible (planned) | 3 | 9% |

Features asmi has that nvidia-smi lacks:
- Cluster-native multi-node monitoring (nvidia-smi is single-host only)
- Dynamic node discovery (5 methods)
- ML framework-aware process detection (model name, serving port)
- Distributed inference tracking (JACCL/Ring)
- HTTP daemon with SSE streaming (programmatic access)
- MLX server lifecycle management (load/stop/reload/share)
- JACCL hostfile generation from discovered RDMA topology
- Thunderbolt device tree + ARP correlation
- Crash recovery with persistent state files
- Cluster-wide daemon management (deploy, start, stop across nodes)
- Smart TTY detection for output format

Features that don't apply (N/A):
- Persistence mode, compute mode, ECC toggle, MIG, clock locking, power limit setting, fan speed, PCI bus ID, VBIOS — none of these exist on Apple Silicon
