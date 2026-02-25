# Cluster Monitor Research

Captured from live cluster on 2026-02-21. All parsers must match these real formats.

## powermetrics (text mode)

`sudo powermetrics -n 1 -i 1000 --samplers cpu_power,gpu_power`

**JSON mode (`--json-print`) returns exit code 64 on macOS 26.x** — text-only parsing required.

### Header
```
Machine model: Mac15,14
OS version: 25A123x
```

### Cluster Format
```
E0-Cluster HW active residency:  90.12% (...)
P0-Cluster HW active residency:   0.00% (...)
```
- E-clusters = efficiency cores, P-clusters = performance cores
- M3 Ultra has: E0, P0, P1, E1, P2, P3 (6 clusters, 32 cores total)

### Per-CPU Format
```
CPU 0 frequency: 2244 MHz
CPU 0 active residency:  67.95% (...)
CPU 0 idle residency:  32.05%
```

### Power Summary (KEY LINES)
```
CPU Power: 8916 mW
GPU Power: 9462 mW
ANE Power: 0 mW
Combined Power (CPU + GPU + ANE): 18378 mW
```
**Regex:** `^(CPU|GPU|ANE) Power:\s+(\d+)\s+mW`
**Combined:** `^Combined Power.*:\s+(\d+)\s+mW`

### GPU Section
```
GPU HW active frequency: 1380 MHz
GPU HW active residency: 100.00% (338 MHz: 0% ... 1380 MHz: 100%)
GPU idle residency:   0.00%
```
**Regex for GPU%:** `^GPU HW active residency:\s+([\d.]+)%`
**Regex for GPU idle:** `^GPU idle residency:\s+([\d.]+)%`

### CPU Utilization Calculation
Average of all `CPU N active residency` values. Or use cluster-level residency.
**Regex:** `^CPU \d+ active residency:\s+([\d.]+)%`

## powermetrics JSON

Exit code 64 on macOS 26.x — **NOT AVAILABLE**. Use text parsing only.

## vm_stat

`vm_stat` on Apple Silicon M3 Ultra:

```
Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                  9563156.
Pages active:                               11058419.
Pages inactive:                             11287075.
Pages speculative:                            133882.
Pages throttled:                                    0.
Pages wired down:                            1434855.
Pages purgeable:                               89568.
```

**CRITICAL: page size = 16384 bytes (16KB) on Apple Silicon, NOT 4096!**

Parse page size from header: `\(page size of (\d+) bytes\)`
Parse page counts: `^Pages (\w[\w\s]+\w):\s+(\d+)\.`

### RAM Calculation
```
total_bytes = sysctl -n hw.memsize  (e.g., 549755813888 = 512GB)
page_size = 16384 (from vm_stat header)
free_pages = Pages free
speculative_pages = Pages speculative
used_bytes = total_bytes - (free_pages + speculative_pages) * page_size
used_gb = used_bytes / (1024^3)
total_gb = total_bytes / (1024^3)
```

## sysctl

```
549755813888          # hw.memsize (bytes) = 512GB
Apple M3 Ultra        # machdep.cpu.brand_string
```

## ps aux (MLX processes)

```
runner  62283  0.0  37.8 866456400 202951136  ??  S  12:51AM  1:35.69 .../Python -m mlx_lm.server --model /Users/runner/models/ExampleModel-8bit --port 8003 --host 0.0.0.0
```

**Fields:** user(0) pid(1) cpu%(2) mem%(3) vsz(4) rss(5) tty(6) stat(7) started(8) time(9) command(10+)

**Framework detection regex on command:**
- `mlx_lm.server` or `mlx_lm\.server` → MlxLm
- `mlx_vlm.server` or `mlx_vlm\.server` → MlxVlm
- `vllm_mlx` → VllmMlx
- `watchdog` → Watchdog (but filter out chrome-devtools-mcp watchdog and system watchdogd)

**Model extraction:** `--model\s+(\S+)` → take last path component
**Port extraction:** `--port\s+(\d+)`

**NOTE:** The `ps aux` grep also catches:
- Chrome DevTools MCP watchdog (PID 23397) — filter by `chrome-devtools-mcp`
- Custom shell script watchdog (PID 2050) — `node-watchdog.sh`
- Microsoft Teams GPU watchdog (PID 1889) — filter by `Microsoft`
- System watchdogd (PID 585) — filter by `/usr/libexec/watchdogd`

**Filter strategy:** Only match lines containing `mlx_lm` or `mlx_vlm` or `vllm_mlx`, then separately match `watchdog` lines that contain `r1o` or custom script names.

## footprint

`sudo footprint -p <PID>` output:

```
======================================================================
zsh [9002]: 64-bit    Footprint: 2272 KB (16384 bytes per page)
======================================================================

  Dirty      Clean  Reclaimable    Regions    Category
```

**Line 2 format:** `<process_name> [<pid>]: 64-bit    Footprint: <number> <unit> (<page_size> bytes per page)`
**Regex:** `Footprint:\s+([\d.]+)\s+(KB|MB|GB)`

For MLX processes, footprint is typically in GB:
```
Python [62283]: 64-bit    Footprint: 199.2 GB (16384 bytes per page)
```

Convert: KB→MB÷1024, MB→MB, GB→MB×1024

## RDMA Status

### rdma_ctl status
```
enabled
```
Simple: line is "enabled" or "disabled".

### ibv_devices
```
    device              node GUID
    ------          ----------------
    rdma_en2        d0b5e9ab23cfac05
    rdma_en3        d1b5e9ab23cfac05
    rdma_en4        d2b5e9ab23cfac05
    rdma_en5        d3b5e9ab23cfac05
    rdma_en6        d4b5e9ab23cfac05
    rdma_en7        d5b5e9ab23cfac05
```
**Regex:** `^\s+(rdma_\w+)\s+([0-9a-f]+)`

M3 Ultra (m3u2) has 6 RDMA devices (rdma_en2 through rdma_en7), one per TB port.

### ibv_devinfo
```
hca_id: rdma_en3
    transport:          Thunderbolt (100)
    ...
        port:   1
            state:          PORT_ACTIVE (4)
            ...
            link_layer:     Thunderbolt
```

**Port state regex:** `state:\s+(PORT_\w+)`
**Values:** `PORT_ACTIVE` (connected), `PORT_DOWN` (no cable or no peer)

On node-a: rdma_en3, rdma_en4, rdma_en5 are PORT_ACTIVE (3 cables connected); rdma_en2, rdma_en6, rdma_en7 are PORT_DOWN.

## ifconfig bridges

The `ifconfig | grep -E "^bridge[0-9]|inet 169.254"` command triggered a grep/rg conflict in the sandbox. Use `ifconfig` output directly.

**TB bridge format:**
```
bridge0: flags=...
    inet 169.254.x.x netmask ...
```

Each TB5 cable creates a bridge with a 169.254.x.x link-local address. These IPs change on reconnection — always discover at runtime.

## Tailscale

`tailscale status --json` returns a large JSON with peer info. Key fields:
- `.Peer` — map of peer node keys → peer objects
- Each peer: `.HostName`, `.OS`, `.Online`, `.TailscaleIPs`, `.CurAddr`

Filter: `OS == "macOS"` and `Online == true` for cluster nodes.

## Design Decisions

1. **SSH + powermetrics uniformly** — same code path for local and remote nodes. No IOKit/macmon mixing. One parser, one test surface.
2. **Text parsing only** — JSON powermetrics unavailable on macOS 26.x. All regex patterns tested against real output above.
3. **16KB page size** — hardcoded assumption for Apple Silicon is safe; all current M-series use 16KB pages. But we parse it from vm_stat header anyway for safety.
4. **footprint over RSS** — `footprint` reports real Metal GPU memory. RSS misses GPU allocations entirely. On a 200GB model: RSS=193GB, footprint=199GB.
5. **Process filtering** — grep for `mlx_lm|mlx_vlm|vllm_mlx` then filter out chrome/teams/system watchdogs by exclusion patterns.
