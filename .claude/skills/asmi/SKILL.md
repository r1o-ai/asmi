---
name: asmi
description: >
  Apple Silicon Machine Intelligence — cluster monitoring CLI and HTTP daemon.
  Query CPU/GPU/RAM/power/RDMA metrics, discover nodes, check cluster health,
  inspect processes, and generate JACCL hostfiles. Use when: "asmi", "cluster status",
  "node metrics", "rdma status", "what's running on m3u1", "gpu usage", "power draw",
  "cluster health", "jaccl hostfile", "thunderbolt topology", "node discovery",
  "which nodes are online", "check cluster", "asmi json", "asmi watch".
argument-hint: "[--hosts node1,node2] [--format json|table] [--watch]"
---

# asmi — Apple Silicon Machine Intelligence

> Like `nvidia-smi` for Apple Silicon clusters. SSH-based, zero agent install.

## CLI Quick Reference

```bash
# One-shot: discover + print metrics
asmi --hosts m3u1,m3u3,m4m1

# Auto-discover via Thunderbolt
asmi --scan thunderbolt

# JSON output (for piping/parsing)
asmi --hosts m3u1,m3u3,m4m1 --format json

# Live watch mode (refreshes every 2s)
asmi --hosts m3u1,m3u3 --watch --interval 2

# Combine discovery methods
asmi --scan thunderbolt,tailscale,arp
```

## Reading asmi Output

### JSON Structure

```json
{
  "aggregates": {
    "nodes_online": 3,
    "nodes_total": 3,
    "total_ram_total_bytes": 962072674304,
    "total_ram_used_bytes": 462833598464,
    "total_watts": 1.432,
    "gpu_avg_percent": 1.1,
    "models_loaded": []
  },
  "nodes": [{ "hostname": "m3u1", "online": true, ... }]
}
```

### Per-Node Fields

| Field | Unit | Notes |
|-------|------|-------|
| `cpu_watts`, `gpu_watts` | **milliwatts** | Divide by 1000 for watts |
| `ane_watts` | milliwatts | Neural Engine power |
| `cpu_percent`, `gpu_percent` | % (0-100) | Utilization |
| `ram_total_bytes` | bytes | Total unified memory |
| `ram_used_bytes` | bytes | Used (app + cached) |
| `ram_app_bytes` | bytes | App memory only |
| `ram_cached_bytes` | bytes | File cache (reclaimable) |
| `chip_model` | string | "Apple M3 Ultra", "Apple M4 Max" |
| `processes` | array | MLX servers, frameworks, ports |
| `rdma` | object | `devices[]` with port_state Active/Down |

**Critical:** `cpu_watts` is milliwatts. A value of 631 = 0.631W, not 631W.

### RDMA Fields

```json
"rdma": {
  "enabled": true,
  "devices": [
    { "name": "rdma_en3", "port_state": "Active" },
    { "name": "rdma_en4", "port_state": "Down" }
  ]
}
```

- **Active** = TB5 cable connected, link negotiated
- **Down** = no cable, cable unseated, or peer node off
- Each M3 Ultra has up to 6 RDMA devices (en2-en7)
- Each M4 Max has up to 4 (en2-en5)
- Only 3 ports needed for 4-node full mesh

### Process Detection

asmi detects MLX framework processes automatically:

```json
"processes": [{
  "framework": "mlx-vlm",
  "pid": 41353,
  "port": 19082,
  "footprint_mb": 377.0,
  "model": null,
  "distributed": null
}]
```

Supported frameworks: `mlx-lm`, `mlx-vlm`, `vllm-mlx`, `mlx-share`

## HTTP Daemon Endpoints

When running as `asmi --serve --port 9090`:

### Metrics
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/metrics` | GET | Full NodeSnapshot JSON |
| `/health` | GET | `{ ok, hostname, uptime_secs }` |
| `/processes` | GET | Running MLX processes |
| `/stream` | GET | SSE push every ~2s |
| `/runtime` | GET | Python, MLX, vLLM, macOS versions |

### Cluster (requires `--cluster`)
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/cluster` | GET | All nodes' snapshots |
| `/nodes` | GET | Known hostnames |

### Hardware
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/thunderbolt` | GET | TB device tree (60s cache) |
| `/arp` | GET | ARP table |
| `/disk` | GET | Per-device I/O stats |
| `/network` | GET | Per-interface throughput |
| `/models` | GET | Local model inventory (60s cache) |
| `/volumes` | GET | Mounted external drives |

### JACCL
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/jaccl/config` | GET | Generated hostfile from RDMA topology |
| `/jaccl/config` | POST | Write hostfile to disk |

### MLX Server Lifecycle
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/serve/status` | GET | All managed server ports |
| `/serve/load` | POST | Start server + load model |
| `/serve/stop` | POST | Stop server on port |
| `/serve/reload` | POST | Hot-reload model |
| `/serve/share` | POST | Start RDMA share session |
| `/serve/share/status` | GET | Share session state |
| `/serve/share/stop` | POST | Stop share session |

### Topology (wraps mlx.distributed_config)
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/topology` | GET | Full mesh adjacency, RDMA devices, missing links |
| `/topology/dot` | GET | Graphviz DOT format |
| `/topology/validate` | GET | Mesh completeness + JACCL readiness |

### Health & Setup
| Endpoint | Method | Returns |
|----------|--------|---------|
| `/health/setup` | GET | MLX/RDMA/SSH/disk validation |
| `/health/network` | GET | TB service health |
| `/health/network/fix` | GET | Auto-repair TB services |

## Common Tasks

### Check which nodes are online
```bash
asmi --hosts m3u1,m3u3,m4m1 --format json | python3 -c "
import json,sys
d=json.load(sys.stdin)
print(f'{d[\"aggregates\"][\"nodes_online\"]}/{d[\"aggregates\"][\"nodes_total\"]} nodes online')
for n in d['nodes']:
    st = 'ONLINE' if n['online'] else 'OFFLINE'
    print(f'  {n[\"hostname\"]}: {st}')
"
```

### Check RDMA connectivity for a node
```bash
asmi --hosts <NODE> --format json | python3 -c "
import json,sys
n=json.load(sys.stdin)['nodes'][0]
rdma=n.get('rdma',{})
if not rdma.get('enabled'): print('RDMA not enabled'); sys.exit()
for d in rdma.get('devices',[]):
    print(f'{d[\"name\"]}: {d[\"port_state\"]}')
"
```

### Get total cluster RAM and power
```bash
asmi --hosts m3u1,m3u3,m4m1 --format json | python3 -c "
import json,sys
a=json.load(sys.stdin)['aggregates']
print(f'RAM: {a[\"total_ram_used_bytes\"]/1e9:.0f}/{a[\"total_ram_total_bytes\"]/1e9:.0f} GB')
print(f'Power: {a[\"total_watts\"]:.1f}W')
print(f'Nodes: {a[\"nodes_online\"]}/{a[\"nodes_total\"]}')
"
```

### Check what models are loaded
```bash
curl -s http://localhost:9090/serve/status | python3 -m json.tool
# Or via CLI:
asmi --hosts m3u1,m3u3,m4m1 --format json | python3 -c "
import json,sys
for n in json.load(sys.stdin)['nodes']:
    for p in n.get('processes',[]):
        print(f'{n[\"hostname\"]}: {p[\"framework\"]} port={p[\"port\"]} mem={p[\"footprint_mb\"]}MB')
"
```

### Generate JACCL hostfile
```bash
curl -s http://localhost:9090/jaccl/config | python3 -m json.tool
```

### Verify TB5 mesh topology
```bash
# Use mlx.distributed_config for authoritative topology discovery
mlx.distributed_config --verbose --hosts m3u1,m3u2,m3u3,m4m1 \
  --over thunderbolt --backend jaccl --dot

# Or via asmi endpoint (wraps mlx.distributed_config)
curl -s http://localhost:9090/topology | python3 -m json.tool
```

## Daemon Management

```bash
asmi daemon status              # health check all nodes
asmi daemon start [--node X]    # start launchd agent
asmi daemon stop [--node X]     # stop launchd agent
asmi daemon restart [--node X]  # restart
asmi daemon deploy [--node X]   # scp binary + plist to node
asmi daemon logs [--node X]     # tail log
```

See `asmi-setup` skill for full deployment instructions.

## Node Discovery Methods

| Method | Flag | Discovers Via |
|--------|------|---------------|
| Thunderbolt | `--scan thunderbolt` | `ifconfig` link-local 169.254.x.x |
| Tailscale | `--scan tailscale` | `tailscale status --json` |
| ARP | `--scan arp` | `arp -a` local network |
| ARP All | `--scan arp-all` | Broader ARP scan |
| System Profiler | `--scan profiler` | `system_profiler SPThunderboltDataType` |
| Explicit | `--hosts a,b,c` | Direct hostname/IP list |

## Cluster Hardware Reference

| Node | Chip | RAM | TB5 Ports | RDMA Devices |
|------|------|-----|-----------|--------------|
| m3u2 (coordinator) | M3 Ultra | 512 GB | 6 | en2-en7 |
| m3u1 | M3 Ultra | 512 GB | 6 | en2-en7 |
| m3u3 | M3 Ultra | 256 GB | 6 | en2-en7 |
| m4m1 | M4 Max | 128 GB | 4 | en2-en5 |

Total cluster: 1,408 GB (1.375 TB) unified memory.

### Non-Cluster Nodes

| Node | Chip | RAM | Role |
|------|------|-----|------|
| m4mini | M4 (base) | 16 GB | General purpose, not in RDMA cluster |

## Key Paths

| What | Where |
|------|-------|
| Source | `~/Projects/Personal/apple-smi/` |
| Binary | `~/.local/bin/asmi` or `~/.cargo/bin/asmi` |
| NodeMap | `~/.config/asmi/config.json` |
| Daemon log | `~/Library/Logs/asmi-daemon.log` |
| Plist | `~/Library/LaunchAgents/com.asmi.daemon.plist` |

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| 0 nodes discovered | Use `--hosts` explicitly or `--scan thunderbolt` |
| Stale metrics | Check daemon: `asmi daemon status` |
| RDMA all Down | Nodes may be sleeping, or TB5 cables unseated |
| `cpu_watts` looks huge | It's milliwatts — divide by 1000 |
| High `ram_cached_bytes` | File cache is reclaimable, not a problem |
| Process not detected | Only MLX frameworks detected (mlx-lm, mlx-vlm, vllm-mlx) |

<boundaries>
- Always use `--format json` when parsing output programmatically
- Remember `cpu_watts` / `gpu_watts` are milliwatts, not watts
- `ram_cached_bytes` is file cache — reclaimable, not "used" in the traditional sense
- RDMA "Down" doesn't mean broken — it means no cable or peer is off
- Use `--hosts` for reliable results; auto-discovery can miss nodes behind NAT
- Don't confuse asmi HTTP endpoints with r1o or other services on same nodes
- See `asmi-setup` for daemon deployment, `asmi-share` for model transfers
</boundaries>

## Related Skills

- **asmi-setup** — Deploy asmi daemons to cluster nodes
- **asmi-share** — Transfer models between nodes via TB5 rsync
- **mlx-cluster:rdma** — RDMA activation and debugging
- **mlx-cluster:cluster** — Full cluster boot/status/operations
- **connection-debugger** — SSH and TB5 connectivity troubleshooting
