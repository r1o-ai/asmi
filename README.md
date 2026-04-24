# asmi — Apple Silicon Machine Intelligence

A lightweight daemon that reports live hardware metrics from Apple Silicon Macs over a JSON/HTTP API. One binary per node, launchd-managed, designed for clusters.

## What it does

- **Live metrics** — CPU, GPU, ANE, RAM (app/cached/available), power draw, thermal state
- **Process inventory** — detects running MLX, vLLM, llama.cpp, and distributed inference processes
- **Model scanner** — indexes models from `~/Models/`, HuggingFace cache, and external volumes
- **Thunderbolt 5 RDMA** — peer detection, link-local IP mapping, JACCL backend verification
- **Serve lifecycle** — start/stop/status for `mlx_lm.server` with hostfile + backend params
- **Setup validation** — checks MLX, RDMA, SSH, disk, network in one call
- **SSE streaming** — push metrics on every poll tick via `/stream`

## Install

### From source

```bash
cargo install --path .
```

### From release binary

```bash
# Download the latest release for your architecture
curl -L https://github.com/<OWNER>/apple-smi/releases/latest/download/asmi-arm64-apple-darwin.tar.gz | tar xz
# Re-sign after download (macOS requirement for binaries from other machines)
codesign -f -s - asmi
# Move to a directory in your PATH
sudo mv asmi /usr/local/bin/
```

## Run

```bash
# Start the HTTP daemon on port 9090
asmi serve

# One-shot metrics snapshot
curl -s http://localhost:9090/snapshot | jq

# Stream metrics via SSE
curl -s http://localhost:9090/stream

# CLI mode — table output to terminal
asmi --hosts localhost
```

## API

26 HTTP endpoints on port 9090. Key routes:

| Endpoint | Method | Description |
|---|---|---|
| `/snapshot` | GET | Full node metrics snapshot |
| `/health` | GET | Quick health check |
| `/health/setup` | GET | MLX/RDMA/SSH/disk validation |
| `/models` | GET | Model inventory scan |
| `/serve/load` | POST | Start mlx_lm.server |
| `/serve/stop` | POST | Stop a running server |
| `/serve/status` | GET | Running server state |
| `/rdma/check` | GET | RDMA device + peer status |
| `/thunderbolt` | GET | TB5 device tree |
| `/stream` | GET | SSE metrics stream |
| `/runtime` | GET | Python, MLX, macOS versions |

See `ROADMAP.md` for the full endpoint list and version history.

## Requirements

- macOS 14 (Sonoma) or later
- Apple Silicon (M1 or later)
- Rust 1.80+ to build from source

## launchd integration

asmi is designed to run as a launchd user agent:

```bash
# Install the plist (r1o does this automatically on first run)
cp com.r1o.asmi.plist ~/Library/LaunchAgents/
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.r1o.asmi.plist
```

## Status

Alpha. The API shape is stabilizing but may change between 0.x versions. Used in production on a 5-node M3 Ultra / M5 Max cluster.

## License

MIT — see [LICENSE](LICENSE).
