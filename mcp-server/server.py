#!/usr/bin/env python3
"""asmi MCP Server — wraps asmi's HTTP API as MCP tools.

6 workflow-centric tools covering 35+ HTTP endpoints:
  asmi_status   — dashboard (health + metrics + processes + disk + network)
  asmi_cluster  — cluster-wide view (all nodes)
  asmi_models   — local model inventory + serve status
  asmi_serve    — load/stop/reload/share model servers
  asmi_topology — RDMA mesh topology + validation
  asmi_watchdog — watchdog report + GPU lock + peers

Usage:
  python server.py                    # stdio (for MCP gateway)
  ASMI_HOSTS=m3u1,m3u2 python server.py  # multi-node
"""

import json
import os
import shutil
from pathlib import Path
from typing import Optional


def _resolve_python() -> str:
    """Find the best Python 3 interpreter (prefer 3.14+)."""
    for candidate in ("python3.14", "python3.13", "python3.12", "python3"):
        p = shutil.which(candidate)
        if p:
            return p
    return "python3"

import httpx
from fastmcp import FastMCP

mcp = FastMCP("asmi")

# Default to localhost:9090, override with ASMI_BASE_URL
BASE_URL = os.environ.get("ASMI_BASE_URL", "http://localhost:9090")
# Additional hosts for multi-node queries
HOSTS = [h.strip() for h in os.environ.get("ASMI_HOSTS", "").split(",") if h.strip()]
TIMEOUT = 10


async def _get(path: str, base: Optional[str] = None) -> dict:
    """GET request to asmi HTTP API."""
    url = f"{base or BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()


async def _get_text(path: str, base: Optional[str] = None) -> str:
    """GET request returning raw text."""
    url = f"{base or BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.text


async def _post(path: str, data: Optional[dict] = None, base: Optional[str] = None) -> dict:
    """POST request to asmi HTTP API."""
    url = f"{base or BASE_URL}{path}"
    async with httpx.AsyncClient(timeout=TIMEOUT) as client:
        resp = await client.post(url, json=data or {})
        resp.raise_for_status()
        return resp.json()


def _resolve_base(host: Optional[str]) -> str:
    """Resolve a hostname to a base URL."""
    if not host:
        return BASE_URL
    if host.startswith("http"):
        return host
    return f"http://{host}:9090"


async def _post_flight_check(base: str, port: Optional[int], is_share: bool = False) -> dict:
    """Verify a server/share actually stopped by checking status + processes.

    Returns a dict with:
      - verified: bool — True if the server is confirmed stopped
      - state: str — current server state from /serve/status
      - orphan_processes: list — any MLX processes still running on that port
      - ram_reclaimed: bool — True if no footprint detected for that port
    """
    check = {
        "verified": False,
        "state": "unknown",
        "orphan_processes": [],
        "ram_reclaimed": True,
    }

    try:
        if is_share:
            status = await _get("/serve/share/status", base)
            state = status.get("state", "unknown")
            check["state"] = state
            check["verified"] = state in ("idle", "error")
        else:
            port_param = f"?port={port}" if port else ""
            status = await _get(f"/serve/status{port_param}", base)
            # Single-port query returns ServeStatus, all-ports returns {servers: [...]}
            if "servers" in status:
                servers = status["servers"]
                target = [s for s in servers if s.get("port") == (port or 19080)]
                state = target[0]["state"] if target else "not_found"
            else:
                state = status.get("state", "unknown")
            check["state"] = state
            check["verified"] = state in ("idle", "error")
    except Exception as e:
        check["state"] = f"check_failed: {e}"

    # Check for orphan processes still bound to the port
    try:
        procs = await _get("/processes", base)
        proc_list = procs if isinstance(procs, list) else procs.get("processes", [])
        target_port = port or 19080
        orphans = [p for p in proc_list if p.get("port") == target_port]
        if orphans:
            check["orphan_processes"] = orphans
            check["verified"] = False
            check["ram_reclaimed"] = False
    except Exception:
        pass

    return check


# ─── Preflight Checks ────────────────────────────────────────────────────────

# Engine mapping loaded from engine-map.json (editable without touching Python).
_ENGINE_MAP_PATH = Path(__file__).parent / "engine-map.json"
_ENGINE_MAP: dict = {}


def _load_engine_map() -> dict:
    """Load engine-map.json. Cached after first load."""
    global _ENGINE_MAP
    if _ENGINE_MAP:
        return _ENGINE_MAP
    try:
        _ENGINE_MAP = json.loads(_ENGINE_MAP_PATH.read_text())
    except (FileNotFoundError, json.JSONDecodeError):
        _ENGINE_MAP = {}
    return _ENGINE_MAP


def _detect_engine(model_path: str) -> dict:
    """Read config.json and engine-map.json to detect the correct serving engine.

    Priority: by_model_type → by_config_key → by_architecture_pattern → default mlx_lm.

    Returns:
        {engine, port, reason, config_found, is_moe, has_vision, distributed}
    """
    emap = _load_engine_map()
    engines_meta = emap.get("_engines", {})
    by_model_type = emap.get("by_model_type", {})
    by_arch = emap.get("by_architecture_pattern", [])
    by_key = emap.get("by_config_key", [])

    result = {
        "engine": "unknown",
        "port": 19080,
        "reason": "config.json not found — specify engine manually",
        "config_found": False,
        "is_moe": False,
        "has_vision": False,
    }

    expanded = os.path.expanduser(model_path)
    config_path = Path(expanded) / "config.json"
    if not config_path.exists():
        return result

    try:
        config = json.loads(config_path.read_text())
    except (json.JSONDecodeError, OSError):
        return result

    result["config_found"] = True
    model_type = config.get("model_type", "")
    architectures = config.get("architectures", [])
    has_vision = "vision_config" in config or "visual" in config
    is_moe = "moe" in model_type.lower() or bool(config.get("num_experts") or config.get("num_local_experts"))

    result["has_vision"] = has_vision
    result["is_moe"] = is_moe

    def _set_engine(engine: str, reason: str):
        result["engine"] = engine
        result["port"] = engines_meta.get(engine, {}).get("port", 19080)
        result["reason"] = reason

    # 1. Exact model_type match from engine-map.json
    if model_type in by_model_type:
        entry = by_model_type[model_type]
        _set_engine(entry["engine"], entry.get("reason", f"model_type='{model_type}'"))
        return result

    # 2. Config keys (vision_config, visual)
    for rule in by_key:
        if rule["key"] in config:
            _set_engine(rule["engine"], rule.get("reason", f"config has '{rule['key']}'"))
            return result

    # 3. Architecture pattern match
    for rule in by_arch:
        if any(rule["pattern"] in arch for arch in architectures):
            _set_engine(rule["engine"], rule.get("reason", f"arch matches '{rule['pattern']}'"))
            return result

    # 4. Default: text-only → mlx_lm
    _set_engine("mlx_lm", f"text-only model (model_type='{model_type}')")
    return result


async def _preflight_share(base: str, model: Optional[str] = None) -> dict:
    """Pre-flight checks before starting a distributed share session.

    Returns:
        {passed: bool, warnings: [...], blockers: [...], checks: {...}}
    """
    result = {"passed": True, "warnings": [], "blockers": [], "checks": {}}

    # 1. Check if a share session is already running (singleton)
    try:
        share_st = await _get("/serve/share/status", base)
        state = share_st.get("state", "unknown")
        result["checks"]["existing_session"] = state
        if state not in ("idle", "error"):
            result["blockers"].append(
                f"Share session already active (state={state}, "
                f"model={share_st.get('model', '?')}). Stop it first with share_stop."
            )
            result["passed"] = False
    except Exception as e:
        result["checks"]["existing_session"] = f"check_failed: {e}"

    # 2. Check RDMA mesh health via topology/validate
    try:
        topo = await _get("/topology/validate")
        jaccl_ready = topo.get("jaccl_ready", False)
        subsets = topo.get("jaccl_ready_subsets", [])
        result["checks"]["jaccl_ready"] = jaccl_ready
        result["checks"]["jaccl_subsets"] = subsets
        if not jaccl_ready and not subsets:
            result["blockers"].append(
                "RDMA mesh has no JACCL-ready subsets. "
                "Check ibv_devinfo on nodes and verify PORT_ACTIVE."
            )
            result["passed"] = False
        elif not jaccl_ready:
            result["warnings"].append(
                f"Full mesh not ready, but {len(subsets)} JACCL-ready subset(s) available: "
                f"{subsets}"
            )
    except Exception:
        result["warnings"].append(
            "Could not verify RDMA mesh (topology endpoint unavailable). "
            "Share may still work if hostfile is correct."
        )

    # 3. Detect engine from config.json
    if model:
        engine_info = _detect_engine(model)
        result["checks"]["engine"] = engine_info

        if engine_info["config_found"]:
            result["warnings"].append(
                f"Engine: {engine_info['engine']} (port {engine_info['port']}). "
                f"{engine_info['reason']}"
            )
            # Surface RDMA caveats for this engine (e.g. mlx_vlm utils.py crash)
            emap = _load_engine_map()
            engine_meta = emap.get("_engines", {}).get(engine_info["engine"], {})
            rdma_caveat = engine_meta.get("rdma_caveat")
            if rdma_caveat:
                result["warnings"].append(f"RDMA caveat: {rdma_caveat}")
        else:
            result["warnings"].append(
                "Could not read config.json — engine detection unavailable. "
                "Specify engine manually."
            )

    # 4. Check node health + RAM
    hostname = "unknown"
    try:
        health = await _get("/health", base)
        result["checks"]["node_healthy"] = health.get("ok", False)
        hostname = health.get("hostname", "unknown")
        result["checks"]["hostname"] = hostname
    except Exception:
        result["checks"]["node_healthy"] = False
        result["blockers"].append("Target node not responding on :9090")
        result["passed"] = False

    try:
        metrics = await _get("/metrics", base)
        ram_total = metrics.get("ram_total_bytes", 0)
        ram_app = metrics.get("ram_app_bytes", 0)
        ram_available_gb = (ram_total - ram_app) / (1024**3) if ram_total else 0
        result["checks"]["ram_available_gb"] = round(ram_available_gb, 1)
        result["checks"]["ram_total_gb"] = round(ram_total / (1024**3), 0) if ram_total else 0
        if ram_available_gb < 30:
            result["warnings"].append(
                f"Low available RAM: {ram_available_gb:.0f} GiB (app memory: {ram_app / (1024**3):.0f} GiB). "
                "Large models may fail to load."
            )
    except Exception:
        result["checks"]["ram_free_gb"] = "check_failed"

    # 5. Detect running servers on node — block and let user decide
    try:
        serve_status = await _get("/serve/status", base)
        servers = serve_status.get("servers", [serve_status] if "state" in serve_status else [])
        running = [s for s in servers if s.get("state") == "ready" and s.get("model")]
        result["checks"]["running_servers"] = [
            {"port": s.get("port"), "model": s.get("model")} for s in running
        ]
        if running:
            stop_cmds = [
                f"  asmi_serve(action='stop', port={s.get('port')}, host='{hostname}')"
                for s in running
            ]
            result["blockers"].append(
                f"Node {hostname} has {len(running)} running server(s) consuming RAM. "
                "Stop them first to free memory for the share session:\n"
                + "\n".join(stop_cmds)
            )
            result["passed"] = False
    except Exception:
        result["checks"]["running_servers"] = "check_failed"

    # 6. Hostfile existence and validation
    import os
    default_hf = os.path.expanduser("~/.r1o/hostfiles/default.json")
    fallback_hfs = sorted(
        Path(os.path.expanduser("~")).glob("hostfile-jaccl-*.json"),
        key=lambda p: p.stat().st_mtime if p.exists() else 0,
        reverse=True,
    )
    hostfile_path = None
    if os.path.exists(default_hf):
        hostfile_path = default_hf
    elif fallback_hfs:
        hostfile_path = str(fallback_hfs[0])
        result["warnings"].append(
            f"No default hostfile at ~/.r1o/hostfiles/default.json — "
            f"falling back to {hostfile_path}"
        )
    result["checks"]["hostfile_path"] = hostfile_path

    if not hostfile_path:
        result["blockers"].append(
            "No JACCL hostfile found. Expected at ~/.r1o/hostfiles/default.json "
            "or ~/hostfile-jaccl-*.json. Generate one with: asmi topology --format=hostfile"
        )
        result["passed"] = False
    else:
        # Validate hostfile JSON structure
        try:
            import json as _json
            with open(hostfile_path) as f:
                hf_data = _json.load(f)

            # Must be dict with {backend, hosts} (mlx_lm >= 0.31 format)
            if isinstance(hf_data, list):
                result["blockers"].append(
                    f"Hostfile {hostfile_path} is a bare array — needs dict format with "
                    "'backend' and 'hosts' keys. Fix with:\n"
                    "  python3 -c \"import json; d=json.load(open('{hf}')); "
                    "json.dump({{'backend':'jaccl','envs':['MLX_METAL_FAST_SYNCH=1'],'hosts':d}}, "
                    f"open('{hostfile_path}','w'), indent=2)\""
                )
                result["passed"] = False
            elif isinstance(hf_data, dict):
                backend_val = hf_data.get("backend", "")
                hosts_list = hf_data.get("hosts", [])
                if not backend_val:
                    result["blockers"].append(
                        f"Hostfile missing 'backend' field. Add '\"backend\": \"jaccl\"' to {hostfile_path}"
                    )
                    result["passed"] = False
                if len(hosts_list) < 2:
                    result["blockers"].append(
                        f"Hostfile needs >= 2 host entries, got {len(hosts_list)}"
                    )
                    result["passed"] = False
                # Verify each host entry
                for i, entry in enumerate(hosts_list):
                    if not isinstance(entry, dict) or "ssh" not in entry:
                        result["blockers"].append(f"Hostfile hosts[{i}] missing 'ssh' key")
                        result["passed"] = False
                        break
                result["checks"]["hostfile_backend"] = backend_val or "(empty)"
                result["checks"]["hostfile_nodes"] = [
                    e.get("ssh", "?") for e in hosts_list if isinstance(e, dict)
                ]
                result["checks"]["hostfile_node_count"] = len(hosts_list)
            else:
                result["blockers"].append(f"Hostfile must be a JSON dict, got {type(hf_data).__name__}")
                result["passed"] = False
        except Exception as e:
            result["blockers"].append(f"Failed to parse hostfile {hostfile_path}: {e}")
            result["passed"] = False

    # 7. Check mlx._distributed_utils + mlx_lm.share JACCL availability
    try:
        import subprocess
        check = subprocess.run(
            [_resolve_python(), "-c",
             "from mlx_lm.share import Hostfile, launch_jaccl; print('ok')"],
            capture_output=True, text=True, timeout=10,
        )
        jaccl_ok = check.returncode == 0 and "ok" in check.stdout
        result["checks"]["jaccl_launch"] = jaccl_ok
        if not jaccl_ok:
            stderr = check.stderr.strip().split("\n")[-1] if check.stderr else "import failed"
            result["blockers"].append(
                f"mlx_lm.share JACCL support not available ({stderr}). "
                "Ensure mlx_lm >= 0.31 is installed: pip install -U mlx-lm"
            )
            result["passed"] = False
        else:
            # Validate hostfile is parseable by mlx_lm
            if hostfile_path:
                hf_check = subprocess.run(
                    [_resolve_python(), "-c",
                     f"from mlx_lm.share import Hostfile; "
                     f"hf = Hostfile.from_file('{hostfile_path}'); "
                     f"print(len(hf.hosts))"],
                    capture_output=True, text=True, timeout=10,
                )
                if hf_check.returncode == 0:
                    n = hf_check.stdout.strip()
                    result["checks"]["hostfile_parsed_by_mlx"] = f"{n} hosts"
                else:
                    err = hf_check.stderr.strip().split("\n")[-1]
                    result["blockers"].append(
                        f"Hostfile failed mlx_lm parsing: {err}"
                    )
                    result["passed"] = False
    except Exception as e:
        result["checks"]["jaccl_launch"] = f"check_failed: {e}"

    # 8. Port availability check (default share port)
    share_port = 19080
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        port_in_use = sock.connect_ex(("localhost", share_port)) == 0
        sock.close()
        result["checks"]["port_available"] = not port_in_use
        if port_in_use:
            result["blockers"].append(
                f"Port {share_port} is already in use. "
                f"Stop the existing server first: asmi_serve(action='stop', port={share_port})"
            )
            result["passed"] = False
    except Exception:
        result["checks"]["port_available"] = "check_failed"

    # 9. Duplicate process detection — check if same model is already loading
    try:
        import subprocess
        ps_out = subprocess.run(
            ["pgrep", "-af", "mlx_lm.*server.*" + (model or "")],
            capture_output=True, text=True, timeout=5,
        )
        existing = [
            line.strip() for line in ps_out.stdout.strip().split("\n")
            if line.strip() and "pgrep" not in line
        ]
        result["checks"]["duplicate_processes"] = len(existing)
        if existing and model:
            result["warnings"].append(
                f"Found {len(existing)} existing mlx_lm process(es) matching '{model}'. "
                "This may cause port conflicts or memory waste. Consider stopping them first."
            )
    except Exception:
        result["checks"]["duplicate_processes"] = "check_failed"

    return result


async def _preflight_load(base: str, model: Optional[str] = None, port: Optional[int] = None) -> dict:
    """Pre-flight checks before loading a model on a server port.

    Returns:
        {passed: bool, warnings: [...], blockers: [...], checks: {...}}
    """
    result = {"passed": True, "warnings": [], "blockers": [], "checks": {}}
    target_port = port or 19080

    # 1. Check all running servers on this node
    running_servers = []
    try:
        serve_status = await _get("/serve/status", base)
        servers = serve_status.get("servers", [serve_status] if "state" in serve_status else [])
        running_servers = [s for s in servers if s.get("state") == "ready" and s.get("model")]
        result["checks"]["running_servers"] = [
            {"port": s.get("port"), "model": s.get("model")} for s in running_servers
        ]
    except Exception:
        result["checks"]["running_servers"] = "check_failed"

    # 2. Check if target port specifically has a model loaded
    target_server = next((s for s in running_servers if s.get("port") == target_port), None)
    if target_server:
        current_model = target_server.get("model", "unknown")
        result["checks"]["port_state"] = "ready"
        result["checks"]["current_model"] = current_model
        result["blockers"].append(
            f"Port {target_port} is serving '{current_model}'. "
            f"Stop it first: asmi_serve(action='stop', port={target_port})"
        )
        result["passed"] = False
    else:
        result["checks"]["port_state"] = "idle"
        result["checks"]["current_model"] = None

    # 3. Check if other ports have servers consuming RAM
    other_servers = [s for s in running_servers if s.get("port") != target_port]
    if other_servers:
        models_str = ", ".join(
            f"'{s.get('model', '?')}' on port {s.get('port', '?')}" for s in other_servers
        )
        result["warnings"].append(
            f"{len(other_servers)} other server(s) running: {models_str}. "
            "These consume RAM — stop them if needed to fit a larger model."
        )

    # 4. Auto-detect engine from config.json and check port match
    if model:
        engine_info = _detect_engine(model)
        result["checks"]["engine"] = engine_info
        if engine_info["config_found"]:
            recommended_port = engine_info["port"]
            if target_port != recommended_port:
                result["blockers"].append(
                    f"Port mismatch: loading on port {target_port} but model needs "
                    f"{engine_info['engine']} (port {recommended_port}). "
                    f"Reason: {engine_info['reason']}"
                )
                result["passed"] = False
            else:
                result["warnings"].append(
                    f"Engine: {engine_info['engine']} on port {recommended_port}. "
                    f"Reason: {engine_info['reason']}"
                )

    # 2. Check node RAM
    try:
        metrics = await _get("/metrics", base)
        ram_total = metrics.get("ram_total_bytes", 0)
        ram_app = metrics.get("ram_app_bytes", 0)
        ram_available_gb = (ram_total - ram_app) / (1024**3) if ram_total else 0
        result["checks"]["ram_available_gb"] = round(ram_available_gb, 1)
        if ram_available_gb < 30:
            result["warnings"].append(
                f"Low available RAM: {ram_available_gb:.0f} GiB. Large models may OOM."
            )
    except Exception:
        result["checks"]["ram_free_gb"] = "check_failed"

    # 3. Check node health
    try:
        health = await _get("/health", base)
        result["checks"]["node_healthy"] = health.get("ok", False)
    except Exception:
        result["checks"]["node_healthy"] = False
        result["blockers"].append("Target node not responding on :9090")
        result["passed"] = False

    return result


# ─── Tools ───────────────────────────────────────────────────────────────────


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def asmi_status(host: Optional[str] = None) -> str:
    """Dashboard view of a node: health, CPU/GPU/RAM metrics, ANE power, and running processes.

    Combines /health, /metrics, /processes, and /ane into one call.
    Use when you need a quick overview of a node's state.

    Parameters:
        host: Target hostname (e.g. 'm3u2'). Defaults to local node.
    """
    base = _resolve_base(host)
    results = {}

    for key, path in [("health", "/health"), ("metrics", "/metrics"),
                       ("processes", "/processes"), ("ane", "/ane")]:
        try:
            results[key] = await _get(path, base)
        except Exception as e:
            results[key] = {"error": str(e)}

    # Format human-readable summary
    h = results.get("health", {})
    m = results.get("metrics", {})
    procs = results.get("processes", {})
    ane = results.get("ane", {})

    lines = [
        f"Node: {h.get('hostname', host or 'local')} | Up: {h.get('uptime_secs', 0) // 3600}h",
        f"CPU: {m.get('cpu_percent', '?')}% | GPU: {m.get('gpu_percent', '?')}%",
    ]
    ram_total = m.get('ram_total_bytes', 0)
    ram_app = m.get('ram_app_bytes', 0)
    if ram_total:
        ram_available = ram_total - ram_app
        lines.append(f"RAM: {ram_app / (1024**3):.1f} GiB used | {ram_available / (1024**3):.0f} GiB available | {ram_total / (1024**3):.0f} GiB total")
    else:
        lines.append("RAM: ?/? GiB")
    lines.append(f"ANE: {ane.get('ane_watts', ane.get('ane_mw', '?'))} mW")

    proc_list = procs.get("processes", procs) if isinstance(procs, dict) else procs
    if isinstance(proc_list, list) and proc_list:
        lines.append(f"Processes ({len(proc_list)}):")
        for p in proc_list[:10]:
            model = p.get("model", "unknown")
            port = p.get("port", "?")
            lines.append(f"  - {model} (port {port})")

    return "\n".join(lines) + "\n\n" + json.dumps(results, indent=2, default=str)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def asmi_cluster() -> str:
    """Cluster-wide view: snapshots from all nodes via the hub's /cluster endpoint.

    Shows each node's CPU, GPU, RAM, power, and running models.
    Only works when asmi is running in cluster hub mode.
    """
    try:
        data = await _get("/cluster")
    except Exception as e:
        return f"Cluster endpoint unavailable (hub mode required): {e}"

    # /cluster returns a list directly (not {"nodes": [...]})
    nodes = data if isinstance(data, list) else data.get("nodes", [])
    if not nodes:
        return "No nodes reporting. Is asmi running with --cluster on the hub?"

    lines = [f"Cluster: {len(nodes)} nodes\n"]
    for node in nodes:
        hostname = node.get("hostname", "?")
        cpu = node.get("cpu_percent", "?")
        gpu = node.get("gpu_percent", "?")
        ram_total_bytes = node.get("ram_total_bytes", 0)
        ram_app_bytes = node.get("ram_app_bytes", 0)
        ram_avail = f"{(ram_total_bytes - ram_app_bytes) / (1024**3):.0f}" if ram_total_bytes else "?"
        ram_total = f"{ram_total_bytes / (1024**3):.0f}" if ram_total_bytes else "?"
        lines.append(f"  {hostname}: CPU {cpu}% | GPU {gpu}% | RAM {ram_avail}/{ram_total} GiB available")

        procs = node.get("processes", [])
        for p in procs[:5]:
            lines.append(f"    - {p.get('model', '?')} (port {p.get('port', '?')})")

    return "\n".join(lines) + "\n\n" + json.dumps(data, indent=2, default=str)


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False,
    }
)
async def asmi_models(host: Optional[str] = None) -> str:
    """List local model inventory and active server states.

    Combines /models (downloaded models) with /serve/status (running servers).
    Use to check what's available to load and what's currently serving.

    Parameters:
        host: Target hostname. Defaults to local node.
    """
    base = _resolve_base(host)

    models = {}
    serve = {}
    try:
        models = await _get("/models", base)
    except Exception as e:
        models = {"error": str(e)}
    try:
        serve = await _get("/serve/status", base)
    except Exception as e:
        serve = {"error": str(e)}

    return json.dumps({"models": models, "serve_status": serve}, indent=2, default=str)


@mcp.tool(
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    }
)
async def asmi_serve(
    action: str,
    model: Optional[str] = None,
    port: Optional[int] = None,
    host: Optional[str] = None,
) -> str:
    """Manage model servers: load, stop, reload, or start distributed share sessions.

    Actions:
      - status: Show all server states
      - load: Load a model (requires 'model' param, optional 'port')
      - stop: Stop a server (optional 'port')
      - reload: Reload a model (optional 'port')
      - share: Start distributed inference share session
      - share_status: Check share session status
      - share_stop: Stop share session
      - preflight: Run pre-flight checks without starting anything (requires 'model')

    Parameters:
        action: One of: status, load, stop, reload, share, share_status, share_stop
        model: Model repo path (required for load action)
        port: Server port (default: 19080)
        host: Target hostname. Defaults to local node.
    """
    base = _resolve_base(host)
    port_param = f"?port={port}" if port else ""

    try:
        if action == "status":
            return json.dumps(await _get(f"/serve/status{port_param}", base), indent=2)
        elif action == "load":
            if not model:
                return "Error: 'model' parameter required for load action"
            # Run preflight checks
            preflight = await _preflight_load(base, model, port)
            if not preflight["passed"]:
                return json.dumps({
                    "error": "Preflight checks failed — load not started",
                    "preflight": preflight,
                }, indent=2)
            body = {"model_path": model}
            load_result = await _post(f"/serve/load{port_param}", body, base)
            note = "Load started with warnings" if preflight["warnings"] else None
            return json.dumps({
                "result": load_result,
                "preflight": preflight,
                **({"note": note} if note else {}),
            }, indent=2)
        elif action == "stop":
            stop_result = await _post(f"/serve/stop{port_param}", base=base)
            # Post-flight check: verify the server actually stopped
            import asyncio
            await asyncio.sleep(1)
            post_flight = await _post_flight_check(base, port)
            return json.dumps({
                "stop_result": stop_result,
                "post_flight": post_flight,
            }, indent=2)
        elif action == "reload":
            return json.dumps(await _post(f"/serve/reload{port_param}", base=base), indent=2)
        elif action == "share":
            if not model:
                return "Error: 'model' parameter required for share action"
            # Run preflight checks
            preflight = await _preflight_share(base, model)
            if not preflight["passed"]:
                return json.dumps({
                    "error": "Preflight checks failed — share not started",
                    "preflight": preflight,
                }, indent=2)
            body = {"model_path": model}
            share_result = await _post("/serve/share", body, base)
            return json.dumps({
                "result": share_result,
                "preflight": preflight,
            }, indent=2)
        elif action == "share_status":
            return json.dumps(await _get("/serve/share/status", base), indent=2)
        elif action == "share_stop":
            stop_result = await _post("/serve/share/stop", base=base)
            import asyncio
            await asyncio.sleep(1)
            post_flight = await _post_flight_check(base, None, is_share=True)
            return json.dumps({
                "stop_result": stop_result,
                "post_flight": post_flight,
            }, indent=2)
        elif action == "preflight":
            if not model:
                return "Error: 'model' parameter required for preflight action"
            share_pf = await _preflight_share(base, model)
            load_pf = await _preflight_load(base, model, port)
            return json.dumps({
                "share_preflight": share_pf,
                "load_preflight": load_pf,
            }, indent=2)
        else:
            return f"Unknown action '{action}'. Use: status, load, stop, reload, share, share_status, share_stop, preflight"
    except Exception as e:
        return f"Error: {e}"


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def asmi_topology(format: str = "json") -> str:
    """RDMA/Thunderbolt mesh topology and validation.

    Shows node interconnections, link speeds, and JACCL readiness.
    Only available in cluster hub mode.

    Parameters:
        format: Output format — 'json' (default), 'dot' (Graphviz), or 'validate' (mesh check)
    """
    try:
        if format == "dot":
            return await _get_text("/topology/dot")
        elif format == "validate":
            return json.dumps(await _get("/topology/validate"), indent=2)
        else:
            return json.dumps(await _get("/topology"), indent=2)
    except Exception as e:
        return f"Topology unavailable (requires cluster hub mode): {e}"


@mcp.tool(
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    }
)
async def asmi_watchdog(host: Optional[str] = None) -> str:
    """Process watchdog report: monitored processes, GPU lock detection, RDMA peer heartbeats.

    Combines /watchdog, /watchdog/gpu-lock, and /watchdog/peers.
    Use to diagnose inference hangs or GPU lock issues.

    Parameters:
        host: Target hostname. Defaults to local node.
    """
    base = _resolve_base(host)
    results = {}

    for key, path in [("watchdog", "/watchdog"),
                       ("gpu_lock", "/watchdog/gpu-lock"),
                       ("peers", "/watchdog/peers")]:
        try:
            results[key] = await _get(path, base)
        except Exception as e:
            results[key] = {"error": str(e)}

    return json.dumps(results, indent=2, default=str)


if __name__ == "__main__":
    mcp.run(transport="stdio", show_banner=False)
