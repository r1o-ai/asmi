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
from typing import Optional

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
    ram_used = m.get('ram_used_bytes', 0)
    if ram_total:
        lines.append(f"RAM: {ram_used / (1024**3):.1f}/{ram_total / (1024**3):.0f} GiB")
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
        ram_used_bytes = node.get("ram_used_bytes", 0)
        ram_total = f"{ram_total_bytes / (1024**3):.0f}" if ram_total_bytes else "?"
        ram_used = f"{ram_used_bytes / (1024**3):.0f}" if ram_used_bytes else "?"
        lines.append(f"  {hostname}: CPU {cpu}% | GPU {gpu}% | RAM {ram_used}/{ram_total} GiB")

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
            body = {"model_path": model}
            return json.dumps(await _post(f"/serve/load{port_param}", body, base), indent=2)
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
            body = {"model_path": model} if model else {}
            return json.dumps(await _post("/serve/share", body, base), indent=2)
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
        else:
            return f"Unknown action '{action}'. Use: status, load, stop, reload, share, share_status, share_stop"
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
