#!/usr/bin/env python3
"""
image-gen-server — thin HTTP wrapper around mflux CLIs (hub).

Zero-dependency (stdlib only). Serializes generations with a lock (MLX is
memory-heavy; one generation at a time). Runs under launchd (KeepAlive).

Endpoints:
  GET  /health            → {"status":"ok","engine":"mflux","models":{...},"busy":bool}
  GET  /models            → model registry (name → CLI + defaults)
  POST /generate          → JSON body:
        {"prompt": "...",                  # required
         "model": "z-image-turbo",         # default; see MODELS
         "steps": int, "width": int, "height": int,
         "seed": int, "quantize": 4|8,     # optional
         "response": "png"|"json"}         # png (default) streams image bytes;
                                           # json returns {"path": ..., "b64": ...}
  POST /jobs              → async job queue (Plan:2026-07-04-multimodal-gen:57).
        Same body as /generate; returns 202 {"id":..., "status":"queued",
        "poll":"/jobs/<id>", "busy":bool}. Worker thread renders serially.
  GET  /jobs/<id>         → poll a job: {"status":"queued|generating|done|error",
        "result":{...}|null, "error":str|str}
  GET  /images/<name>.png → fetch a previously generated image

Config via env:
  IMAGE_GEN_PORT   (default 19095)
  IMAGE_GEN_BIND   (default 127.0.0.1 — asmi's daemon proxies for the mesh)
  IMAGE_GEN_DIR    (default ~/images-gen)
  MFLUX_BIN_DIR    (default ~/.local/bin)

Managed by asmi as ServeEngine::ImageGen — POST :9090/serve/load
{"engine":"image_gen"}. Deployed to ~/.r1o/bin/ by `asmi daemon deploy`.
"""
import base64
import json
import os
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("IMAGE_GEN_PORT", "19095"))
BIND = os.environ.get("IMAGE_GEN_BIND", "127.0.0.1")
OUT_DIR = Path(os.environ.get("IMAGE_GEN_DIR", str(Path.home() / "images-gen")))
BIN_DIR = Path(os.environ.get("MFLUX_BIN_DIR", str(Path.home() / ".local/bin")))
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Model registry: name → (CLI binary, default args). Defaults tuned from mid-2026
# community numbers: Z-Image-Turbo = speed default; qwen = quality; flux2-klein = light.
MODELS = {
    "z-image-turbo": {
        "bin": "mflux-generate-z-image-turbo",
        "defaults": {"steps": 8, "width": 1024, "height": 1024, "quantize": 4},
    },
    "qwen": {
        "bin": "mflux-generate-qwen",
        "defaults": {"steps": 28, "width": 1024, "height": 1024, "quantize": 8},
    },
    "flux2": {
        "bin": "mflux-generate-flux2",
        "defaults": {"steps": 4, "width": 1024, "height": 1024, "quantize": 4},
    },
    "kontext": {
        "bin": "mflux-generate-kontext",
        "defaults": {"steps": 25, "width": 1024, "height": 1024, "quantize": 8},
    },
}
DEFAULT_MODEL = "z-image-turbo"
GEN_LOCK = threading.Lock()
GEN_TIMEOUT_S = 30 * 60  # qwen first-run includes HF download; be generous

# Async job queue — "subagent" pattern (mirrors video-gen-server.py:76-95).
# POST /jobs returns immediately, a worker thread generates serially under
# GEN_LOCK, GET /jobs/<id> reports status. In-memory only (KeepAlive restart
# clears state; the png itself survives in OUT_DIR).
# Plan:2026-07-04-multimodal-gen:57.
JOBS = {}
JOBS_LOCK = threading.Lock()


def _job_worker(job_id, params):
    """Background worker — runs one image generation under the global lock."""
    with JOBS_LOCK:
        JOBS[job_id]["status"] = "queued"
    with GEN_LOCK:
        with JOBS_LOCK:
            JOBS[job_id]["status"] = "generating"
            JOBS[job_id]["started"] = time.time()
        result, err = run_generation(params)
    with JOBS_LOCK:
        if err or result is None:
            JOBS[job_id].update(status="error", error=err or "generation failed")
        else:
            JOBS[job_id].update(status="done", result=result)


def run_generation(params):
    model_key = params.get("model", DEFAULT_MODEL)
    if model_key not in MODELS:
        return None, f"unknown model '{model_key}'; choose from {sorted(MODELS)}"
    spec = MODELS[model_key]
    cli = BIN_DIR / spec["bin"]
    if not cli.exists():
        return None, f"CLI not found: {cli}"

    merged = dict(spec["defaults"])
    for k in ("steps", "width", "height", "quantize", "seed"):
        if k in params and params[k] is not None:
            merged[k] = params[k]

    name = f"{model_key}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.png"
    out_path = OUT_DIR / name
    cmd = [
        str(cli),
        "--prompt", str(params["prompt"]),
        "--steps", str(merged["steps"]),
        "--width", str(merged["width"]),
        "--height", str(merged["height"]),
        "--output", str(out_path),
    ]
    if merged.get("quantize"):
        cmd += ["-q", str(merged["quantize"])]
    if merged.get("seed") is not None:
        cmd += ["--seed", str(merged["seed"])]

    env = dict(os.environ)
    env["PATH"] = f"{BIN_DIR}:/opt/homebrew/bin:/usr/bin:/bin"
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=GEN_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return None, f"generation timed out after {GEN_TIMEOUT_S}s"
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return None, f"mflux exit {proc.returncode}: {tail}"
    return {"path": str(out_path), "name": name, "seconds": round(time.time() - t0, 1),
            "model": model_key, "args": merged}, None


class Handler(BaseHTTPRequestHandler):
    server_version = "image-gen/1.0"

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):  # quiet-ish log to stdout (launchd captures)
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {format % args}", flush=True)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "engine": "mflux", "port": PORT,
                             "models": sorted(MODELS), "default": DEFAULT_MODEL,
                             "busy": GEN_LOCK.locked()})
        elif self.path == "/models":
            self._json(200, MODELS)
        elif self.path.startswith("/jobs/"):
            # Plan:2026-07-04-multimodal-gen:57. Poll an async job.
            job_id = os.path.basename(self.path)
            with JOBS_LOCK:
                job = JOBS.get(job_id)
            if not job:
                return self._json(404, {"error": "unknown job", "id": job_id})
            # Don't leak the full in-memory dict; project to the public shape.
            payload = {"id": job_id, "status": job["status"]}
            if "result" in job:
                payload["result"] = {
                    "name": job["result"]["name"],
                    "seconds": job["result"]["seconds"],
                    "model": job["result"]["model"],
                }
            if "error" in job:
                payload["error"] = job["error"]
            if "prompt" in job:
                payload["prompt"] = job["prompt"]
            return self._json(200, payload)
        elif self.path.startswith("/images/"):
            fname = os.path.basename(self.path)  # no traversal
            fpath = OUT_DIR / fname
            if fpath.exists() and fpath.suffix == ".png":
                data = fpath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "image/png")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json(404, {"error": "not found"})
        else:
            self._json(404, {"error": "unknown path", "paths": ["/health", "/models", "/generate", "/jobs", "/jobs/<id>", "/images/<name>"]})

    def do_POST(self):
        if self.path == "/jobs":
            # Plan:2026-07-04-multimodal-gen:57. Async enqueue — mirrors
            # video-gen-server.py:198-214 (POST /jobs → 202 + worker spawn).
            try:
                length = int(self.headers.get("Content-Length", "0"))
                params = json.loads(self.rfile.read(length) or b"{}")
            except (ValueError, json.JSONDecodeError) as e:
                return self._json(400, {"error": f"bad JSON: {e}"})
            if not params.get("prompt"):
                return self._json(400, {"error": "missing 'prompt'"})
            job_id = uuid.uuid4().hex[:12]
            with JOBS_LOCK:
                JOBS[job_id] = {"status": "queued", "created": time.time(),
                                "model": params.get("model", DEFAULT_MODEL),
                                "prompt": str(params["prompt"])[:200]}
            threading.Thread(target=_job_worker, args=(job_id, params), daemon=True).start()
            return self._json(202, {"id": job_id, "status": "queued",
                                    "poll": f"/jobs/{job_id}", "busy": GEN_LOCK.locked()})
        if self.path != "/generate":
            return self._json(404, {"error": "unknown path"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            params = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(400, {"error": f"bad JSON: {e}"})
        if not params.get("prompt"):
            return self._json(400, {"error": "missing 'prompt'"})

        with GEN_LOCK:  # serialize: one generation at a time
            result, err = run_generation(params)
        if err or result is None:
            return self._json(500, {"error": err or "generation failed"})

        if params.get("response") == "json":
            with open(result["path"], "rb") as f:
                result["b64"] = base64.b64encode(f.read()).decode()
            return self._json(200, result)

        data = Path(result["path"]).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "image/png")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Gen-Seconds", str(result["seconds"]))
        self.send_header("X-Gen-Model", result["model"])
        self.send_header("X-Gen-Name", result["name"])
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"image-gen-server on {BIND}:{PORT}  out={OUT_DIR}  bins={BIN_DIR}", flush=True)
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
