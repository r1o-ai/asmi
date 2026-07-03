#!/usr/bin/env python3
"""
video-gen-server — thin HTTP wrapper around mlx-video (LTX-2 / LTX-2.3) on hub.

Zero-dependency stdlib server, same pattern as image-gen-server (:19095).
Serializes generations (one at a time) and clamps MLX cache so a heavy run
cannot wedge the box (lesson from the qwen image smoke test).

Endpoints:
  GET  /health              → {"status":"ok","engine":"mlx-video","busy":bool,...}
  GET  /models              → model registry
  GET  /videos/<name>       → fetch a previously generated .mp4
  POST /generate            → JSON:
        {"prompt": "...",                    # required
         "model": "ltx-2.3-distilled",       # default; see MODELS
         "num_frames": 97, "width": 768, "height": 512,
         "fps": 24, "seed": int,             # optional
         "image": "<abs path or /images/name.png from image-gen>",  # optional I2V
         "response": "mp4"|"json"}           # mp4 (default) streams bytes

Config via env:
  VIDEO_GEN_PORT     (default 19096)
  VIDEO_GEN_BIND     (default 127.0.0.1 — asmi's daemon proxies for the mesh)
  VIDEO_GEN_DIR      (default ~/videos-gen)
  MLX_VIDEO_PYTHON   (default ~/venvs/mlx-video/bin/python3)
  VIDEO_MLX_CACHE_GB (default 96 — MLX cache clamp so gens can't eat 512GB)

Managed by asmi as ServeEngine::VideoGen — POST :9090/serve/load
{"engine":"video_gen"}. Deployed to ~/.r1o/bin/ by `asmi daemon deploy`.
"""
import base64
import json
import os
import re
import subprocess
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("VIDEO_GEN_PORT", "19096"))
BIND = os.environ.get("VIDEO_GEN_BIND", "127.0.0.1")
OUT_DIR = Path(os.environ.get("VIDEO_GEN_DIR", str(Path.home() / "videos-gen")))
PYBIN = os.environ.get("MLX_VIDEO_PYTHON", str(Path.home() / "venvs/mlx-video/bin/python3"))
CACHE_GB = os.environ.get("VIDEO_MLX_CACHE_GB", "96")
IMAGES_DIR = Path.home() / "images-gen"  # cross-link: image-gen output usable as I2V input
OUT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    # LTX-2.3 repos ship transformer+VAE only; the Gemma text encoder lives in the
    # LTX-2 (2.0) repos — pass it via --text-encoder-repo or loading fails with
    # "Config file not found at <snapshot root>".
    "ltx-2.3-distilled": {
        "repo": "prince-canuma/LTX-2.3-distilled",
        "pipeline": "distilled",
        "text_encoder_repo": "prince-canuma/LTX-2-distilled",
        "defaults": {"num_frames": 97, "width": 768, "height": 512, "fps": 24},
    },
    "ltx-2-distilled": {
        "repo": "prince-canuma/LTX-2-distilled",
        "pipeline": "distilled",
        "defaults": {"num_frames": 97, "width": 768, "height": 512, "fps": 24},
    },
    "ltx-2.3-dev": {
        "repo": "prince-canuma/LTX-2.3-dev",
        "pipeline": "dev",
        "text_encoder_repo": "prince-canuma/LTX-2-dev",
        "defaults": {"num_frames": 97, "width": 768, "height": 512, "fps": 24, "cfg_scale": 3.0},
    },
}
DEFAULT_MODEL = "ltx-2.3-distilled"
GEN_LOCK = threading.Lock()
GEN_TIMEOUT_S = 60 * 60  # first run downloads ~20GB weights

# Async job queue — "subagent" pattern: POST /jobs returns immediately, a worker
# thread generates, GET /jobs/<id> reports status. Jobs are in-memory (KeepAlive
# restart clears them; the mp4 itself survives in OUT_DIR).
JOBS = {}
JOBS_LOCK = threading.Lock()


def _job_worker(job_id, params):
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


def safe_name(s):
    return re.sub(r"[^A-Za-z0-9._-]", "", s)


def run_generation(params):
    model_key = params.get("model", DEFAULT_MODEL)
    if model_key not in MODELS:
        return None, f"unknown model '{model_key}'; choose from {sorted(MODELS)}"
    spec = MODELS[model_key]
    merged = dict(spec["defaults"])
    for k in ("num_frames", "width", "height", "fps", "seed", "cfg_scale"):
        if k in params and params[k] is not None:
            merged[k] = params[k]
    # LTX requires H/W divisible by 64
    for dim in ("width", "height"):
        merged[dim] = max(256, (int(merged[dim]) // 64) * 64)

    name = f"{model_key}-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}.mp4"
    out_path = OUT_DIR / name
    cmd = [
        PYBIN, "-m", "mlx_video.models.ltx_2.generate",
        "--prompt", str(params["prompt"]),
        "--model-repo", spec["repo"],
        "--pipeline", spec["pipeline"],
        "-n", str(merged["num_frames"]),
        "--width", str(merged["width"]),
        "--height", str(merged["height"]),
        "--fps", str(merged["fps"]),
        "--output-path", str(out_path),
    ]
    if spec.get("text_encoder_repo"):
        cmd += ["--text-encoder-repo", spec["text_encoder_repo"]]
    if merged.get("cfg_scale") is not None:
        cmd += ["--cfg-scale", str(merged["cfg_scale"])]
    if merged.get("seed") is not None:
        cmd += ["--seed", str(merged["seed"])]
    img = params.get("image")
    if img:
        # allow bare image-gen names ("z-image-...png") or absolute paths under home
        cand = IMAGES_DIR / safe_name(os.path.basename(img)) if not os.path.isabs(img) else Path(img)
        if not cand.exists():
            return None, f"image not found: {cand}"
        cmd += ["--image", str(cand)]

    env = dict(os.environ)
    env["PATH"] = f"{Path.home()}/.local/bin:/opt/homebrew/bin:/usr/bin:/bin"
    env["MLX_CACHE_LIMIT_GB"] = CACHE_GB  # guard: don't let cache balloon wedge the box
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=GEN_TIMEOUT_S, env=env)
    except subprocess.TimeoutExpired:
        return None, f"generation timed out after {GEN_TIMEOUT_S}s"
    if proc.returncode != 0 or not out_path.exists():
        tail = (proc.stderr or proc.stdout or "")[-2000:]
        return None, f"mlx_video exit {proc.returncode}: {tail}"
    return {"path": str(out_path), "name": name, "seconds": round(time.time() - t0, 1),
            "model": model_key, "args": merged}, None


class Handler(BaseHTTPRequestHandler):
    server_version = "video-gen/1.0"

    def _json(self, code, obj):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        print(f"[{time.strftime('%H:%M:%S')}] {self.address_string()} {format % args}", flush=True)

    def do_GET(self):
        if self.path == "/health":
            self._json(200, {"status": "ok", "engine": "mlx-video", "port": PORT,
                             "models": sorted(MODELS), "default": DEFAULT_MODEL,
                             "busy": GEN_LOCK.locked()})
        elif self.path == "/models":
            self._json(200, MODELS)
        elif self.path.startswith("/jobs/"):
            job_id = safe_name(os.path.basename(self.path))
            with JOBS_LOCK:
                job = JOBS.get(job_id)
                self._json(200 if job else 404, dict(job, id=job_id) if job else {"error": "unknown job"})
        elif self.path.startswith("/videos/"):
            fname = safe_name(os.path.basename(self.path))
            fpath = OUT_DIR / fname
            if fpath.exists() and fpath.suffix == ".mp4":
                data = fpath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "video/mp4")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json(404, {"error": "not found"})
        else:
            self._json(404, {"error": "unknown path", "paths": ["/health", "/models", "/generate", "/videos/<name>"]})

    def do_POST(self):
        if self.path == "/jobs":
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
        if GEN_LOCK.locked() and params.get("wait") is False:
            return self._json(409, {"error": "busy — one generation at a time"})

        with GEN_LOCK:
            result, err = run_generation(params)
        if err or result is None:
            return self._json(500, {"error": err or "generation failed"})

        if params.get("response") == "json":
            with open(result["path"], "rb") as f:
                result["b64"] = base64.b64encode(f.read()).decode()
            return self._json(200, result)

        data = Path(result["path"]).read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("X-Gen-Seconds", str(result["seconds"]))
        self.send_header("X-Gen-Model", result["model"])
        self.send_header("X-Gen-Name", result["name"])
        self.end_headers()
        self.wfile.write(data)


if __name__ == "__main__":
    print(f"video-gen-server on {BIND}:{PORT}  out={OUT_DIR}  python={PYBIN}  cache_clamp={CACHE_GB}GB", flush=True)
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
