#!/opt/homebrew/bin/python3
"""
audio-gen-server — thin stdlib HTTP wrapper around mlx-audio (hub).

Mirrors image-gen-server.py / video-gen-server.py shape: zero-dependency
stdlib HTTP server, ThreadingHTTPServer, JSON in / JSON or bytes out, runs
under launchd KeepAlive.

Plan: docs/plans/2026-07-04-multimodal-gen.md (Phase 4a — author only,
NO deploy; orchestrator deploys to hub).

Endpoints:
  GET  /health                          → {"status":"ok","engine":"mlx-audio",
                                           "model":<default>,"busy":bool}
  POST /generate                        → JSON body:
        {"text": "...",                  # required
         "voice": "serena",              # default; see VOICES
         "language": "english",          # default; see LANGUAGES
         "speed": 1.0, "seed": int,
         "ref_audio": "/path/to/ref.wav"}
        Returns 200 {"name":<name>.wav, "seconds":<s>, "model":<model_key>,
                     "voice":<voice>, "language":<lang>}.
        Generation is sync (TTS is ~1-3s; no job queue needed — Plan:42).
  POST /v1/audio/speech                 → OpenAI-compatible alias of /generate.
        Same body shape; same response.
  POST /v1/audio/transcriptions         → multipart/form-data (OpenAI shape):
        file=<audio bytes>, model=whisper-large-v3, language=<lang>?
        Returns 200 {"text":<transcript>, "language":<lang>, "duration":<s>}.
  GET  /audios/<name>.wav               → fetch a previously generated wav.

Config via env:
  AUDIO_GEN_PORT       (default 19097)
  AUDIO_GEN_BIND       (default 127.0.0.1)
  AUDIO_GEN_DIR        (default ~/audio-gen)
  AUDIO_GEN_MODEL      (default ~/Models/Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16)
  AUDIO_GEN_STT_MODEL  (default mlx-community/whisper-large-v3-turbo-asr-fp16)

Shebang: /opt/homebrew/bin/python3 — where mlx-audio 0.4.3 lives on hub.
"""
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

PORT = int(os.environ.get("AUDIO_GEN_PORT", "19097"))
BIND = os.environ.get("AUDIO_GEN_BIND", "127.0.0.1")
OUT_DIR = Path(os.environ.get("AUDIO_GEN_DIR", str(Path.home() / "audio-gen")))
DEFAULT_MODEL_PATH = os.environ.get(
    "AUDIO_GEN_MODEL",
    str(Path.home() / "Models" / "Qwen3-TTS-12Hz-1.7B-CustomVoice-bf16"),
)
DEFAULT_STT_MODEL = os.environ.get(
    "AUDIO_GEN_STT_MODEL", "mlx-community/whisper-large-v3-turbo-asr-fp16"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Plan:2026-07-04-multimodal-gen:39-41. Mirror VOICES/LANGUAGES from
# web/src/lib/media-tools.ts (sourced from Qwen3-TTS config.json).
DEFAULT_VOICE = "serena"
DEFAULT_LANGUAGE = "english"
GEN_LOCK = threading.Lock()  # serialize: MLX is memory-heavy
GEN_TIMEOUT_S = 5 * 60  # first run loads ~3.4GB weights


def _run_tts(text, voice=None, language=None, speed=None, seed=None, ref_audio=None,
             model_path=None, file_prefix=None):
    """Call mlx_audio.tts.generate.generate_audio and return result metadata."""
    from mlx_audio.tts.generate import generate_audio  # imported lazily

    model_path = model_path or DEFAULT_MODEL_PATH
    voice = voice or DEFAULT_VOICE
    language = language or DEFAULT_LANGUAGE
    if file_prefix is None:
        file_prefix = f"tts-{time.strftime('%Y%m%d-%H%M%S')}-{uuid.uuid4().hex[:6]}"

    # mlx_audio writes <output_path>/<file_prefix>.wav (audio_format='wav' default).
    # generate_audio signature (verified on hub, mlx-audio 0.4.3):
    #   text, model, max_tokens, voice, prompt, instruct, speed, lang_code,
    #   cfg_scale, ddpm_steps, sigma, ref_audio, ref_text, stt_model,
    #   output_path, file_prefix, audio_format, ..., temperature, ...
    # NOTE: mlx-audio has NO `seed` param. temperature=0 gives greedy/deterministic
    # output — the closest seed analog. Dict annotated Any so mixed-type values
    # (str/bool/float) don't trip Pyright's narrowing.
    kwargs: dict[str, Any] = dict(
        text=text,
        model=model_path,
        voice=voice,
        lang_code=language,  # Qwen3-TTS expects long names ('english', 'chinese', ...)
        audio_format="wav",
        file_prefix=file_prefix,
        output_path=str(OUT_DIR),
        verbose=False,
    )
    if speed is not None:
        kwargs["speed"] = speed
    if seed is not None:
        # mlx-audio has no seed; temperature=0 → greedy decoding (deterministic).
        kwargs["temperature"] = 0.0
    if ref_audio:
        kwargs["ref_audio"] = ref_audio

    t0 = time.time()
    generate_audio(**kwargs)
    seconds = round(time.time() - t0, 1)

    # mlx-audio splits long text into segments and writes <prefix>_000.wav,
    # _001.wav, ... (single segment for short utterances). Glob for the real
    # output rather than assuming <prefix>.wav.
    candidates = sorted(OUT_DIR.glob(f"{file_prefix}*.wav"))
    if not candidates:
        return None, f"generate_audio wrote no wav in {OUT_DIR} (expected {file_prefix}*.wav)"
    out_wav = candidates[0]  # _000 — first segment; multi-segment join is a follow-up
    return {
        "path": str(out_wav),
        "name": out_wav.name,
        "seconds": seconds,
        "model": "qwen3-tts-12hz",
        "voice": voice,
        "language": language,
    }, None


def _run_stt(audio_path, language=None, model=None):
    """Call mlx_audio.stt.generate.generate_transcription."""
    from mlx_audio.stt.generate import generate_transcription  # imported lazily

    model = model or DEFAULT_STT_MODEL
    kwargs: dict[str, Any] = dict(
        model=model,
        audio=audio_path,
        output_path=str(OUT_DIR / f"stt-{uuid.uuid4().hex[:6]}"),
        format="json",
        verbose=False,
    )
    # OpenAI uses 2-letter codes; mlx-audio's whisper expects them too.
    if language:
        kwargs["language"] = language

    t0 = time.time()
    result = generate_transcription(**kwargs)
    seconds = round(time.time() - t0, 1)

    # mlx-audio 0.4.x generate_transcription may return list[Segment], an
    # STTOutput wrapper, or None — normalize defensively.
    if result is None:
        segments = []
    elif hasattr(result, "segments"):
        segments = result.segments or []
    elif hasattr(result, "__iter__"):
        segments = list(result)
    else:
        segments = [result]

    # Flatten segments to text.
    text_parts = []
    last_language = None
    for seg in segments:
        text_parts.append(getattr(seg, "text", str(seg)))
        last_language = last_language or getattr(seg, "language", None)
    return {
        "text": " ".join(text_parts).strip(),
        "language": last_language or language,
        "duration": seconds,
    }, None


class Handler(BaseHTTPRequestHandler):
    server_version = "audio-gen/1.0"

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
            self._json(200, {
                "status": "ok",
                "engine": "mlx-audio",
                "port": PORT,
                "model": "qwen3-tts-12hz",
                "model_path": DEFAULT_MODEL_PATH,
                "stt_model": DEFAULT_STT_MODEL,
                "default_voice": DEFAULT_VOICE,
                "default_language": DEFAULT_LANGUAGE,
                "busy": GEN_LOCK.locked(),
            })
        elif self.path.startswith("/audios/"):
            fname = os.path.basename(self.path)  # no traversal
            fpath = OUT_DIR / fname
            if fpath.exists() and fpath.suffix == ".wav":
                data = fpath.read_bytes()
                self.send_response(200)
                self.send_header("Content-Type", "audio/wav")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            else:
                self._json(404, {"error": "not found"})
        else:
            self._json(404, {"error": "unknown path",
                             "paths": ["/health", "/generate", "/v1/audio/speech",
                                       "/v1/audio/transcriptions", "/audios/<name>"]})

    def do_POST(self):
        if self.path in ("/generate", "/v1/audio/speech"):
            self._handle_generate()
        elif self.path == "/v1/audio/transcriptions":
            self._handle_transcribe()
        else:
            self._json(404, {"error": "unknown path"})

    def _handle_generate(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
            params = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as e:
            return self._json(400, {"error": f"bad JSON: {e}"})
        if not params.get("text"):
            return self._json(400, {"error": "missing 'text'"})

        with GEN_LOCK:
            try:
                result, err = _run_tts(
                    text=params["text"],
                    voice=params.get("voice"),
                    language=params.get("language"),
                    speed=params.get("speed"),
                    seed=params.get("seed"),
                    ref_audio=params.get("ref_audio"),
                )
            except Exception as e:  # noqa: BLE001 — surface mlx errors verbatim
                return self._json(500, {"error": f"tts exception: {e}"})
        if err or result is None:
            return self._json(500, {"error": err or "tts failed"})
        return self._json(200, {
            "name": result["name"],
            "seconds": result["seconds"],
            "model": result["model"],
            "voice": result["voice"],
            "language": result["language"],
        })

    def _handle_transcribe(self):
        # OpenAI multipart/form-data parse — minimal hand-rolled parser since
        # we can't depend on streaming_form_data or similar (stdlib only).
        ctype = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in ctype:
            return self._json(400, {"error": "expected multipart/form-data"})
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = self.rfile.read(length)
        except (ValueError,):
            return self._json(400, {"error": "bad content-length"})

        parts = _parse_multipart(body, ctype)
        audio_bytes = parts.get("file")
        language = (parts.get("language") or b"").decode("utf-8", errors="replace") or None
        if not audio_bytes:
            return self._json(400, {"error": "missing 'file' field"})

        # Persist bytes to a tmp wav, transcribe, remove.
        tmp_path = OUT_DIR / f"upload-{uuid.uuid4().hex[:8]}.wav"
        try:
            tmp_path.write_bytes(audio_bytes)
            with GEN_LOCK:
                try:
                    result, err = _run_stt(str(tmp_path), language=language)
                except Exception as e:  # noqa: BLE001
                    return self._json(500, {"error": f"stt exception: {e}"})
        finally:
            try:
                tmp_path.unlink(missing_ok=True)
            except OSError:
                pass
        if err or result is None:
            return self._json(500, {"error": err or "stt failed"})
        return self._json(200, {
            "text": result["text"],
            "language": result["language"],
            "duration": result["duration"],
        })


def _parse_multipart(body: bytes, content_type: str) -> dict:
    """Tiny multipart/form-data parser — returns {field_name: bytes_value}.

    stdlib has no public multipart parser; cgi.parse_header is deprecated in
    3.13. We need just the boundary + a name="..." extract per part. Good
    enough for an OpenAI-compatible /v1/audio/transcriptions on a loopback
    server behind asmi's auth — not a general-purpose parser.
    """
    # Extract boundary from content-type
    boundary = None
    for tok in content_type.split(";"):
        tok = tok.strip()
        if tok.lower().startswith("boundary="):
            boundary = tok.split("=", 1)[1].strip().strip('"')
            break
    if not boundary:
        return {}
    delim = b"--" + boundary.encode("utf-8")
    parts = {}
    chunks = body.split(delim)
    for chunk in chunks[1:-1]:  # skip preamble and the trailing "--" closing
        # Each chunk: \r\n<headers>\r\n\r\n<bytes>\r\n
        if b"\r\n\r\n" not in chunk:
            continue
        header_blob, _, value = chunk.partition(b"\r\n\r\n")
        name = None
        for line in header_blob.split(b"\r\n"):
            if b'name="' in line:
                low = line.lower()
                if b'content-disposition' in low:
                    i = line.find(b'name="')
                    j = line.find(b'"', i + 6)
                    if i >= 0 and j > i:
                        name = line[i + 6:j].decode("utf-8", errors="replace")
        if name and value:
            # strip trailing \r\n
            if value.endswith(b"\r\n"):
                value = value[:-2]
            parts[name] = value
    return parts


if __name__ == "__main__":
    print(f"audio-gen-server on {BIND}:{PORT}  out={OUT_DIR}  "
          f"model={DEFAULT_MODEL_PATH}  stt={DEFAULT_STT_MODEL}", flush=True)
    ThreadingHTTPServer((BIND, PORT), Handler).serve_forever()
