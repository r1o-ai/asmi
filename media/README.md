# asmi media generation

Image (mflux) + video (mlx-video LTX) generation as asmi serve slots.

| Engine | Server | Port | Models |
|--------|--------|------|--------|
| `image_gen` | `image-gen-server.py` (mflux CLIs) | 19095 | z-image-turbo (default), qwen, flux2, kontext |
| `video_gen` | `video-gen-server.py` (mlx-video LTX-2.x) | 19096 | ltx-2.3-distilled (default), ltx-2-distilled, ltx-2.3-dev |

Both servers are stdlib-only Python, bind loopback, and serialize generations
(one at a time; MLX is memory-heavy). The asmi daemon proxies `/media/*` so
the rest of the mesh can reach them.

## Use

```bash
asmi media status                 # where are the gen servers, what do they offer
asmi image "a lighthouse at dusk" --open
asmi image "hyperreal portrait" -m qwen --steps 28 -o portrait.png
asmi video "waves crashing, slow motion"          # async job, polls, downloads
asmi video "pixel art walk cycle" --frames 49 --width 512 --height 512
asmi video "ken burns over a harbor" --image init.png   # I2V

# Daemon API (any node routes to wherever the servers live)
curl -X POST localhost:9090/serve/load -d '{"engine":"image_gen"}'   # start slot
GET  /media/status
POST /media/image                {"prompt": "...", "model": "...", ...}
POST /media/video                (sync — long)
POST /media/video/jobs           → {"id": ...}   GET /media/video/jobs/{id}
GET  /media/image/{name}  /media/video/{name}    (artifacts)
```

Resolution order (CLI and daemon): `IMAGE_GEN_ENDPOINT`/`VIDEO_GEN_ENDPOINT`
env → healthy local server on the engine port → `endpoints.image_gen`/
`.video_gen` in `~/.config/asmi/config.json` → (CLI only) any cluster
daemon's `/media/status`.

## Node deploy (where generations should run)

1. Toolchain: `uv tool install mflux` (image); mlx-video venv at
   `~/venvs/mlx-video` (video). First runs download weights (~20GB for LTX).
2. Scripts: `cp media/*.py ~/.r1o/bin/` (or leave in `~/` — both resolved;
   `ASMI_IMAGE_GEN_SCRIPT`/`ASMI_VIDEO_GEN_SCRIPT` env overrides).
3. Opt in: add `"media_autostart": ["image_gen", "video_gen"]` to the
   daemon's config — note the daemon usually runs as **root**, so that is
   `/var/root/.config/asmi/config.json`, not the user config.
4. Other nodes: set `endpoints.image_gen`/`.video_gen` in their configs
   (tailnet URLs survive LAN changes). asmi ≥0.3.0 persists these; older
   binaries silently drop unknown config keys on save.

## hub migration (pending — hub offline 2026-07-02)

hub still runs the pre-asmi deployment: launchd agents `com.r1o.image-gen` /
`com.r1o.video-gen` + tailscale serve on :19095/:19096. asmi ≥0.3.0 will
**adopt** those processes on port collision (probe recognizes the media
`/health` dialect), so the upgrade is non-breaking in either order. To hand
ownership fully to asmi once hub is back:

```bash
launchctl bootout gui/$(id -u)/com.r1o.image-gen
launchctl bootout gui/$(id -u)/com.r1o.video-gen
rm ~/Library/LaunchAgents/com.r1o.{image,video}-gen.plist
# add media_autostart to /var/root/.config/asmi/config.json, then
sudo launchctl kickstart -k system/com.asmi.daemon
```

Keep the tailscale serve entries — they publish the tailnet HTTPS endpoints
the rest of the fleet's configs point at.
