//! Media generation — image (mflux) + video (mlx-video LTX) as asmi citizens.
//!
//! The gen servers are stdlib-only Python HTTP wrappers (vendored in media/),
//! managed as serve slots: `ServeEngine::ImageGen` (:19095) and
//! `ServeEngine::VideoGen` (:19096). They bind loopback on the node that runs
//! them; this module gives the rest of the mesh a way in:
//!
//! - **Daemon routes** `/media/*` proxy to wherever the gen server lives
//!   (env override → healthy local port → `endpoints` in config.json).
//! - **CLI** `asmi image` / `asmi video` / `asmi media status` resolve a
//!   target the same way, falling back to any cluster daemon's `/media/*`.

use std::time::Duration;

use axum::extract::{Path as AxumPath, State};
use axum::routing::{get, post};
use axum::{Json, Router};

use asmi_core::ServeEngine;

use crate::daemon::{ApiError, AppState};

/// Generation timeout for image proxying (mflux qwen first run downloads weights).
const IMAGE_GEN_TIMEOUT_SECS: u64 = 31 * 60;
/// Generation timeout for sync video proxying (first run downloads ~20GB).
const VIDEO_GEN_TIMEOUT_SECS: u64 = 61 * 60;

/// The two media kinds: (config/endpoint key, engine).
pub const MEDIA_KINDS: [(&str, ServeEngine); 2] = [
    ("image_gen", ServeEngine::ImageGen),
    ("video_gen", ServeEngine::VideoGen),
];

fn endpoint_key(engine: ServeEngine) -> &'static str {
    match engine {
        ServeEngine::ImageGen => "image_gen",
        ServeEngine::VideoGen => "video_gen",
        _ => unreachable!("endpoint_key called for non-media engine"),
    }
}

/// Env var carrying a direct endpoint override (same vars Claude Code uses).
fn endpoint_env(engine: ServeEngine) -> &'static str {
    match engine {
        ServeEngine::ImageGen => "IMAGE_GEN_ENDPOINT",
        ServeEngine::VideoGen => "VIDEO_GEN_ENDPOINT",
        _ => unreachable!(),
    }
}

/// GET {base}/health and parse the media-gen JSON body.
async fn probe_health(base: &str, timeout: Duration) -> Option<serde_json::Value> {
    let client = reqwest::Client::builder().timeout(timeout).build().ok()?;
    let resp = client.get(format!("{base}/health")).send().await.ok()?;
    if !resp.status().is_success() {
        return None;
    }
    let body: serde_json::Value = resp.json().await.ok()?;
    // Only claim the base if it speaks the media dialect.
    matches!(body["engine"].as_str(), Some("mflux") | Some("mlx-video")).then_some(body)
}

/// Resolve the base URL of a gen server. Returns (base, via).
/// Order: env override → healthy local port → config `endpoints` entry.
async fn resolve_base(state: &AppState, engine: ServeEngine) -> Option<(String, &'static str)> {
    if let Ok(ep) = std::env::var(endpoint_env(engine)) {
        if !ep.is_empty() {
            return Some((ep.trim_end_matches('/').to_string(), "env"));
        }
    }
    let port = crate::serve::port_for_engine(engine);
    let local = format!("http://127.0.0.1:{port}");
    if probe_health(&local, Duration::from_millis(600)).await.is_some() {
        return Some((local, "local"));
    }
    let ep = {
        let nm = state.node_map.read().await;
        nm.endpoints.get(endpoint_key(engine)).cloned()
    };
    ep.map(|e| (e.trim_end_matches('/').to_string(), "config"))
}

fn unavailable(engine: ServeEngine) -> ApiError {
    let key = endpoint_key(engine);
    ApiError::NotFound(format!(
        "no {key} server reachable — POST /serve/load {{\"engine\":\"{key}\"}} to start one \
         here, or set endpoints.{key} in ~/.config/asmi/config.json"
    ))
}

/// Mirror a gen-server response (status, content-type, X-Gen-* headers, body).
async fn passthrough(resp: reqwest::Response) -> Result<axum::response::Response, ApiError> {
    let status = axum::http::StatusCode::from_u16(resp.status().as_u16())
        .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
    let mut builder = axum::response::Response::builder().status(status);
    for h in ["content-type", "x-gen-seconds", "x-gen-model", "x-gen-name"] {
        if let Some(v) = resp.headers().get(h) {
            builder = builder.header(h, v.clone());
        }
    }
    let bytes = resp
        .bytes()
        .await
        .map_err(|e| ApiError::Internal(format!("reading gen server response: {e}")))?;
    builder
        .body(axum::body::Body::from(bytes))
        .map_err(|e| ApiError::Internal(e.to_string()))
}

async fn proxy_post(
    state: &AppState,
    engine: ServeEngine,
    path: &str,
    body: serde_json::Value,
    timeout_secs: u64,
) -> Result<axum::response::Response, ApiError> {
    let (base, _via) = resolve_base(state, engine).await.ok_or_else(|| unavailable(engine))?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    let resp = client
        .post(format!("{base}{path}"))
        .json(&body)
        .send()
        .await
        .map_err(|e| ApiError::Internal(format!("gen server unreachable at {base}: {e}")))?;
    passthrough(resp).await
}

async fn proxy_get(
    state: &AppState,
    engine: ServeEngine,
    path: &str,
    timeout_secs: u64,
) -> Result<axum::response::Response, ApiError> {
    let (base, _via) = resolve_base(state, engine).await.ok_or_else(|| unavailable(engine))?;
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(timeout_secs))
        .build()
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    let resp = client
        .get(format!("{base}{path}"))
        .send()
        .await
        .map_err(|e| ApiError::Internal(format!("gen server unreachable at {base}: {e}")))?;
    passthrough(resp).await
}

/// Strip anything that isn't a safe artifact/job name (defense-in-depth; the
/// gen servers basename + sanitize again on their side).
fn safe_name(name: &str) -> String {
    name.chars()
        .filter(|c| c.is_ascii_alphanumeric() || matches!(c, '.' | '_' | '-'))
        .collect()
}

// ===========================================================================
// Daemon handlers
// ===========================================================================

/// GET /media/status — availability + health of both media kinds.
async fn media_status_handler(State(state): State<AppState>) -> Json<serde_json::Value> {
    let mut out = serde_json::Map::new();
    for (key, engine) in MEDIA_KINDS {
        let entry = match resolve_base(&state, engine).await {
            Some((base, via)) => match probe_health(&base, Duration::from_secs(3)).await {
                Some(h) => serde_json::json!({
                    "available": true, "base": base, "via": via,
                    "models": h["models"], "default": h["default"], "busy": h["busy"],
                }),
                None => serde_json::json!({
                    "available": false, "base": base, "via": via,
                    "error": "resolved but /health not responding",
                }),
            },
            None => serde_json::json!({
                "available": false,
                "hint": format!(
                    "POST /serve/load {{\"engine\":\"{key}\"}} to start locally, \
                     or set endpoints.{key} in ~/.config/asmi/config.json"
                ),
            }),
        };
        out.insert(key.to_string(), entry);
    }
    Json(serde_json::Value::Object(out))
}

/// POST /media/image — proxy a synchronous image generation.
async fn media_image_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response, ApiError> {
    proxy_post(&state, ServeEngine::ImageGen, "/generate", body, IMAGE_GEN_TIMEOUT_SECS).await
}

/// GET /media/image/{name} — fetch a previously generated PNG.
async fn media_image_artifact_handler(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<axum::response::Response, ApiError> {
    let name = safe_name(&name);
    proxy_get(&state, ServeEngine::ImageGen, &format!("/images/{name}"), 120).await
}

/// POST /media/video — proxy a synchronous video generation (long!).
async fn media_video_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response, ApiError> {
    proxy_post(&state, ServeEngine::VideoGen, "/generate", body, VIDEO_GEN_TIMEOUT_SECS).await
}

/// POST /media/video/jobs — submit an async video job (returns job id fast).
async fn media_video_job_submit_handler(
    State(state): State<AppState>,
    Json(body): Json<serde_json::Value>,
) -> Result<axum::response::Response, ApiError> {
    proxy_post(&state, ServeEngine::VideoGen, "/jobs", body, 30).await
}

/// GET /media/video/jobs/{id} — poll an async video job.
async fn media_video_job_status_handler(
    State(state): State<AppState>,
    AxumPath(id): AxumPath<String>,
) -> Result<axum::response::Response, ApiError> {
    let id = safe_name(&id);
    proxy_get(&state, ServeEngine::VideoGen, &format!("/jobs/{id}"), 30).await
}

/// GET /media/video/{name} — fetch a previously generated MP4.
async fn media_video_artifact_handler(
    State(state): State<AppState>,
    AxumPath(name): AxumPath<String>,
) -> Result<axum::response::Response, ApiError> {
    let name = safe_name(&name);
    proxy_get(&state, ServeEngine::VideoGen, &format!("/videos/{name}"), 300).await
}

/// Media routes, merged into the daemon router.
pub fn routes() -> Router<AppState> {
    Router::new()
        .route("/media/status", get(media_status_handler))
        .route("/media/image", post(media_image_handler))
        .route("/media/image/{name}", get(media_image_artifact_handler))
        .route("/media/video", post(media_video_handler))
        .route("/media/video/jobs", post(media_video_job_submit_handler))
        .route("/media/video/jobs/{id}", get(media_video_job_status_handler))
        .route("/media/video/{name}", get(media_video_artifact_handler))
}

// ===========================================================================
// CLI — `asmi image`, `asmi video`, `asmi media status`
// ===========================================================================

/// Where a CLI request goes: straight at a gen server, or through a daemon proxy.
enum Target {
    /// Base URL of a gen server (e.g. http://127.0.0.1:19095).
    Direct(String),
    /// Base URL of an asmi daemon (e.g. http://m3u2:9090).
    Daemon(String),
}

impl Target {
    fn generate_url(&self, engine: ServeEngine) -> String {
        match (self, engine) {
            (Target::Direct(b), _) => format!("{b}/generate"),
            (Target::Daemon(b), ServeEngine::ImageGen) => format!("{b}/media/image"),
            (Target::Daemon(b), _) => format!("{b}/media/video"),
        }
    }

    fn jobs_url(&self) -> String {
        match self {
            Target::Direct(b) => format!("{b}/jobs"),
            Target::Daemon(b) => format!("{b}/media/video/jobs"),
        }
    }

    fn job_status_url(&self, id: &str) -> String {
        match self {
            Target::Direct(b) => format!("{b}/jobs/{id}"),
            Target::Daemon(b) => format!("{b}/media/video/jobs/{id}"),
        }
    }

    fn artifact_url(&self, engine: ServeEngine, name: &str) -> String {
        match (self, engine) {
            (Target::Direct(b), ServeEngine::ImageGen) => format!("{b}/images/{name}"),
            (Target::Direct(b), _) => format!("{b}/videos/{name}"),
            (Target::Daemon(b), ServeEngine::ImageGen) => format!("{b}/media/image/{name}"),
            (Target::Daemon(b), _) => format!("{b}/media/video/{name}"),
        }
    }

    fn describe(&self) -> String {
        match self {
            Target::Direct(b) => format!("{b} (direct)"),
            Target::Daemon(b) => format!("{b} (daemon proxy)"),
        }
    }
}

/// CLI-side resolution, same order as the daemon (env → local → config),
/// then falling back to any cluster daemon's /media proxy.
async fn resolve_target(engine: ServeEngine) -> anyhow::Result<Target> {
    let key = endpoint_key(engine);

    if let Ok(ep) = std::env::var(endpoint_env(engine)) {
        if !ep.is_empty() {
            return Ok(Target::Direct(ep.trim_end_matches('/').to_string()));
        }
    }

    let local = format!("http://127.0.0.1:{}", crate::serve::port_for_engine(engine));
    if probe_health(&local, Duration::from_millis(600)).await.is_some() {
        return Ok(Target::Direct(local));
    }

    // Config endpoint — only if it actually answers (hub may be offline;
    // a healthy local/daemon fallback should win over a dead config entry).
    let nm = asmi_core::NodeMap::load();
    if let Some(ep) = nm.endpoints.get(key) {
        let ep = ep.trim_end_matches('/').to_string();
        if probe_health(&ep, Duration::from_secs(2)).await.is_some() {
            return Ok(Target::Direct(ep));
        }
    }

    // Ask daemons: localhost first, then every known node.
    let daemon_port: u16 = std::env::var("ASMI_DAEMON_PORT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(9090);
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(4))
        .build()?;
    let mut candidates = vec!["localhost".to_string()];
    candidates.extend(nm.nodes.iter().cloned());
    for node in candidates {
        let daemon = format!("http://{node}:{daemon_port}");
        let Ok(resp) = client.get(format!("{daemon}/media/status")).send().await else {
            continue;
        };
        let Ok(status) = resp.json::<serde_json::Value>().await else {
            continue;
        };
        if status[key]["available"].as_bool() == Some(true) {
            return Ok(Target::Daemon(daemon));
        }
    }

    anyhow::bail!(
        "no {key} server reachable.\n  Start one:  curl -X POST localhost:9090/serve/load \
         -d '{{\"engine\":\"{key}\"}}'\n  Or set:     endpoints.{key} in \
         ~/.config/asmi/config.json (e.g. \"http://hub:19095\")\n  Or export:  {}",
        endpoint_env(engine)
    )
}

fn default_output(name: &str) -> String {
    name.to_string()
}

fn save_artifact(bytes: &[u8], output: &Option<String>, server_name: Option<&str>, ext: &str) -> anyhow::Result<String> {
    let path = match output {
        Some(o) => o.clone(),
        None => default_output(server_name.unwrap_or(&format!("asmi-gen.{ext}"))),
    };
    std::fs::write(&path, bytes)?;
    Ok(path)
}

fn maybe_open(path: &str, open: bool) {
    if open {
        let _ = std::process::Command::new("open").arg(path).status();
    }
}

fn header_str(resp: &reqwest::Response, name: &str) -> Option<String> {
    resp.headers().get(name).and_then(|v| v.to_str().ok()).map(String::from)
}

/// `asmi image "<prompt>" [...]` — generate an image, save it locally.
#[allow(clippy::too_many_arguments)]
pub async fn run_image(
    prompt: String,
    model: Option<String>,
    steps: Option<u32>,
    width: Option<u32>,
    height: Option<u32>,
    seed: Option<u64>,
    output: Option<String>,
    open: bool,
) -> anyhow::Result<()> {
    let target = resolve_target(ServeEngine::ImageGen).await?;
    let mut body = serde_json::json!({ "prompt": prompt });
    if let Some(m) = model { body["model"] = m.into(); }
    if let Some(v) = steps { body["steps"] = v.into(); }
    if let Some(v) = width { body["width"] = v.into(); }
    if let Some(v) = height { body["height"] = v.into(); }
    if let Some(v) = seed { body["seed"] = v.into(); }

    println!("image-gen via {} …", target.describe());
    let t0 = std::time::Instant::now();
    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(IMAGE_GEN_TIMEOUT_SECS))
        .build()?;
    let resp = client
        .post(target.generate_url(ServeEngine::ImageGen))
        .json(&body)
        .send()
        .await?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("generation failed ({status}): {text}");
    }
    let name = header_str(&resp, "x-gen-name");
    let secs = header_str(&resp, "x-gen-seconds");
    let bytes = resp.bytes().await?;
    let path = save_artifact(&bytes, &output, name.as_deref(), "png")?;
    println!(
        "✓ {path}  ({} KB, {}s gen, {:.1}s total)",
        bytes.len() / 1024,
        secs.unwrap_or_else(|| "?".into()),
        t0.elapsed().as_secs_f32()
    );
    maybe_open(&path, open);
    Ok(())
}

/// `asmi video "<prompt>" [...]` — async job by default, poll, download.
#[allow(clippy::too_many_arguments)]
pub async fn run_video(
    prompt: String,
    model: Option<String>,
    frames: Option<u32>,
    width: Option<u32>,
    height: Option<u32>,
    fps: Option<u32>,
    seed: Option<u64>,
    image: Option<String>,
    sync: bool,
    output: Option<String>,
    open: bool,
) -> anyhow::Result<()> {
    let target = resolve_target(ServeEngine::VideoGen).await?;
    let mut body = serde_json::json!({ "prompt": prompt });
    if let Some(m) = model { body["model"] = m.into(); }
    if let Some(v) = frames { body["num_frames"] = v.into(); }
    if let Some(v) = width { body["width"] = v.into(); }
    if let Some(v) = height { body["height"] = v.into(); }
    if let Some(v) = fps { body["fps"] = v.into(); }
    if let Some(v) = seed { body["seed"] = v.into(); }
    if let Some(v) = image { body["image"] = v.into(); }

    let t0 = std::time::Instant::now();

    if sync {
        println!("video-gen (sync) via {} …", target.describe());
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(VIDEO_GEN_TIMEOUT_SECS))
            .build()?;
        let resp = client
            .post(target.generate_url(ServeEngine::VideoGen))
            .json(&body)
            .send()
            .await?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            anyhow::bail!("generation failed ({status}): {text}");
        }
        let name = header_str(&resp, "x-gen-name");
        let secs = header_str(&resp, "x-gen-seconds");
        let bytes = resp.bytes().await?;
        let path = save_artifact(&bytes, &output, name.as_deref(), "mp4")?;
        println!(
            "✓ {path}  ({:.1} MB, {}s gen)",
            bytes.len() as f64 / 1e6,
            secs.unwrap_or_else(|| "?".into())
        );
        maybe_open(&path, open);
        return Ok(());
    }

    // Async job flow: submit → poll → download artifact.
    println!("video-gen (async job) via {} …", target.describe());
    let client = reqwest::Client::builder().timeout(Duration::from_secs(30)).build()?;
    let submit: serde_json::Value = client
        .post(target.jobs_url())
        .json(&body)
        .send()
        .await?
        .json()
        .await?;
    let Some(job_id) = submit["id"].as_str().map(String::from) else {
        anyhow::bail!("job submit failed: {submit}");
    };
    println!("job {job_id} queued — polling …");

    let mut last_status = String::new();
    let result = loop {
        tokio::time::sleep(Duration::from_secs(3)).await;
        let job: serde_json::Value = client
            .get(target.job_status_url(&job_id))
            .send()
            .await?
            .json()
            .await?;
        let status = job["status"].as_str().unwrap_or("?").to_string();
        if status != last_status {
            println!("  [{:>4.0}s] {status}", t0.elapsed().as_secs_f32());
            last_status = status.clone();
        }
        match status.as_str() {
            "done" => break job["result"].clone(),
            "error" => anyhow::bail!("job failed: {}", job["error"].as_str().unwrap_or("unknown")),
            _ => {}
        }
        if t0.elapsed().as_secs() > VIDEO_GEN_TIMEOUT_SECS {
            anyhow::bail!("gave up after {}s (job {job_id} still {status})", t0.elapsed().as_secs());
        }
    };

    let Some(name) = result["name"].as_str() else {
        anyhow::bail!("job done but no artifact name in result: {result}");
    };
    let dl = reqwest::Client::builder()
        .timeout(Duration::from_secs(300))
        .build()?
        .get(target.artifact_url(ServeEngine::VideoGen, name))
        .send()
        .await?;
    if !dl.status().is_success() {
        anyhow::bail!("artifact download failed: {}", dl.status());
    }
    let bytes = dl.bytes().await?;
    let path = save_artifact(&bytes, &output, Some(name), "mp4")?;
    println!(
        "✓ {path}  ({:.1} MB, {}s gen, {:.0}s total)",
        bytes.len() as f64 / 1e6,
        result["seconds"].as_f64().unwrap_or(0.0),
        t0.elapsed().as_secs_f32()
    );
    maybe_open(&path, open);
    Ok(())
}

/// `asmi media status` — resolve + report both kinds, CLI-side.
pub async fn run_media_status() -> anyhow::Result<()> {
    for (key, engine) in MEDIA_KINDS {
        match resolve_target(engine).await {
            Ok(target) => {
                // For daemon targets, /media/status has the detail; for direct, /health.
                let detail = match &target {
                    Target::Direct(base) => probe_health(base, Duration::from_secs(3)).await,
                    Target::Daemon(base) => {
                        let client = reqwest::Client::builder()
                            .timeout(Duration::from_secs(5))
                            .build()?;
                        match client.get(format!("{base}/media/status")).send().await {
                            Ok(r) => r.json::<serde_json::Value>().await.ok().map(|v| v[key].clone()),
                            Err(_) => None,
                        }
                    }
                };
                match detail {
                    Some(d) => {
                        let models = d["models"]
                            .as_array()
                            .map(|a| a.iter().filter_map(|m| m.as_str()).collect::<Vec<_>>().join(", "))
                            .unwrap_or_default();
                        let busy = d["busy"].as_bool().unwrap_or(false);
                        println!(
                            "{key:<10} ✓ {}  models: [{models}]{}",
                            target.describe(),
                            if busy { "  (busy)" } else { "" }
                        );
                    }
                    None => println!("{key:<10} ✗ resolved {} but health probe failed", target.describe()),
                }
            }
            Err(_) => println!("{key:<10} ✗ unavailable (no local server, config endpoint, or cluster daemon)"),
        }
    }
    Ok(())
}
