//! HuggingFace model download manager.
//!
//! Spawns `huggingface-cli download` as a child process per job, parses progress
//! from its stderr/stdout, and surfaces job state via REST + SSE.
//!
//! Endpoints (registered in `daemon.rs`):
//! * `POST /models/download`                     start a job, return `{job_id}`
//! * `GET  /models/download/{job_id}`            one-shot snapshot
//! * `GET  /models/download/{job_id}/progress`   SSE stream until terminal
//! * `GET  /models/downloads`                    all jobs (in-flight + recent)
//!
//! No persistence. Jobs live in an in-memory `Arc<RwLock<HashMap>>`.

use axum::{
    extract::{Path, State},
    http::StatusCode,
    response::{
        sse::{Event, KeepAlive, Sse},
        IntoResponse, Json, Response,
    },
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::process::Stdio;
use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::process::Command;
use tokio::sync::{broadcast, RwLock};
use uuid::Uuid;

const PROGRESS_CHANNEL_CAPACITY: usize = 256;
/// Keep completed/failed jobs visible in `/models/downloads` for this long.
const TERMINAL_RETENTION: Duration = Duration::from_secs(60 * 60);

// ---------------------------------------------------------------------------
// Public types
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "lowercase")]
pub enum JobState {
    Queued,
    Running,
    Completed,
    Failed { error: String },
}

impl JobState {
    fn is_terminal(&self) -> bool {
        matches!(self, JobState::Completed | JobState::Failed { .. })
    }
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct ProgressEvent {
    pub job_id: String,
    pub state: String,
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub current_file: Option<String>,
    pub percent: Option<f64>,
    pub speed_bytes_per_sec: Option<u64>,
    /// Absolute filesystem path of the downloaded model. `None` until the
    /// job reaches `Completed` state. Lets clients skip the fragile
    /// `hf_id.replace('/', "--")` inference.
    pub local_path: Option<String>,
}

#[derive(Debug, Clone, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct JobSnapshot {
    pub id: String,
    pub hf_id: String,
    pub target_dir: String,
    pub started_at_ms: u128,
    pub finished_at_ms: Option<u128>,
    pub state: JobState,
    pub bytes_downloaded: u64,
    pub total_bytes: Option<u64>,
    pub current_file: Option<String>,
    pub percent: Option<f64>,
    pub speed_bytes_per_sec: Option<u64>,
    /// Absolute filesystem path of the downloaded model. `None` until the
    /// job reaches `Completed` state.
    pub local_path: Option<String>,
}

// ---------------------------------------------------------------------------
// Internal job representation
// ---------------------------------------------------------------------------

struct DownloadJob {
    id: String,
    hf_id: String,
    target_dir: String,
    started_at: Instant,
    started_at_ms: u128,
    finished_at_ms: Option<u128>,
    state: JobState,
    bytes_downloaded: u64,
    total_bytes: Option<u64>,
    current_file: Option<String>,
    percent: Option<f64>,
    speed_bytes_per_sec: Option<u64>,
    /// Set to `Some(target_dir)` only when the job has completed
    /// successfully. Surfaced to clients as `localPath` so they don't
    /// have to guess the path from `hf_id`.
    local_path: Option<String>,
    tx: broadcast::Sender<ProgressEvent>,
}

impl DownloadJob {
    fn snapshot(&self) -> JobSnapshot {
        JobSnapshot {
            id: self.id.clone(),
            hf_id: self.hf_id.clone(),
            target_dir: self.target_dir.clone(),
            started_at_ms: self.started_at_ms,
            finished_at_ms: self.finished_at_ms,
            state: self.state.clone(),
            bytes_downloaded: self.bytes_downloaded,
            total_bytes: self.total_bytes,
            current_file: self.current_file.clone(),
            percent: self.percent,
            speed_bytes_per_sec: self.speed_bytes_per_sec,
            local_path: self.local_path.clone(),
        }
    }

    fn to_event(&self) -> ProgressEvent {
        let state_str = match &self.state {
            JobState::Queued => "Queued",
            JobState::Running => "Running",
            JobState::Completed => "Completed",
            JobState::Failed { .. } => "Failed",
        };
        ProgressEvent {
            job_id: self.id.clone(),
            state: state_str.to_string(),
            bytes_downloaded: self.bytes_downloaded,
            total_bytes: self.total_bytes,
            current_file: self.current_file.clone(),
            percent: self.percent,
            speed_bytes_per_sec: self.speed_bytes_per_sec,
            local_path: self.local_path.clone(),
        }
    }
}

// ---------------------------------------------------------------------------
// Registry (process-wide, lazy-init)
// ---------------------------------------------------------------------------

type Registry = Arc<RwLock<HashMap<String, DownloadJob>>>;

fn registry() -> Registry {
    static REG: OnceLock<Registry> = OnceLock::new();
    REG.get_or_init(|| Arc::new(RwLock::new(HashMap::new())))
        .clone()
}

// ---------------------------------------------------------------------------
// Validation
// ---------------------------------------------------------------------------

fn validate_hf_id(id: &str) -> Result<(), String> {
    static RE: OnceLock<regex::Regex> = OnceLock::new();
    let re = RE.get_or_init(|| {
        regex::Regex::new(r"^[\w\-\.]+/[\w\-\.]+$").expect("valid regex")
    });
    if re.is_match(id) {
        Ok(())
    } else {
        Err(format!("Invalid model ID format: {id}"))
    }
}

fn validate_target_dir(dir: &str) -> Result<(), String> {
    if dir.contains("..") || dir.contains(';') || dir.contains('|') || dir.contains('`') {
        Err("Invalid target directory".into())
    } else {
        Ok(())
    }
}

/// Expand a leading `~` against `$HOME`. Anything else passes through.
fn expand_tilde(p: &str) -> String {
    if let Some(rest) = p.strip_prefix("~/") {
        if let Some(home) = dirs::home_dir() {
            return home.join(rest).to_string_lossy().to_string();
        }
    } else if p == "~" {
        if let Some(home) = dirs::home_dir() {
            return home.to_string_lossy().to_string();
        }
    }
    p.to_string()
}

// ---------------------------------------------------------------------------
// Progress parser
// ---------------------------------------------------------------------------

/// Parse a single line of `huggingface-cli` output. Updates job fields in
/// place. Returns `true` if the line carried any progress info worth
/// publishing.
///
/// The CLI emits two relevant patterns:
///
/// * `"Downloading 'model-00001.safetensors'"` (or similar)
/// * tqdm bar fragments like:
///   `model-00001.safetensors: 18%|██▍ | 1.23G/6.79G [00:34<02:31, 36.5MB/s]`
///   We pull the byte counts + speed out of the bracketed segment.
fn parse_progress_line(line: &str, job: &mut DownloadJob) -> bool {
    let mut updated = false;

    // current_file: "Downloading <something>" or "filename: ..." prefix.
    if let Some(rest) = line.strip_prefix("Downloading ") {
        let trimmed = rest.trim().trim_matches(|c| c == '\'' || c == '"');
        if !trimmed.is_empty() {
            job.current_file = Some(trimmed.to_string());
            updated = true;
        }
    } else if let Some(idx) = line.find(": ") {
        // Heuristic: tqdm prepends the filename before ": ".
        let prefix = &line[..idx];
        // A filename heuristic — it's "short" and contains no whitespace,
        // and is followed by a tqdm-style percent.
        if !prefix.is_empty()
            && !prefix.contains(char::is_whitespace)
            && line[idx + 2..].contains('%')
        {
            job.current_file = Some(prefix.to_string());
            updated = true;
        }
    }

    // percent: "<n>%"
    if let Some(pct) = parse_percent(line) {
        job.percent = Some(pct);
        updated = true;
    }

    // counts + speed: "1.23G/6.79G [00:34<02:31, 36.5MB/s]"
    if let Some((cur, total, speed)) = parse_bracketed(line) {
        if let Some(c) = cur {
            job.bytes_downloaded = c;
        }
        if total.is_some() {
            job.total_bytes = total;
        }
        if speed.is_some() {
            job.speed_bytes_per_sec = speed;
        }
        updated = true;
    }

    updated
}

fn parse_percent(line: &str) -> Option<f64> {
    // Find the right-most '%' that has a number directly before it.
    let bytes = line.as_bytes();
    let mut i = bytes.len();
    while i > 0 {
        if bytes[i - 1] == b'%' {
            // Walk back over digits and an optional dot.
            let mut start = i - 1;
            while start > 0 {
                let c = bytes[start - 1];
                if c.is_ascii_digit() || c == b'.' {
                    start -= 1;
                } else {
                    break;
                }
            }
            if start < i - 1 {
                if let Ok(v) = line[start..i - 1].parse::<f64>() {
                    if (0.0..=100.0).contains(&v) {
                        return Some(v);
                    }
                }
            }
        }
        i -= 1;
    }
    None
}

/// Pull `cur/total` and `speed` out of a tqdm bracket segment.
/// Returns `(cur_bytes, total_bytes, speed_bytes_per_sec)`.
fn parse_bracketed(line: &str) -> Option<(Option<u64>, Option<u64>, Option<u64>)> {
    let lb = line.rfind('[')?;
    let rb = line[lb..].find(']').map(|i| lb + i)?;
    let inner = &line[lb + 1..rb];

    // Look for a slash before the bracket: "1.23G/6.79G".
    let pre = &line[..lb];
    let (cur, total) = pre
        .rsplit(|c: char| c.is_whitespace())
        .find(|tok| tok.contains('/') && !tok.is_empty())
        .map(|tok| {
            let mut split = tok.splitn(2, '/');
            let a = split.next().and_then(parse_size_token);
            let b = split.next().and_then(parse_size_token);
            (a, b)
        })
        .unwrap_or((None, None));

    // Speed inside the bracket: e.g. "36.5MB/s".
    let speed = inner
        .split([',', ' '])
        .map(str::trim)
        .find(|tok| tok.ends_with("/s"))
        .and_then(|tok| {
            let no_suffix = tok.trim_end_matches("/s");
            parse_size_token(no_suffix)
        });

    if cur.is_none() && total.is_none() && speed.is_none() {
        None
    } else {
        Some((cur, total, speed))
    }
}

/// Parse strings like "1.23G", "456M", "789K", "100B" → bytes.
fn parse_size_token(tok: &str) -> Option<u64> {
    let tok = tok.trim();
    if tok.is_empty() {
        return None;
    }
    // Strip optional trailing "B".
    let trimmed = tok.strip_suffix('B').or_else(|| tok.strip_suffix('b')).unwrap_or(tok);
    let (num_str, mult) = match trimmed.chars().last()? {
        'K' | 'k' => (&trimmed[..trimmed.len() - 1], 1024u64),
        'M' | 'm' => (&trimmed[..trimmed.len() - 1], 1024u64 * 1024),
        'G' | 'g' => (&trimmed[..trimmed.len() - 1], 1024u64 * 1024 * 1024),
        'T' | 't' => (&trimmed[..trimmed.len() - 1], 1024u64 * 1024 * 1024 * 1024),
        c if c.is_ascii_digit() || c == '.' => (trimmed, 1u64),
        _ => return None,
    };
    let n: f64 = num_str.parse().ok()?;
    Some((n * mult as f64) as u64)
}

// ---------------------------------------------------------------------------
// Spawn / drive
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize)]
pub struct DownloadRequest {
    pub hf_id: String,
    #[serde(default)]
    pub target_dir: Option<String>,
}

#[derive(Debug, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct DownloadStartResponse {
    pub job_id: String,
    pub status: String,
    pub target_dir: String,
}

/// POST /models/download
pub async fn start_handler(
    State(_state): State<crate::daemon::AppState>,
    Json(body): Json<DownloadRequest>,
) -> Response {
    let hf_id = body.hf_id.trim().to_string();
    if let Err(e) = validate_hf_id(&hf_id) {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e}))).into_response();
    }
    let raw_target = body.target_dir.unwrap_or_else(|| "~/Models".to_string());
    if let Err(e) = validate_target_dir(&raw_target) {
        return (StatusCode::BAD_REQUEST, Json(serde_json::json!({"error": e}))).into_response();
    }

    let target_base = expand_tilde(&raw_target);
    let model_dir_name = hf_id.replace('/', "--");
    let full_path = format!("{}/{}", target_base.trim_end_matches('/'), model_dir_name);

    let job_id = Uuid::new_v4().to_string();
    let started_at = Instant::now();
    let started_at_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    let (tx, _) = broadcast::channel::<ProgressEvent>(PROGRESS_CHANNEL_CAPACITY);

    let job = DownloadJob {
        id: job_id.clone(),
        hf_id: hf_id.clone(),
        target_dir: full_path.clone(),
        started_at,
        started_at_ms,
        finished_at_ms: None,
        state: JobState::Queued,
        bytes_downloaded: 0,
        total_bytes: None,
        current_file: None,
        percent: None,
        speed_bytes_per_sec: None,
        local_path: None,
        tx: tx.clone(),
    };

    let reg = registry();
    reg.write().await.insert(job_id.clone(), job);

    // Drive the job in the background.
    let drive_id = job_id.clone();
    let drive_target = full_path.clone();
    let drive_hf = hf_id.clone();
    tokio::spawn(async move {
        run_download(drive_id, drive_hf, drive_target).await;
    });

    let body = DownloadStartResponse {
        job_id,
        status: "started".into(),
        target_dir: full_path,
    };
    (StatusCode::ACCEPTED, Json(body)).into_response()
}

async fn run_download(job_id: String, hf_id: String, target_dir: String) {
    let reg = registry();

    // Ensure parent dir exists.
    if let Some(parent) = std::path::Path::new(&target_dir).parent() {
        if let Err(e) = tokio::fs::create_dir_all(parent).await {
            mark_failed(&reg, &job_id, format!("create target parent failed: {e}")).await;
            return;
        }
    }

    // Build the command.
    //
    // huggingface_hub 1.x ships the binary as `hf` and deprecated `huggingface-cli`.
    // We try `huggingface-cli` first (still common on older installs), then fall
    // back to `hf download`. Both accept `--local-dir <path> <repo_id>`.
    let (program, mut args) = resolve_hf_cli().await;
    args.extend([
        "download".to_string(),
        "--local-dir".to_string(),
        target_dir.clone(),
        hf_id.clone(),
    ]);

    let mut cmd = Command::new(&program);
    cmd.args(&args)
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .kill_on_drop(true);

    // Augment PATH so the child can find python/git/hf if it shells out.
    // (launchd-spawned asmi has a minimal PATH by default.)
    let path = std::env::var("PATH").unwrap_or_default();
    let extra = "/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin";
    let combined = if path.contains("/opt/homebrew/bin") {
        path
    } else {
        format!("{extra}:{path}")
    };
    cmd.env("PATH", combined);

    // Pass token via env if one is configured.
    if let Some(token) = read_hf_token().await {
        cmd.env("HF_TOKEN", &token);
        cmd.env("HUGGING_FACE_HUB_TOKEN", token);
    }

    tracing::info!(
        job_id = %job_id,
        hf_id = %hf_id,
        target_dir = %target_dir,
        "starting huggingface-cli download"
    );

    let mut child = match cmd.spawn() {
        Ok(c) => c,
        Err(e) => {
            mark_failed(&reg, &job_id, format!("spawn huggingface-cli failed: {e}")).await;
            return;
        }
    };

    // Mark Running and broadcast initial event.
    {
        let mut guard = reg.write().await;
        if let Some(job) = guard.get_mut(&job_id) {
            job.state = JobState::Running;
            let _ = job.tx.send(job.to_event());
        }
    }

    let stdout = child.stdout.take();
    let stderr = child.stderr.take();

    // Spawn readers. Each reader updates the job + broadcasts.
    let reg_a = reg.clone();
    let job_a = job_id.clone();
    let stdout_task = stdout.map(|s| {
        tokio::spawn(async move {
            let mut reader = BufReader::new(s).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                update_from_line(&reg_a, &job_a, &line).await;
            }
        })
    });

    let reg_b = reg.clone();
    let job_b = job_id.clone();
    let stderr_task = stderr.map(|s| {
        tokio::spawn(async move {
            let mut reader = BufReader::new(s).lines();
            while let Ok(Some(line)) = reader.next_line().await {
                update_from_line(&reg_b, &job_b, &line).await;
            }
        })
    });

    let exit = child.wait().await;

    if let Some(t) = stdout_task {
        let _ = t.await;
    }
    if let Some(t) = stderr_task {
        let _ = t.await;
    }

    let now_ms = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis())
        .unwrap_or(0);

    match exit {
        Ok(status) if status.success() => {
            let mut guard = reg.write().await;
            if let Some(job) = guard.get_mut(&job_id) {
                job.state = JobState::Completed;
                job.finished_at_ms = Some(now_ms);
                if job.percent.is_none() {
                    job.percent = Some(100.0);
                }
                if let Some(total) = job.total_bytes {
                    job.bytes_downloaded = total;
                }
                // Expose the canonical filesystem path so clients don't
                // have to reconstruct it from hf_id.
                job.local_path = Some(job.target_dir.clone());
                let _ = job.tx.send(job.to_event());
                tracing::info!(
                    job_id = %job_id,
                    elapsed_secs = job.started_at.elapsed().as_secs(),
                    local_path = %job.target_dir,
                    "download complete"
                );
            }
        }
        Ok(status) => {
            mark_failed(
                &reg,
                &job_id,
                format!("huggingface-cli exited with status {status}"),
            )
            .await;
        }
        Err(e) => {
            mark_failed(&reg, &job_id, format!("wait failed: {e}")).await;
        }
    }

    // Schedule retention sweep so completed/failed jobs don't accumulate forever.
    tokio::spawn(async move {
        tokio::time::sleep(TERMINAL_RETENTION).await;
        let reg = registry();
        let mut guard = reg.write().await;
        let should_remove = guard
            .get(&job_id)
            .map(|j| j.state.is_terminal() && j.started_at.elapsed() >= TERMINAL_RETENTION)
            .unwrap_or(false);
        if should_remove {
            guard.remove(&job_id);
        }
    });
}

async fn update_from_line(reg: &Registry, job_id: &str, line: &str) {
    let mut guard = reg.write().await;
    let Some(job) = guard.get_mut(job_id) else {
        return;
    };
    if parse_progress_line(line, job) {
        let _ = job.tx.send(job.to_event());
    }
}

async fn mark_failed(reg: &Registry, job_id: &str, error: String) {
    let mut guard = reg.write().await;
    if let Some(job) = guard.get_mut(job_id) {
        let now_ms = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_millis())
            .unwrap_or(0);
        job.state = JobState::Failed { error: error.clone() };
        job.finished_at_ms = Some(now_ms);
        let _ = job.tx.send(job.to_event());
        tracing::warn!(job_id = %job_id, %error, "download failed");
    }
}

async fn read_hf_token() -> Option<String> {
    let home = dirs::home_dir()?;
    let path = home.join(".cache").join("huggingface").join("token");
    let raw = tokio::fs::read_to_string(&path).await.ok()?;
    let trimmed = raw.trim().to_string();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed)
    }
}

/// Resolve the HuggingFace download CLI.
///
/// huggingface_hub 1.0+ renamed the binary to `hf` and the legacy
/// `huggingface-cli` shim is going away. We try absolute paths in common
/// install locations first (so we work under launchd's stripped PATH), then
/// fall back to whatever the runtime PATH resolves.
///
/// Returns `(program, leading_args)`. The caller appends `download <flags> <repo>`.
async fn resolve_hf_cli() -> (String, Vec<String>) {
    // Candidates in priority order. The first existing executable wins.
    const CANDIDATES: &[&str] = &[
        "/opt/homebrew/bin/huggingface-cli",
        "/usr/local/bin/huggingface-cli",
        "/opt/homebrew/bin/hf",
        "/usr/local/bin/hf",
    ];

    for c in CANDIDATES {
        if tokio::fs::metadata(c).await.is_ok() {
            return (c.to_string(), vec![]);
        }
    }

    // Last resort — let the runtime PATH resolve. Prefer `huggingface-cli` for
    // backward compat; if that's missing the launch will fail and we'll log a
    // helpful error.
    ("huggingface-cli".to_string(), vec![])
}

// ---------------------------------------------------------------------------
// One-shot snapshot handler
// ---------------------------------------------------------------------------

/// GET /models/download/{job_id}
pub async fn snapshot_handler(
    State(_state): State<crate::daemon::AppState>,
    Path(job_id): Path<String>,
) -> Response {
    let reg = registry();
    let guard = reg.read().await;
    match guard.get(&job_id) {
        Some(job) => Json(job.snapshot()).into_response(),
        None => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"error": format!("unknown job_id: {job_id}")})),
        )
            .into_response(),
    }
}

// ---------------------------------------------------------------------------
// SSE progress handler
// ---------------------------------------------------------------------------

/// GET /models/download/{job_id}/progress
pub async fn progress_sse_handler(
    State(_state): State<crate::daemon::AppState>,
    Path(job_id): Path<String>,
) -> Response {
    use futures::stream::StreamExt;

    let reg = registry();
    let (rx, initial, already_terminal) = {
        let guard = reg.read().await;
        match guard.get(&job_id) {
            Some(job) => {
                let initial = job.to_event();
                (job.tx.subscribe(), initial, job.state.is_terminal())
            }
            None => {
                return (
                    StatusCode::NOT_FOUND,
                    Json(serde_json::json!({"error": format!("unknown job_id: {job_id}")})),
                )
                    .into_response();
            }
        }
    };

    let initial_terminal = is_terminal_event(&initial);

    // If the job is already done, emit one event and close.
    if already_terminal {
        let stream = futures::stream::once(async move {
            let json = serde_json::to_string(&initial).unwrap_or_else(|_| "{}".into());
            Ok::<_, std::convert::Infallible>(Event::default().data(json))
        });
        return Sse::new(stream)
            .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("ping"))
            .into_response();
    }

    // Live path: emit the current snapshot first, then forward broadcast events
    // until we observe a terminal one.
    let initial_event = futures::stream::once(async move {
        let json = serde_json::to_string(&initial).unwrap_or_else(|_| "{}".into());
        Ok::<_, std::convert::Infallible>(Event::default().data(json))
    });

    // (event_string, is_terminal)
    let live = tokio_stream::wrappers::BroadcastStream::new(rx).filter_map(|res| async move {
        match res {
            Ok(ev) => {
                let terminal = is_terminal_event(&ev);
                let json = serde_json::to_string(&ev).unwrap_or_else(|_| "{}".into());
                Some((json, terminal))
            }
            Err(_) => None,
        }
    });

    // Stop after the first terminal event has been forwarded.
    let stop_after_terminal = live.scan(initial_terminal, |done, (json, terminal)| {
        if *done {
            // Already saw a terminal event — close the stream.
            return std::future::ready(None);
        }
        if terminal {
            *done = true;
        }
        std::future::ready(Some(Ok::<_, std::convert::Infallible>(
            Event::default().data(json),
        )))
    });

    let stream = initial_event.chain(stop_after_terminal);

    Sse::new(stream)
        .keep_alive(KeepAlive::new().interval(Duration::from_secs(15)).text("ping"))
        .into_response()
}

fn is_terminal_event(ev: &ProgressEvent) -> bool {
    matches!(ev.state.as_str(), "Completed" | "Failed")
}

// ---------------------------------------------------------------------------
// List handler
// ---------------------------------------------------------------------------

/// GET /models/downloads
pub async fn list_handler(
    State(_state): State<crate::daemon::AppState>,
) -> Json<serde_json::Value> {
    let reg = registry();
    let guard = reg.read().await;
    let mut snapshots: Vec<JobSnapshot> = guard.values().map(|j| j.snapshot()).collect();
    snapshots.sort_by(|a, b| b.started_at_ms.cmp(&a.started_at_ms));
    Json(serde_json::json!({
        "jobs": snapshots,
        "total": snapshots.len(),
    }))
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validates_hf_id() {
        assert!(validate_hf_id("Org/Model").is_ok());
        assert!(validate_hf_id("mlx-community/Qwen3-8B-4bit").is_ok());
        assert!(validate_hf_id("a.b/c.d-e_f").is_ok());
        assert!(validate_hf_id("badformat").is_err());
        assert!(validate_hf_id("a/b/c").is_err());
        assert!(validate_hf_id("../foo/bar").is_err());
        assert!(validate_hf_id("foo/bar;rm").is_err());
    }

    #[test]
    fn rejects_unsafe_target_dir() {
        assert!(validate_target_dir("~/Models").is_ok());
        assert!(validate_target_dir("/tmp/models").is_ok());
        assert!(validate_target_dir("../etc").is_err());
        assert!(validate_target_dir("/tmp;rm").is_err());
        assert!(validate_target_dir("/tmp|cat").is_err());
        assert!(validate_target_dir("/tmp`whoami`").is_err());
    }

    #[test]
    fn parses_size_tokens() {
        assert_eq!(parse_size_token("100"), Some(100));
        assert_eq!(parse_size_token("1K"), Some(1024));
        assert_eq!(parse_size_token("1KB"), Some(1024));
        assert_eq!(parse_size_token("1M"), Some(1024 * 1024));
        assert_eq!(parse_size_token("1.5G"), Some((1.5 * (1024.0 * 1024.0 * 1024.0)) as u64));
        assert_eq!(parse_size_token(""), None);
    }

    #[test]
    fn parses_percent_from_tqdm_line() {
        let line = "model.safetensors:  18%|████| 1.23G/6.79G [00:34<02:31, 36.5MB/s]";
        assert_eq!(parse_percent(line), Some(18.0));
    }

    #[test]
    fn parses_bracketed_segment() {
        let line = "model.safetensors: 18%|████| 1.23G/6.79G [00:34<02:31, 36.5MB/s]";
        let (cur, total, speed) = parse_bracketed(line).expect("should parse");
        assert!(cur.unwrap() > 1_000_000_000);
        assert!(total.unwrap() > 6_000_000_000);
        let s = speed.expect("speed");
        // 36.5 MiB/s in bytes/s
        assert!(s > 30_000_000 && s < 50_000_000, "speed={s}");
    }

    #[test]
    fn parse_progress_line_updates_job() {
        let (tx, _) = broadcast::channel::<ProgressEvent>(8);
        let mut job = DownloadJob {
            id: "x".into(),
            hf_id: "a/b".into(),
            target_dir: "/tmp/x".into(),
            started_at: Instant::now(),
            started_at_ms: 0,
            finished_at_ms: None,
            state: JobState::Running,
            bytes_downloaded: 0,
            total_bytes: None,
            current_file: None,
            percent: None,
            speed_bytes_per_sec: None,
            local_path: None,
            tx,
        };
        let line = "model-00001.safetensors: 50%|██████| 3.40G/6.79G [00:30<00:30, 100MB/s]";
        let updated = parse_progress_line(line, &mut job);
        assert!(updated);
        assert_eq!(job.percent, Some(50.0));
        assert!(job.bytes_downloaded > 3_000_000_000);
        assert!(job.total_bytes.unwrap() > 6_000_000_000);
        assert!(job.speed_bytes_per_sec.unwrap() > 90_000_000);
        assert_eq!(job.current_file.as_deref(), Some("model-00001.safetensors"));
    }

    #[test]
    fn expand_tilde_replaces_home() {
        let home = dirs::home_dir().unwrap();
        let expanded = expand_tilde("~/Models");
        assert!(expanded.starts_with(home.to_str().unwrap()));
        assert!(expanded.ends_with("Models"));

        let unchanged = expand_tilde("/tmp/foo");
        assert_eq!(unchanged, "/tmp/foo");
    }

    /// Helper: build a `DownloadJob` for tests in a known state.
    fn make_job(state: JobState) -> DownloadJob {
        let (tx, _) = broadcast::channel::<ProgressEvent>(8);
        DownloadJob {
            id: "job-1".into(),
            hf_id: "mlx-community/SmolLM2-360M-Instruct".into(),
            target_dir: "/Users/ma/Models/mlx-community--SmolLM2-360M-Instruct".into(),
            started_at: Instant::now(),
            started_at_ms: 0,
            finished_at_ms: None,
            state,
            bytes_downloaded: 0,
            total_bytes: None,
            current_file: None,
            percent: None,
            speed_bytes_per_sec: None,
            local_path: None,
            tx,
        }
    }

    #[test]
    fn local_path_is_none_while_queued() {
        let job = make_job(JobState::Queued);
        let snap = job.snapshot();
        assert!(snap.local_path.is_none());
        let ev = job.to_event();
        assert!(ev.local_path.is_none());
    }

    #[test]
    fn local_path_is_none_while_running() {
        let job = make_job(JobState::Running);
        let snap = job.snapshot();
        assert!(snap.local_path.is_none());
        let ev = job.to_event();
        assert!(ev.local_path.is_none());
    }

    #[test]
    fn local_path_is_none_while_failed() {
        let job = make_job(JobState::Failed { error: "boom".into() });
        let snap = job.snapshot();
        assert!(snap.local_path.is_none());
        let ev = job.to_event();
        assert!(ev.local_path.is_none());
    }

    #[test]
    fn local_path_set_on_completion() {
        // Simulate the same field assignment the success path performs in
        // `run_download` so any drift is caught here.
        let mut job = make_job(JobState::Running);
        job.state = JobState::Completed;
        job.local_path = Some(job.target_dir.clone());

        let snap = job.snapshot();
        assert_eq!(
            snap.local_path.as_deref(),
            Some("/Users/ma/Models/mlx-community--SmolLM2-360M-Instruct"),
        );
        let ev = job.to_event();
        assert_eq!(
            ev.local_path.as_deref(),
            Some("/Users/ma/Models/mlx-community--SmolLM2-360M-Instruct"),
        );
        assert_eq!(ev.state, "Completed");
    }

    #[test]
    fn local_path_serialises_as_camel_case() {
        let mut job = make_job(JobState::Running);
        job.state = JobState::Completed;
        job.local_path = Some(job.target_dir.clone());

        let snap_json = serde_json::to_string(&job.snapshot()).expect("snapshot json");
        assert!(
            snap_json.contains("\"localPath\""),
            "snapshot must use camelCase localPath: {snap_json}",
        );
        assert!(
            !snap_json.contains("\"local_path\""),
            "snapshot must not leak snake_case local_path: {snap_json}",
        );

        let ev_json = serde_json::to_string(&job.to_event()).expect("event json");
        assert!(
            ev_json.contains("\"localPath\""),
            "event must use camelCase localPath: {ev_json}",
        );
        assert!(
            !ev_json.contains("\"local_path\""),
            "event must not leak snake_case local_path: {ev_json}",
        );

        // While running, the field must still serialise as null (not absent).
        let running = make_job(JobState::Running);
        let running_json = serde_json::to_string(&running.snapshot()).expect("running json");
        assert!(
            running_json.contains("\"localPath\":null"),
            "running snapshot must serialise localPath as null: {running_json}",
        );
    }
}
