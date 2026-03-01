# Asmi Serve: Early Crash Detection

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Detect early MLX server process crashes during health polling and immediately report the actual Python error instead of waiting 60s to time out.

**Architecture:** Modify `poll_health()` in `serve.rs` to accept a `&mut Child` handle and check `child.try_wait()` on each poll iteration. If the process has exited, read the last N lines from the stderr log and return them as the error. This turns a 60s silent timeout into a ~1s informative failure. Also: read stderr from the log file on any health check failure to surface the real error.

**Tech Stack:** Rust, Tokio async, `tokio::process::Child`

---

## Key Files Reference

| File | Purpose |
|------|---------|
| `src/serve.rs` | Server lifecycle — `do_load_inner()`, `poll_health()`, `kill_child()` |
| `src/daemon.rs` | HTTP handlers — `/serve/load`, `/serve/status` |
| `Cargo.toml` | Dependencies (no new ones needed) |
| `tests/daemon_endpoints.rs` | Integration tests |

## Current Bug

When `mlx_lm.server` crashes during startup (e.g., model path doesn't exist), the process exits immediately with a Python traceback. But `poll_health()` has no awareness of the child process — it keeps polling `http://127.0.0.1:{port}/health` for 60 seconds against a dead process. The user sees "timeout waiting for health check" instead of the real error.

**Crash timeline today:**
```
t=0s:   Process spawned, crashes instantly (Python traceback to stderr log)
t=1-60s: poll_health polls dead port, gets connection refused each second
t=60s:  "timeout waiting for health check (mlx_lm)" — useless error
```

**After fix:**
```
t=0s:   Process spawned, crashes instantly
t=1s:   poll_health sees child exited, reads stderr log
t=1s:   "server exited during startup: HFValidationError: Repo id must be..." — actual error
```

---

## Task 1: Change `poll_health` to accept a child handle and detect early exit

**Files:**
- Modify: `src/serve.rs:419-440` (poll_health function)
- Modify: `src/serve.rs:346` (call site)

**Step 1: Change `poll_health` signature**

The function currently takes `(client, port, endpoints, timeout_secs)`. Add a `child: &mut tokio::process::Child` parameter and the `log_path` so it can read stderr on crash.

Replace the `poll_health` function (lines 419–440) with:

```rust
/// Poll health endpoints until one returns 200, the child process exits, or timeout.
/// Returns Ok(true) if healthy, Ok(false) if timed out, Err(msg) if process crashed.
async fn poll_health(
    client: &reqwest::Client,
    port: u16,
    endpoints: &[&str],
    timeout_secs: u64,
    child: &mut tokio::process::Child,
    log_path: &str,
) -> Result<bool, String> {
    let deadline = tokio::time::Instant::now() + std::time::Duration::from_secs(timeout_secs);
    loop {
        // Check if child process has exited (non-blocking)
        match child.try_wait() {
            Ok(Some(status)) => {
                // Process exited — read the log for the real error
                let detail = read_log_tail(log_path, 15).await;
                return Err(format!(
                    "server exited during startup (exit {}): {}",
                    status.code().map(|c| c.to_string()).unwrap_or_else(|| "signal".into()),
                    detail,
                ));
            }
            Ok(None) => {} // still running — continue polling
            Err(_) => {}   // can't check — continue polling
        }

        for ep in endpoints {
            let url = format!("http://127.0.0.1:{port}{ep}");
            if let Ok(resp) = client.get(&url).send().await {
                if resp.status().is_success() {
                    return Ok(true);
                }
            }
        }
        if tokio::time::Instant::now() >= deadline {
            // Timed out — also read log for any hints
            let detail = read_log_tail(log_path, 10).await;
            if detail.is_empty() {
                return Ok(false);
            }
            return Err(format!("timeout waiting for health check — log: {detail}"));
        }
        tokio::time::sleep(std::time::Duration::from_secs(1)).await;
    }
}

/// Read the last N lines from a log file (best-effort).
async fn read_log_tail(path: &str, lines: usize) -> String {
    match tokio::fs::read_to_string(path).await {
        Ok(content) => {
            let tail: Vec<&str> = content.lines().rev().take(lines).collect();
            let mut tail = tail.into_iter().rev().collect::<Vec<_>>();
            // Find the most useful line: last Python exception or traceback line
            let useful = tail.iter().find(|l| {
                l.contains("Error:") || l.contains("Exception:") || l.contains("error:")
            });
            if let Some(line) = useful {
                line.trim().to_string()
            } else {
                tail.join("\n").trim().to_string()
            }
        }
        Err(_) => String::new(),
    }
}
```

**Step 2: Update the call site in `do_load_inner`**

Replace lines 341–374 (from `// Poll health endpoints` to the end of the if/else chain) with:

```rust
    // Poll health endpoints (detect early crash via child.try_wait)
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(2))
        .build()?;

    let health_result = poll_health(&client, port, cfg.health_endpoints, 60, &mut child, &log_path).await;

    let mut s = inner.write().await;
    match health_result {
        Ok(true) if verify_port_owner(child_pid, port).await => {
            s.pid = Some(child_pid);
            s.child = Some(child);
            s.engine = engine;
            s.backend = backend;

            if is_bare {
                s.model = None;
                s.state = ServeState::Bare;
                tracing::info!(pid = child_pid, port, %engine, "bare server ready");
            } else {
                s.model = req.model_path.clone();
                s.state = ServeState::Ready;
                tracing::info!(model = ?req.model_path, pid = child_pid, port, "server ready");
            }
            persist_state(&s).await;
        }
        Ok(true) => {
            s.state = ServeState::Error;
            s.error = Some(format!(
                "server started but bound to wrong port (not {port})"
            ));
            let _ = child.kill().await;
        }
        Ok(false) => {
            s.state = ServeState::Error;
            s.error = Some(format!("timeout waiting for health check ({engine})"));
            let _ = child.kill().await;
        }
        Err(crash_msg) => {
            s.state = ServeState::Error;
            s.error = Some(crash_msg.clone());
            tracing::error!(%crash_msg, port, "server process crashed during startup");
            // Child already exited — no need to kill, but clean up handle
            let _ = child.try_wait();
        }
    }
```

**Step 3: Verify it compiles**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check 2>&1 | tail -10`
Expected: No errors

**Step 4: Commit**

```bash
git add src/serve.rs
git commit -m "fix(serve): detect early process crash during health polling — report real error"
```

---

## Task 2: Truncate the log file before each new load

**Files:**
- Modify: `src/serve.rs:316-321` (log file open)

Currently logs are opened with `append(true)`, which means the log file accumulates across restarts. After a crash, `read_log_tail` might read stale output from a previous successful run.

**Step 1: Truncate on new load**

Replace the log file open block (lines 316–321):

```rust
    // Spawn
    let log_path = format!("/tmp/r1o-mlx-server-{port}.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)?;
    let log_stderr = log_file.try_clone()?;
```

With:

```rust
    // Spawn — truncate log so read_log_tail reads only this run's output
    let log_path = format!("/tmp/r1o-mlx-server-{port}.log");
    let log_file = std::fs::OpenOptions::new()
        .create(true)
        .write(true)
        .truncate(true)
        .open(&log_path)?;
    let log_stderr = log_file.try_clone()?;
```

**Step 2: Verify it compiles**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo check 2>&1 | tail -10`
Expected: No errors

**Step 3: Commit**

```bash
git add src/serve.rs
git commit -m "fix(serve): truncate server log before each load for clean error capture"
```

---

## Task 3: Build, deploy, and test

**Files:** None (manual testing)

**Step 1: Build release binary**

Run: `cd /Users/ma/Projects/Personal/apple-smi && cargo build --release 2>&1 | tail -5`
Expected: Successful build

**Step 2: Deploy to m3u2**

```bash
# Copy binary
scp target/release/asmi m3u2:/tmp/asmi-new

# Sign it (macOS requirement)
ssh m3u2 'codesign --force --sign - /tmp/asmi-new'

# Stop existing daemon, swap binary, restart
ssh m3u2 'launchctl bootout gui/$(id -u) ~/Library/LaunchAgents/com.asmi.daemon.plist 2>/dev/null; sleep 1; cp /tmp/asmi-new /usr/local/bin/asmi && launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.asmi.daemon.plist'
```

**Step 3: Wait for daemon to start**

```bash
sleep 3
curl -s http://m3u2:9090/health | python3 -m json.tool
```
Expected: `{"status": "ok", ...}`

**Step 4: Test with bad model path (should fail fast)**

```bash
curl -s -X POST http://m3u2:9090/serve/load \
  -H 'Content-Type: application/json' \
  -d '{"model_path": "/Users/ma/Models/DOES-NOT-EXIST/", "engine": "mlx_lm"}' | python3 -m json.tool
```

Wait 3s (not 60s), then check status:
```bash
sleep 3
curl -s http://m3u2:9090/serve/status | python3 -m json.tool
```

Expected: State = `error` with an error message containing the actual Python error (e.g., "HFValidationError" or "FileNotFoundError"), NOT "timeout waiting for health check".

**Step 5: Test with valid model path (should still work)**

```bash
curl -s -X POST http://m3u2:9090/serve/load \
  -H 'Content-Type: application/json' \
  -d '{"model_path": "/Users/ma/Models/Qwen3-8B-4bit/", "engine": "mlx_lm"}' | python3 -m json.tool
```

Poll until ready:
```bash
for i in $(seq 1 30); do
  sleep 2
  state=$(curl -s http://m3u2:9090/serve/status | python3 -c "import sys,json; print(json.load(sys.stdin)['state'])")
  echo "$i: $state"
  if [ "$state" = "ready" ]; then break; fi
done
```

Expected: Reaches `ready` within 30s.

**Step 6: Commit any fixups from testing**

```bash
cd /Users/ma/Projects/Personal/apple-smi
git add -A
git commit -m "fix(serve): early crash detection tested and verified"
```

---

## Summary

| Task | What | Files | Impact |
|------|------|-------|--------|
| 1 | Early crash detection in `poll_health` + `read_log_tail` | `serve.rs` | 60s timeout → ~1s real error |
| 2 | Truncate log on new load | `serve.rs` | Clean error capture per run |
| 3 | Build, deploy, test on m3u2 | Manual | Verify fix works |

**Total change: ~60 lines modified in `serve.rs`. No new dependencies. No API changes.**

The `poll_health` → `Result<bool, String>` change is internal — the HTTP API response (`ServeStatus.error`) already carries the error string, so the web deploy dialog will automatically show the real error.
