//! Detection of launchd-managed processes + disable/enable actions.
//!
//! Two public entry points:
//! - `describe_pid(pid)` / `describe_label(label)` — passive inspection
//! - `disable(label)` / `enable(label)` — explicit actions with protected-label guard
//!
//! All operations target the current user's GUI domain (`gui/$UID`). System-level
//! agents are out of scope because asmi never runs as root.
//!
//! ## Parser notes (macOS 26)
//!
//! `launchctl list` prints `PID\tStatus\tLabel` — PID is `-` when the agent is
//! registered but not currently running. We skip those rows.
//!
//! `launchctl print gui/$UID/<label>` is a human-readable dump. The relevant
//! lines we parse are:
//!
//! ```text
//! 	state = running              // top-level state (running / not running)
//! 	program = /path/to/bin
//! 	properties = keepalive | runatload | inferred program | …
//! 	disabled on keys = ["label"]  // only present when disabled
//! ```
//!
//! `KeepAlive` is *not* printed as an explicit `KeepAlive = true` key-value. It
//! appears as the token `keepalive` in the pipe-delimited `properties` string.
//! Same for `runatload`.
//!
//! When an agent is disabled, `launchctl print` returns an error (because the
//! service has been booted out). The disabled status is reported by
//! `launchctl print-disabled gui/$UID/` which lists `"label" => disabled`.

use asmi_core::{LaunchdInfo, LaunchdState};
use std::collections::HashMap;
use std::time::Duration;

/// Commands shell out to `launchctl` via tokio; hard-capped at 3 s each so a
/// pathological system cannot block an HTTP handler.
const LAUNCHCTL_TIMEOUT: Duration = Duration::from_secs(3);

/// Labels protected from disable/enable — we never tamper with these.
const PROTECTED_PREFIXES: &[&str] = &["com.asmi.", "com.r1o.watchdog"];

#[derive(Debug, thiserror::Error)]
pub enum LaunchdError {
    #[error("protected label: {0}")]
    Protected(String),
    #[error("launchctl timeout")]
    Timeout,
    #[error("launchctl failed: {0}")]
    Io(#[from] std::io::Error),
    #[error("launchctl {cmd} exited {code}: {stderr}")]
    NonZero {
        cmd: &'static str,
        code: i32,
        stderr: String,
    },
}

// ---------------------------------------------------------------------------
// Pure parsers — no I/O so tests can be deterministic via fixtures.
// ---------------------------------------------------------------------------

/// Parse `launchctl list` output into a `PID → label` map.
///
/// Rows with `-` as PID (agent registered but not running) are excluded.
pub(crate) fn parse_launchctl_list(text: &str) -> HashMap<u32, String> {
    text.lines()
        .skip(1) // "PID\tStatus\tLabel" header
        .filter_map(|line| {
            let mut fields = line.split_whitespace();
            let pid_str = fields.next()?;
            let _status = fields.next()?;
            let label = fields.next()?;
            let pid = pid_str.parse::<u32>().ok()?;
            Some((pid, label.to_string()))
        })
        .collect()
}

/// Parse `launchctl print gui/UID/label` output.
///
/// `label` is passed in by the caller because the print output's first line
/// (`gui/501/com.foo = {`) is format-brittle — safer to trust the query label.
pub(crate) fn parse_launchctl_print(label: &str, text: &str) -> LaunchdInfo {
    let mut program: Option<String> = None;
    let mut properties_line: Option<String> = None;
    let mut state_str: Option<String> = None;
    let mut has_disabled_on_keys = false;

    for raw in text.lines() {
        let line = raw.trim();
        // Only match the TOP-LEVEL `state = ...` line (not nested `state = active`
        // under resource coalitions or event subsystems). Top-level lines are
        // indented with a single tab.
        if raw.starts_with("\tstate = ") && state_str.is_none() {
            state_str = Some(line.trim_start_matches("state = ").trim().to_string());
        } else if raw.starts_with("\tprogram = ") && program.is_none() {
            program = Some(line.trim_start_matches("program = ").trim().to_string());
        } else if raw.starts_with("\tproperties = ") && properties_line.is_none() {
            properties_line = Some(line.trim_start_matches("properties = ").trim().to_string());
        } else if raw.starts_with("\tdisabled on keys") {
            has_disabled_on_keys = true;
        }
    }

    // KeepAlive / RunAtLoad are encoded inside the `properties = a | b | c` line
    // as the tokens `keepalive` and `runatload`. If we didn't see a properties
    // line at all, report `None` (unknown) rather than `Some(false)`.
    let (keep_alive, run_at_load) = match properties_line.as_deref() {
        Some(props) => {
            let tokens: Vec<&str> = props.split('|').map(str::trim).collect();
            (
                Some(tokens.iter().any(|t| t.eq_ignore_ascii_case("keepalive"))),
                Some(tokens.iter().any(|t| t.eq_ignore_ascii_case("runatload"))),
            )
        }
        None => (None, None),
    };

    let state = if has_disabled_on_keys {
        LaunchdState::Disabled
    } else {
        match state_str.as_deref() {
            Some("running") => LaunchdState::Running,
            Some("not running") => LaunchdState::Waiting,
            _ => LaunchdState::Waiting,
        }
    };

    LaunchdInfo {
        label: label.to_string(),
        keep_alive,
        run_at_load,
        state,
        program,
    }
}

/// Parse `launchctl print-disabled gui/UID/` output — a block of
/// `"label" => disabled` / `"label" => enabled` lines.
pub(crate) fn parse_print_disabled(text: &str) -> HashMap<String, bool> {
    let mut map = HashMap::new();
    for raw in text.lines() {
        let line = raw.trim();
        // `"com.foo.bar" => disabled`  or  `"com.foo.bar" => enabled`
        let Some(arrow_idx) = line.find("=>") else { continue };
        let lhs = line[..arrow_idx].trim();
        let rhs = line[arrow_idx + 2..].trim();
        if !lhs.starts_with('"') || !lhs.ends_with('"') {
            continue;
        }
        let label = lhs.trim_matches('"').to_string();
        let disabled = rhs.eq_ignore_ascii_case("disabled");
        map.insert(label, disabled);
    }
    map
}

// ---------------------------------------------------------------------------
// Shell-out helpers — each wrapped in a 3 s timeout.
// ---------------------------------------------------------------------------

fn current_uid() -> u32 {
    nix::unistd::getuid().as_raw()
}

async fn run_launchctl(args: &[&str]) -> Result<String, LaunchdError> {
    let fut = tokio::process::Command::new("launchctl")
        .args(args)
        .output();

    let out = tokio::time::timeout(LAUNCHCTL_TIMEOUT, fut)
        .await
        .map_err(|_| LaunchdError::Timeout)?
        .map_err(LaunchdError::Io)?;

    if !out.status.success() {
        let code = out.status.code().unwrap_or(-1);
        let stderr = String::from_utf8_lossy(&out.stderr).trim().to_string();
        // Keep the first arg as the command name for diagnostics
        let cmd = match args.first().copied() {
            Some("list") => "list",
            Some("print") => "print",
            Some("print-disabled") => "print-disabled",
            Some("disable") => "disable",
            Some("enable") => "enable",
            Some("bootstrap") => "bootstrap",
            Some("bootout") => "bootout",
            _ => "launchctl",
        };
        return Err(LaunchdError::NonZero { cmd, code, stderr });
    }

    Ok(String::from_utf8_lossy(&out.stdout).into_owned())
}

// ---------------------------------------------------------------------------
// Public async API.
// ---------------------------------------------------------------------------

/// Describe the launchd agent (if any) that owns the given PID.
/// Returns `None` if `launchctl list` has no matching row or the describe fails.
pub async fn describe_pid(pid: u32) -> Option<LaunchdInfo> {
    let list_out = run_launchctl(&["list"]).await.ok()?;
    let map = parse_launchctl_list(&list_out);
    let label = map.get(&pid)?.clone();
    describe_label(&label).await
}

/// Describe a launchd agent by label. Falls back to `print-disabled` if
/// `print` fails (that's the macOS 26 behavior when the service is disabled
/// and booted out).
pub async fn describe_label(label: &str) -> Option<LaunchdInfo> {
    let uid = current_uid();
    let target = format!("gui/{}/{}", uid, label);

    match run_launchctl(&["print", &target]).await {
        Ok(text) => Some(parse_launchctl_print(label, &text)),
        Err(_) => {
            // Not loaded — check print-disabled to see if the agent is known-but-disabled
            let disabled_out = run_launchctl(&["print-disabled", &format!("gui/{}", uid)])
                .await
                .ok()?;
            let map = parse_print_disabled(&disabled_out);
            let disabled = *map.get(label)?;
            Some(LaunchdInfo {
                label: label.to_string(),
                keep_alive: None,
                run_at_load: None,
                state: if disabled {
                    LaunchdState::Disabled
                } else {
                    LaunchdState::Waiting
                },
                program: None,
            })
        }
    }
}

/// Disable a launchd agent and bootout its running instance.
///
/// `launchctl disable` persists across reboots; `bootout` kills the current
/// process. Both are idempotent — bootout failing when the agent was already
/// unloaded is ignored.
pub async fn disable(label: &str) -> Result<(), LaunchdError> {
    guard_protected(label)?;
    let uid = current_uid();
    let target = format!("gui/{}/{}", uid, label);

    run_launchctl(&["disable", &target]).await?;
    // bootout returns non-zero if the service was not loaded — not a failure for us.
    let _ = run_launchctl(&["bootout", &target]).await;
    Ok(())
}

/// Re-enable a previously-disabled launchd agent and bootstrap its plist.
///
/// Assumes the plist still exists at `~/Library/LaunchAgents/<label>.plist`.
/// If the agent is already loaded, `bootstrap` returns an error which we
/// downgrade (it's the desired steady state).
///
/// Per PR #23 adversarial-critic verdict (2026-05-28, LOW finding #5):
/// previously the bootstrap result was discarded via `let _ = ...`, so
/// genuine plist errors (missing file, malformed XML, invalid program
/// path) surfaced upstream as opaque 504 timeouts after a 15s health
/// wait. Now we inspect the error and only swallow the
/// "already-bootstrapped" steady-state. Anything else is propagated so
/// `hermes_restart_handler` returns a 500 with the real launchctl
/// stderr.
pub async fn enable(label: &str) -> Result<(), LaunchdError> {
    guard_protected(label)?;
    let uid = current_uid();
    let target = format!("gui/{}/{}", uid, label);
    let domain = format!("gui/{}", uid);

    run_launchctl(&["enable", &target]).await?;

    let home = std::env::var("HOME").unwrap_or_default();
    let plist = format!("{}/Library/LaunchAgents/{}.plist", home, label);

    // bootstrap exits non-zero if service is already loaded — that's
    // the steady state we want, so swallow it; but propagate any other
    // failure (missing plist, malformed XML, bad program path) so the
    // caller can return a useful diagnostic instead of timing out at
    // the /health poll.
    match run_launchctl(&["bootstrap", &domain, &plist]).await {
        Ok(_) => Ok(()),
        Err(LaunchdError::NonZero { stderr, .. })
            if stderr.contains("service already bootstrapped")
                || stderr.contains("Service is already loaded")
                || stderr.contains("already loaded") =>
        {
            Ok(())
        }
        Err(e) => Err(e),
    }
}

/// Reject labels that asmi must never touch.
pub fn guard_protected(label: &str) -> Result<(), LaunchdError> {
    if PROTECTED_PREFIXES.iter().any(|p| label.starts_with(p)) {
        return Err(LaunchdError::Protected(label.to_string()));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    const LIST_FIXTURE: &str = include_str!("../tests/fixtures/launchctl_list.txt");
    const PRINT_ASMI_FIXTURE: &str = include_str!("../tests/fixtures/launchctl_print_asmi.txt");
    const PRINT_DISABLED_FIXTURE: &str =
        include_str!("../tests/fixtures/launchctl_print_disabled.txt");

    #[test]
    fn parses_launchctl_list_into_pid_to_label_map() {
        let map = parse_launchctl_list(LIST_FIXTURE);
        // Real fixture captured on hub: com.asmi.daemon has PID 53232
        assert_eq!(
            map.get(&53232).map(|s| s.as_str()),
            Some("com.asmi.daemon"),
            "asmi daemon should map to PID 53232"
        );
        // Several well-known agents should be present with numeric PIDs
        assert!(
            map.values().any(|v| v == "com.apple.Finder"),
            "Finder should appear with a numeric PID"
        );
    }

    #[test]
    fn launchctl_list_skips_dash_pids() {
        let map = parse_launchctl_list(LIST_FIXTURE);
        // None of the values should correspond to a row whose raw PID was `-`.
        // Quick spot-check: `com.ma.graph-indexer` is `-` in the fixture, so its
        // label must not appear in the map at all.
        assert!(
            !map.values().any(|v| v == "com.ma.graph-indexer"),
            "rows with `-` PID must be filtered out"
        );
    }

    #[test]
    fn parses_keep_alive_from_properties_line() {
        let info = parse_launchctl_print("com.asmi.daemon", PRINT_ASMI_FIXTURE);
        assert_eq!(info.label, "com.asmi.daemon");
        assert_eq!(info.keep_alive, Some(true), "properties contains `keepalive`");
        assert_eq!(info.run_at_load, Some(true), "properties contains `runatload`");
        assert_eq!(info.state, LaunchdState::Running);
        assert_eq!(info.program.as_deref(), Some("/usr/local/bin/asmi"));
    }

    #[test]
    fn parses_disabled_state_from_disabled_on_keys() {
        let info = parse_launchctl_print("com.test.fixture1", PRINT_DISABLED_FIXTURE);
        assert_eq!(info.state, LaunchdState::Disabled);
        // Properties line still present -> keep_alive populated
        assert_eq!(info.keep_alive, Some(true));
    }

    #[test]
    fn parses_print_disabled_output() {
        let sample = r#"	disabled services = {
		"com.apple.SpeechRecognitionCore.speechrecognitiond" => disabled
		"com.microsoft.teams2.agent" => enabled
		"com.test.fixture1" => disabled
	}
"#;
        let map = parse_print_disabled(sample);
        assert_eq!(map.get("com.test.fixture1"), Some(&true));
        assert_eq!(map.get("com.microsoft.teams2.agent"), Some(&false));
        assert_eq!(map.len(), 3);
    }

    #[test]
    fn guard_protected_rejects_asmi_prefix() {
        assert!(matches!(
            guard_protected("com.asmi.daemon"),
            Err(LaunchdError::Protected(_))
        ));
        assert!(matches!(
            guard_protected("com.asmi.helper"),
            Err(LaunchdError::Protected(_))
        ));
    }

    #[test]
    fn guard_protected_rejects_r1o_watchdog() {
        assert!(matches!(
            guard_protected("com.r1o.watchdog"),
            Err(LaunchdError::Protected(_))
        ));
    }

    #[test]
    fn guard_protected_allows_user_labels() {
        assert!(guard_protected("com.test.fixture1").is_ok());
        assert!(guard_protected("com.example.mlx-server").is_ok());
    }
}
