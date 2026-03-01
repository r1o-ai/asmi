//! Integration tests for ShareManager lifecycle.
//!
//! These tests verify the ShareManager state transitions and lifecycle methods.
//! ShareManager is in the asmi binary crate's serve.rs module and manages
//! a single mlx_lm.share distributed session with crash recovery.

use asmi_core::{ServeBackend, ServeState, ShareRequest, ShareStatus};
use std::time::Duration;

#[tokio::test]
async fn test_share_request_types_compile() {
    // Verify ShareRequest can be created with all fields.
    let req = ShareRequest {
        model_path: "/home/user/models/Llama-2-7b".to_string(),
        backend: "auto".to_string(),
        hostfile: None,
    };

    assert_eq!(req.model_path, "/home/user/models/Llama-2-7b");
    assert_eq!(req.backend, "auto");
    assert!(req.hostfile.is_none());
}

#[tokio::test]
async fn test_share_request_with_hostfile() {
    // Verify ShareRequest can include a hostfile path.
    let req = ShareRequest {
        model_path: "meta-llama/Llama-2-7b-hf".to_string(),
        backend: "jaccl".to_string(),
        hostfile: Some("/home/user/.r1o/hostfiles/cluster.json".to_string()),
    };

    assert_eq!(req.backend, "jaccl");
    assert_eq!(
        req.hostfile,
        Some("/home/user/.r1o/hostfiles/cluster.json".to_string())
    );
}

#[tokio::test]
async fn test_share_status_serialization() {
    // Verify ShareStatus can be serialized to/from JSON.
    let status = ShareStatus {
        state: ServeState::Idle,
        model: None,
        backend: ServeBackend::Single,
        pid: None,
        elapsed_ms: 0,
        error: None,
    };

    let json = serde_json::to_value(&status).expect("should serialize");
    assert_eq!(json["state"], "idle");
    assert!(json["model"].is_null());
    assert!(json["pid"].is_null());
    assert!(json["error"].is_null());
}

#[tokio::test]
async fn test_share_status_with_error() {
    // Verify ShareStatus correctly represents an error state.
    let error_msg = "Model file not found: /nonexistent/model".to_string();
    let status = ShareStatus {
        state: ServeState::Error,
        model: Some("/nonexistent/model".to_string()),
        backend: ServeBackend::Single,
        pid: None,
        elapsed_ms: 5000,
        error: Some(error_msg.clone()),
    };

    let json = serde_json::to_value(&status).expect("should serialize");
    assert_eq!(json["state"], "error");
    assert_eq!(json["model"], "/nonexistent/model");
    assert_eq!(json["elapsed_ms"], 5000);
    assert_eq!(json["error"], error_msg);
}

#[tokio::test]
async fn test_share_status_ready_with_pid() {
    // Verify ShareStatus correctly represents a ready state with PID.
    let status = ShareStatus {
        state: ServeState::Ready,
        model: Some("meta-llama/Llama-2-7b-hf".to_string()),
        backend: ServeBackend::Jaccl,
        pid: Some(12345),
        elapsed_ms: 45000,
        error: None,
    };

    let json = serde_json::to_value(&status).expect("should serialize");
    assert_eq!(json["state"], "ready");
    assert_eq!(json["model"], "meta-llama/Llama-2-7b-hf");
    assert_eq!(json["backend"], "jaccl");
    assert_eq!(json["pid"], 12345);
}

#[tokio::test]
async fn test_share_status_loading_state() {
    // Verify ShareStatus correctly represents a loading state in progress.
    let status = ShareStatus {
        state: ServeState::Loading,
        model: Some("meta-llama/Llama-2-13b-hf".to_string()),
        backend: ServeBackend::Single,
        pid: None,
        elapsed_ms: 12000,
        error: None,
    };

    assert_eq!(status.state, ServeState::Loading);
    assert!(status.pid.is_none()); // PID assigned only when ready
    assert_eq!(status.elapsed_ms, 12000);
}

#[tokio::test]
async fn test_serve_state_enum_variants() {
    // Verify all ServeState variants exist and can be serialized.
    let states = vec![
        ServeState::Idle,
        ServeState::Loading,
        ServeState::Ready,
        ServeState::Error,
        ServeState::Bare,
    ];

    for state in states {
        let json = serde_json::to_value(state).expect("should serialize");
        assert!(json.is_string());
    }
}

#[tokio::test]
async fn test_serve_backend_enum_variants() {
    // Verify ServeBackend variants serialize correctly.
    let single_json = serde_json::to_value(ServeBackend::Single).expect("should serialize");
    let jaccl_json = serde_json::to_value(ServeBackend::Jaccl).expect("should serialize");

    assert_eq!(single_json, "single");
    assert_eq!(jaccl_json, "jaccl");
}

#[tokio::test]
async fn test_share_request_json_round_trip() {
    // Verify ShareRequest can round-trip through JSON serialization.
    let original = ShareRequest {
        model_path: "/models/custom-model".to_string(),
        backend: "jaccl".to_string(),
        hostfile: Some("/etc/hostfile.json".to_string()),
    };

    let json_str = serde_json::to_string(&original).expect("should serialize");
    let deserialized: ShareRequest =
        serde_json::from_str(&json_str).expect("should deserialize");

    assert_eq!(deserialized.model_path, original.model_path);
    assert_eq!(deserialized.backend, original.backend);
    assert_eq!(deserialized.hostfile, original.hostfile);
}

#[tokio::test]
async fn test_share_status_json_round_trip() {
    // Verify ShareStatus can round-trip through JSON serialization.
    let original = ShareStatus {
        state: ServeState::Ready,
        model: Some("test-model".to_string()),
        backend: ServeBackend::Jaccl,
        pid: Some(54321),
        elapsed_ms: 30000,
        error: None,
    };

    let json_str = serde_json::to_string(&original).expect("should serialize");
    let deserialized: ShareStatus =
        serde_json::from_str(&json_str).expect("should deserialize");

    assert_eq!(deserialized.state, original.state);
    assert_eq!(deserialized.model, original.model);
    assert_eq!(deserialized.backend, original.backend);
    assert_eq!(deserialized.pid, original.pid);
    assert_eq!(deserialized.elapsed_ms, original.elapsed_ms);
    assert_eq!(deserialized.error, original.error);
}

#[tokio::test]
async fn test_share_request_default_backend() {
    // Verify ShareRequest backend field defaults to "auto" via serde.
    let json = r#"{"model_path": "/models/test"}"#;
    let req: ShareRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(req.backend, "auto");
}

#[tokio::test]
async fn test_share_request_explicit_backend() {
    // Verify ShareRequest accepts explicit backend values.
    let json = r#"{"model_path": "/models/test", "backend": "single"}"#;
    let req: ShareRequest = serde_json::from_str(json).expect("should deserialize");
    assert_eq!(req.backend, "single");
}

// ---------------------------------------------------------------------------
// Integration test: daemon endpoint (requires daemon running)
// ---------------------------------------------------------------------------

/// Test the /serve/share endpoints via HTTP.
/// This requires the asmi daemon to be running.
/// Run explicitly: cargo test --test serve_share -- --ignored --nocapture test_share_http
#[tokio::test]
#[ignore] // requires daemon to be running
async fn test_share_http_status_endpoint() {
    let daemon_base = "http://localhost:9090";
    let client = reqwest::Client::new();

    // Check if daemon is running
    let health = match client
        .get(&format!("{daemon_base}/health"))
        .timeout(Duration::from_secs(2))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => true,
        _ => false,
    };

    if !health {
        eprintln!("daemon not running, skipping HTTP test");
        return;
    }

    // Query share status endpoint
    let url = format!("{daemon_base}/serve/share/status");
    let resp = client
        .get(&url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .expect("share status endpoint should respond");

    assert!(resp.status().is_success(), "share status returned {}", resp.status());

    let body: ShareStatus = resp
        .json()
        .await
        .expect("response should be valid ShareStatus JSON");

    // Initially should be Idle (no share session running)
    assert_eq!(body.state, ServeState::Idle);
    assert!(body.pid.is_none());
    eprintln!("share status: {:?}", body);
}

/// Test POST /serve/share/start with non-existent model transitions to Error.
/// This requires the asmi daemon to be running.
/// Run explicitly: cargo test --test serve_share -- --ignored --nocapture test_share_http_nonexistent_model
#[tokio::test]
#[ignore] // requires daemon to be running
async fn test_share_http_nonexistent_model() {
    let daemon_base = "http://localhost:9090";
    let client = reqwest::Client::new();

    // Check if daemon is running
    let health = match client
        .get(&format!("{daemon_base}/health"))
        .timeout(Duration::from_secs(2))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => true,
        _ => false,
    };

    if !health {
        eprintln!("daemon not running, skipping HTTP test");
        return;
    }

    // POST to start share with non-existent model
    let req = ShareRequest {
        model_path: "/nonexistent/model/path".to_string(),
        backend: "auto".to_string(),
        hostfile: None,
    };

    let start_url = format!("{daemon_base}/serve/share/start");
    let start_resp = client
        .post(&start_url)
        .json(&req)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .expect("share start endpoint should respond");

    assert!(
        start_resp.status().is_success(),
        "share start returned {}",
        start_resp.status()
    );

    // Give background task time to fail
    tokio::time::sleep(Duration::from_secs(3)).await;

    // Poll status — should be Error state
    let status_url = format!("{daemon_base}/serve/share/status");
    let status_resp = client
        .get(&status_url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .expect("share status endpoint should respond");

    let status: ShareStatus = status_resp
        .json()
        .await
        .expect("response should be valid ShareStatus JSON");

    eprintln!("status after failed start: {:?}", status);

    // Model doesn't exist — should be Error
    assert_eq!(status.state, ServeState::Error);
    assert!(
        status.error.is_some(),
        "error state should include error message"
    );
    eprintln!("error message: {:?}", status.error);
}

/// Test POST /serve/share/stop returns to Idle.
/// This requires the asmi daemon to be running.
/// Run explicitly: cargo test --test serve_share -- --ignored --nocapture test_share_http_stop
#[tokio::test]
#[ignore] // requires daemon to be running
async fn test_share_http_stop() {
    let daemon_base = "http://localhost:9090";
    let client = reqwest::Client::new();

    // Check if daemon is running
    let health = match client
        .get(&format!("{daemon_base}/health"))
        .timeout(Duration::from_secs(2))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => true,
        _ => false,
    };

    if !health {
        eprintln!("daemon not running, skipping HTTP test");
        return;
    }

    // POST to stop share
    let stop_url = format!("{daemon_base}/serve/share/stop");
    let stop_resp = client
        .post(&stop_url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .expect("share stop endpoint should respond");

    assert!(
        stop_resp.status().is_success(),
        "share stop returned {}",
        stop_resp.status()
    );

    // Poll status — should be Idle
    let status_url = format!("{daemon_base}/serve/share/status");
    let status_resp = client
        .get(&status_url)
        .timeout(Duration::from_secs(5))
        .send()
        .await
        .expect("share status endpoint should respond");

    let status: ShareStatus = status_resp
        .json()
        .await
        .expect("response should be valid ShareStatus JSON");

    assert_eq!(status.state, ServeState::Idle);
    assert!(status.pid.is_none());
    assert!(status.error.is_none());
    eprintln!("status after stop: {:?}", status);
}
