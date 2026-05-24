//! Integration test for ANE compute endpoints.
//!
//! Run explicitly: cargo test --test ane_endpoints -- --nocapture

use std::time::Duration;

/// When --experimental-ane is NOT passed, /ane/compute should respond with enabled: false.
#[tokio::test]
async fn test_ane_status_without_flag() {
    let port = 19290 + (std::process::id() % 500) as u16;
    let mut child = match tokio::process::Command::new(env!("CARGO_BIN_EXE_asmi"))
        .args(["--serve", "--port", &port.to_string()])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not spawn asmi: {e}, skipping");
            return;
        }
    };

    tokio::time::sleep(Duration::from_secs(3)).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://localhost:{port}/ane/compute"))
        .timeout(Duration::from_secs(5))
        .send()
        .await;

    match resp {
        Ok(r) => {
            assert!(r.status().is_success(), "/ane/compute returned {}", r.status());
            let body: serde_json::Value = r.json().await.unwrap();
            assert_eq!(body["enabled"], false);
            assert_eq!(body["experimental"], true);
            eprintln!("/ane/compute (no flag) -> OK");
        }
        Err(e) => panic!("/ane/compute failed: {e}"),
    }

    // /ane/eval should return 503 when disabled
    let resp = client
        .post(format!("http://localhost:{port}/ane/eval"))
        .timeout(Duration::from_secs(5))
        .send()
        .await;

    match resp {
        Ok(r) => {
            assert_eq!(r.status().as_u16(), 503);
            eprintln!("/ane/eval (disabled) -> 503 OK");
        }
        Err(e) => panic!("/ane/eval failed: {e}"),
    }

    child.kill().await.ok();
}

/// When --experimental-ane IS passed, /ane/compute should show enabled: true.
#[tokio::test]
async fn test_ane_status_with_flag() {
    let port = 19790 + (std::process::id() % 500) as u16;
    let mut child = match tokio::process::Command::new(env!("CARGO_BIN_EXE_asmi"))
        .args(["--serve", "--port", &port.to_string(), "--experimental-ane"])
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("could not spawn asmi: {e}, skipping");
            return;
        }
    };

    tokio::time::sleep(Duration::from_secs(3)).await;

    let client = reqwest::Client::new();
    let resp = client
        .get(format!("http://localhost:{port}/ane/compute"))
        .timeout(Duration::from_secs(5))
        .send()
        .await;

    match resp {
        Ok(r) => {
            assert!(r.status().is_success(), "/ane/compute returned {}", r.status());
            let body: serde_json::Value = r.json().await.unwrap();
            assert_eq!(body["enabled"], true);
            assert_eq!(body["compile_limit"], 119);
            eprintln!("/ane/compute (with flag) -> OK");
        }
        Err(e) => panic!("/ane/compute failed: {e}"),
    }

    child.kill().await.ok();
}
