# Fix plan — JACCL transfer RDMA init on macOS

**Date:** 2026-05-24
**Repo:** `/Users/ma/Projects/Personal/apple-smi/`
**Estimated time:** 30 min
**Files:** 3 (transfer.rs, jaccl_shim.cpp, jaccl_ffi.rs)

---

## Root cause

`src/transfer.rs:288` calls `pd_probe("mlx5_0")` — a Linux device name. macOS RDMA devices are named `rdma_enX`. The probe returns -1 ("device not found"), which the transfer handler interprets as "RDMA not available."

The `/rdma/health` endpoint works because it iterates all devices from `ibv_get_device_list()` and probes each by its actual name.

## Tasks

### Task 1 — Add `jaccl_pd_probe_any_active()` to C shim — 10 min

**Edit:** `vendor/jaccl/jaccl_shim.cpp`

Add a new function that iterates all devices, finds the first with PORT_ACTIVE, and probes its PD budget. No hardcoded device name needed.

```c
int jaccl_pd_probe_any_active(void) {
    // iterate ibv_get_device_list
    // for each: open, query_port, if ACTIVE -> try alloc_pd
    // return 1 (ok), 0 (all exhausted), -1 (no devices or no libibverbs)
}
```

Also add to `vendor/jaccl/jaccl_shim.h`.

**Verify:** `xcrun c++ -std=c++20 -c vendor/jaccl/jaccl_shim.cpp -I vendor/jaccl -I vendor -o /dev/null`

**Commit:** `[jaccl-ffi] add jaccl_pd_probe_any_active — no hardcoded device names`

### Task 2 — Expose in Rust FFI — 5 min

**Edit:** `crates/cluster-monitor/src/jaccl_ffi.rs`

Add:
```rust
extern "C" { fn jaccl_pd_probe_any_active() -> c_int; }

pub fn pd_probe_any_active() -> i32 {
    unsafe { jaccl_pd_probe_any_active() }
}
```

**Verify:** `cargo build --features jaccl`

**Commit:** `[jaccl-ffi] expose pd_probe_any_active in Rust FFI`

### Task 3 — Fix transfer.rs to use the new probe — 5 min

**Edit:** `src/transfer.rs:288`

```rust
// Before:
let pd_ok = tokio::task::spawn_blocking(|| jaccl_ffi::pd_probe("mlx5_0"))

// After:
let pd_ok = tokio::task::spawn_blocking(|| jaccl_ffi::pd_probe_any_active())
```

Also review the rest of `transfer.rs` for any other hardcoded device names or Linux-specific assumptions.

**Verify:** `cargo build --features jaccl` + `curl -X POST localhost:9090/transfer -d '{"model_dir":"test","peer":"m3u3","direction":"send"}'` — should get past preflight

**Commit:** `[jaccl-ffi] fix transfer preflight: probe any active RDMA device, not hardcoded mlx5_0`

### Task 4 — Test end-to-end — 10 min

1. Restart asmi: `pkill -f "asmi.*serve"; sleep 2; nohup asmi --serve --bind 0.0.0.0 --cluster --port 9090 --models-dir ~/Models > /tmp/asmi.log 2>&1 &`
2. Test transfer preflight: `curl -X POST localhost:9090/transfer -d '{"model_dir":"Qwen3.6-35B-A3B-4bit","peer":"m3u3","direction":"send"}'`
3. Verify SSE stream gets past preflight into coordination stage
4. Test via web app: drag a small model in Transfer view

**Commit:** (no code change — verification only)

## Acceptance criteria

1. `/transfer` preflight passes on macOS (no "RDMA not available" false negative)
2. SSE stream shows `stage: preflight` → `stage: copying` (or meaningful error from coordination, not from device probe)
3. `cargo build` and `cargo build --features jaccl` both clean
