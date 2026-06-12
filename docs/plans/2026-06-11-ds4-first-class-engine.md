# ds4 First-Class Engine + iOS Restart Button

**Date:** 2026-06-11
**ARDD Phase:** Plan (evidence-cited)
**Scope:** apple-smi (asmi) + r1o iOS app

## Problem

ds4-server runs as a launchd sidecar that asmi can only observe, not manage.
When ds4 jams (stuck prefill loop, hung connections), the only fix is SSH into
the node and manually `launchctl unload/load`. The iOS app has no way to
restart it.

Research found 10 specific gaps (see ARDD research artifact).

## Architecture Decision

**Native binary spawn path in do_serve_load_inner.**

Currently serve.rs:929-949 always runs `python3 -m <binary>`. ds4 is a compiled
C binary. The fix: check if the engine is native (Ds4, DFlash) and spawn the
binary directly instead of through python3.

This means asmi can fully manage ds4: spawn, kill, restart, swap models.
The launchd plist becomes optional (startup convenience), not the lifecycle owner.

## Tasks

### Tier 1: asmi backend (apple-smi repo)

#### T1: Fix ds4 port default
- **File:** `src/serve.rs:41-42`
- **Change:** `ServeEngine::Ds4 => ... .unwrap_or(8080)` (was 41000)
- **Evidence:** plist uses 8080, ds4-server default is 8000, 41000 matches nothing
- **Gate:** unit test asserting port_for_engine(Ds4) == 8080

#### T2: Add ds4 to managed_ports()
- **File:** `src/serve.rs:23-28`
- **Change:** Add `(Ds4, port_for_engine(Ds4))` to the returned vec
- **Evidence:** Only MlxLm + MlxVlm get managers at boot; ds4 only found via adopt_unmanaged
- **Gate:** daemon_startup creates a ServeManager for port 8080

#### T3: Native binary spawn path in do_serve_load_inner
- **File:** `src/serve.rs:929-949`
- **Change:** Before the `python3 -m` path, add a branch for native binaries.
  ONLY Ds4 is native — DFlash is `python3 -m dflash_mlx.serve` (Python package).
  ```rust
  let is_native = matches!(engine, ServeEngine::Ds4);
  if is_native {
      program = resolve_native_binary(cfg.binary, &engine)?;
      // Direct CLI flags — no python3 -m
      if let (Some(flag), Some(ref model_path)) = (cfg.model_flag, &req.model_path) {
          cmd_args.push(flag.into());
          cmd_args.push(model_path.clone());
      }
      // Port binding (critical: without --port, ds4 defaults to 8000, not 8080)
      cmd_args.extend(["--port".into(), port.to_string()]);
      cmd_args.extend(["--host".into(), "0.0.0.0".into()]);
      if let Some(ctx) = req.ctx_size {
          cmd_args.extend(["-c".into(), ctx.to_string()]);
      }
  } else {
      // existing python3 -m path (unchanged)
  }
  ```
- **Evidence:** `python3 -m ds4-server` does not exist; ds4-server is at ~/opensource/ds4/ds4-server
- **Critic fix (v2):** Removed DFlash from is_native (it's Python). Added --port. Fixed Vec<String> types. Fixed model_path Option unwrap.
- **Gate:** `POST /serve/load` with engine=ds4 spawns the correct binary on correct port

#### T4: resolve_native_binary helper
- **File:** `src/serve.rs` (new function)
- **Change:** Resolve binary path: check env `DS4_SERVER_PATH`, then `~/.r1o/bin/ds4-server`, then PATH lookup via `which`. Return Result with clear error if not found.
- **Gate:** unit test with env var override

#### T5: Add ctx_size to LoadRequest
- **File:** `crates/cluster-monitor/src/types.rs` (LoadRequest struct)
- **Change:** Add `pub ctx_size: Option<u64>` with serde default
- **Evidence:** ds4 needs `-c` for context window; mlx engines don't use this
- **Gate:** backward compatible (serde default, existing clients unaffected)

#### T6: DROPPED (per critic)
~~Update EngineConfig for Ds4~~ — existing `model_flag: Some("--model")` is already
correct (ds4 accepts both -m and --model). Default args should come from LoadRequest
params (T5 ctx_size) and the spawn code (T3), not be baked into EngineConfig.

#### T7: POST /serve/restart handler (fix + verify)
- **File:** `src/daemon.rs:1266+`
- **Change:** Fix `l.keep_alive` → `l.keep_alive.unwrap_or(false)` (Option<bool>).
  Verify handler handles:
  - launchd-managed (kill + wait for KeepAlive revive) -- works
  - asmi-managed native (stop + re-spawn with same args) -- needs T3 first
- **Critic fix (v2):** The restart handler's keep_alive type was Option<bool>, not bool. Fixed.
- **Gate:** `curl -X POST "http://localhost:9090/serve/restart?port=8080"` restarts ds4

### Tier 2: iOS app (r1o repo)

#### T8: Add ServeEngine.ds4 to iOS
- **File:** `ios/r1oPackage/Sources/r1oFeature/Services/AsmiServeClient.swift:78-106`
- **Change:** Add case `ds4 = "ds4"` with displayName "DS4", defaultPort 8080, icon "bolt"
- **Gate:** compiles, no enum exhaustiveness errors

#### T9: Add restartServer() to AsmiServeClient
- **File:** `ios/r1oPackage/Sources/r1oFeature/Services/AsmiServeClient.swift`
- **Change:** New method:
  ```swift
  public func restartServer(servePort: Int) async throws -> AsmiServeRestartResponse {
      let data = try await post("/serve/restart?port=\(servePort)", body: nil)
      return try JSONDecoder().decode(AsmiServeRestartResponse.self, from: data)
  }
  ```
  Plus response type with ok, port, state, message fields.
- **Gate:** compiles

#### T10: Add restart button to cluster server rows
- **File:** `ios/r1oPackage/Sources/r1oFeature/Panes/ServersPane.swift`
- **Change:** In the cluster server card/row, add a restart button (arrow.clockwise.circle)
  that calls `AsmiServeClient(host: node.hostname).restartServer(servePort: slot.port)`.
  Show confirmation alert before restart. Show spinner during restart. Show success/error toast.
- **Gate:** button visible, tappable, calls endpoint

## Dependency Order

T1 → T2 → T4 → T3 → T5 → T6 → T7 (asmi, sequential)
T8 → T9 → T10 (iOS, sequential but independent of asmi)

Both tiers can execute in parallel.

## Risk Assessment

- **Low risk:** T1, T2, T5, T8, T9 are additive changes
- **Medium risk:** T3 (native spawn path) — new code path, could break existing mlx flow if branching is wrong
- **Mitigation:** T3 only triggers for `is_native` engines; existing python path unchanged
- **Medium risk:** T7 restart handler — 30s poll timeout might not be enough for ds4 cold start on large GGUF
- **Mitigation:** Make timeout configurable via query param `?timeout=60`
