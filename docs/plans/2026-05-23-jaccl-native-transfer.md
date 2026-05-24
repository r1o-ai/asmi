# Implementation plan — JACCL native RDMA transfer in asmi (v2 — post-critic)

**Date:** 2026-05-23
**Revision:** v3 — corrects critic's false findings re standalone lib path + MeshGroup void* API
**Companion research:** `~/.claude/projects/-Users-ma-Projects-r1o/memory/research_jaccl_rust_ffi_2026_05_23.md`
**Companion RE:** `~/.claude/projects/-Users-ma-Projects-r1o/memory/rdma_re_pd_lifecycle_2026_05_23.md`
**Working branch:** `feat/jaccl-native-transfer`
**Repo:** `/Users/ma/Projects/Personal/apple-smi/`
**Estimated time:** 4–6 hours

---

## Working protocol

Apply dependency-scanner framework before every multi-file edit. See iron-laws for non-negotiables.

## Architecture

```
┌──────────────────────────────────────────────────┐
│ asmi daemon (Rust, axum)                         │
│                                                  │
│  POST /transfer                                  │
│    ├─ preflight: PD budget probe + QP liveness   │
│    ├─ coordinate: HTTP handshake with peer asmi   │
│    │   (assigns rank, exchanges coordinator port) │
│    ├─ transfer: MeshGroup send/recv via FFI        │
│    │   64MB chunks, wallclock timeout per chunk   │
│    ├─ verify: SHA-256 manifest check              │
│    └─ result: file count + total bytes + hashes   │
│                                                  │
│  PeerGroupCache (process-lifetime)               │
│    ├─ HashMap<peer_hostname, JacclGroup>          │
│    ├─ init on first transfer to each peer         │
│    ├─ holds PD for process lifetime (NEVER drop)  │
│    ├─ QP liveness probe before reuse              │
│    └─ re-init ONLY on cable reseat (stale QP)     │
└──────────┬───────────────────────────────────────┘
           │ FFI (extern "C")
┌──────────▼───────────────────────────────────────┐
│ libjaccl.a (static lib, compiled by cc)           │
│                                                  │
│  jaccl_shim.cpp (~50 LOC)                        │
│    ├─ extern "C" wrappers for MeshGroup           │
│    │   (void* API: send, recv, all_sum, barrier)  │
│    ├─ jaccl_init delegates to jaccl::init(Config) │
│    │   which constructs MeshGroup internally      │
│    ├─ send/recv forward through Group vtable      │
│    └─ timeout wrapper: wallclock around send/recv │
│                                                  │
│  Compiled sources (ALL standalone, zero MLX deps):│
│    ├─ jaccl.cpp (init + Config builder)           │
│    ├─ mesh.cpp  (MeshGroup constructor + init)    │
│    ├─ ring.cpp  (RingGroup constructor + init)    │
│    ├─ rdma.cpp  (Connection, SharedBuffer, ibv)   │
│    ├─ tcp.cpp   (SideChannel for QP handshake)    │
│    └─ jaccl_shim.cpp (the C shim)                │
│                                                  │
│  Headers (all in vendor/jaccl/):                  │
│    ├─ group.h (abstract Group — void* API)        │
│    ├─ jaccl.h (init, Config, is_available)        │
│    ├─ mesh.h / mesh_impl.h (MeshGroup + template) │
│    ├─ ring.h / ring_impl.h (RingGroup + template) │
│    ├─ rdma.h (Connection, SharedBuffer, ibv types)│
│    ├─ tcp.h (SideChannel)                         │
│    ├─ types.h (Dtype enum)                        │
│    └─ reduction_ops.h (typed reduction dispatch)  │
│                                                  │
│  Vendored deps:                                   │
│    └─ nlohmann/json.hpp (single header)           │
└──────────────────────────────────────────────────┘
           │ dlopen at runtime (by utils.cpp)
┌──────────▼───────────────────────────────────────┐
│ librdma.dylib (Apple, macOS 26.2+)               │
│  re-exports: libibverbs, libmlx5,                │
│              libthunderboltrdma                   │
└──────────────────────────────────────────────────┘
```

**Key design decisions (all evidence-cited):**

| Decision | Rationale | Source |
|----------|-----------|-------|
| Wrap MeshGroup directly (void* API) | Standalone lib's Group::send takes `const void*`, not `mlx::array` — critic checked stale local checkout, not upstream `lib/jaccl/` | Verified 2026-05-23 on upstream main |
| Compile ALL standalone .cpp files | `lib/jaccl/*.cpp` have ZERO `mlx/` includes — critic was wrong | Verified: `grep -n "mlx/" lib/jaccl/*.cpp` returns empty |
| One Group per peer, NEVER deallocate | max_pd=11/device; kext leaks PDs on dealloc (RE confirmed) | RE session, `rdma_re_pd_lifecycle` |
| Source = rank 0, dynamic coordinator port | Avoids EADDRINUSE for concurrent transfers | Critic attack #4 |
| SHA-256 manifest verification | RDMA can silently corrupt on MR misalignment | Critic attack #7 |
| Wallclock timeout on send/recv | ibv_poll_cq loops forever if peer dies | Critic attack #5 |
| QP liveness probe before reuse | Cable reseat invalidates QP handles | Critic attack #6 |

## Tech stack

- Rust 1.85+ (workspace edition 2024)
- `cc` crate 1.2 (already in workspace, gated behind `ane` feature — adding `jaccl` gate)
- JACCL standalone lib from `mlx/distributed/jaccl/lib/jaccl/` (pinned to specific MLX commit)
- ALL `.cpp` files compile standalone (zero `mlx/` includes verified on upstream main)
- C++20, Apple Clang (cc crate supports C++20 — confirmed cc-1.2.55)
- `nlohmann/json` 3.x (header-only, vendored)
- macOS 26.2+ SDK (`<infiniband/verbs.h>` confirmed present on hub)
- `sha2` crate for Rust-side SHA-256 verification

## Tasks

### Phase A — Vendor JACCL standalone lib (20 min)

#### Task A1 — Sparse-checkout JACCL standalone lib from MLX — 10 min

**Pre-flight:** The standalone lib is at `mlx/distributed/jaccl/lib/jaccl/` (confirmed on upstream main 2026-05-23). ALL `.cpp` files have zero `mlx/` includes — they compile standalone. The `lib/` directory also has `CMakeLists.txt`, `README.md`, and `examples/`.

**Action:**
```bash
cd /Users/ma/Projects/Personal/apple-smi
mkdir -p vendor/jaccl

git clone --depth 1 --filter=blob:none --sparse \
  https://github.com/ml-explore/mlx.git /tmp/mlx-jaccl-vendor
cd /tmp/mlx-jaccl-vendor
git sparse-checkout set mlx/distributed/jaccl/lib

# Copy the entire standalone lib directory
cp -r mlx/distributed/jaccl/lib/jaccl/* \
   /Users/ma/Projects/Personal/apple-smi/vendor/jaccl/

# Record the pinned commit
echo "Vendored from ml-explore/mlx@$(git rev-parse HEAD)" \
  > /Users/ma/Projects/Personal/apple-smi/vendor/jaccl/VENDORED_FROM.txt

rm -rf /tmp/mlx-jaccl-vendor
```

Also vendor `nlohmann/json.hpp`:
```bash
curl -sL https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp \
  -o vendor/jaccl/nlohmann_json.hpp
```

**Verify:**
```bash
ls vendor/jaccl/{group.h,jaccl.h,jaccl.cpp,mesh.h,mesh.cpp,mesh_impl.h,ring.h,ring.cpp,ring_impl.h,rdma.h,rdma.cpp,tcp.h,tcp.cpp,types.h,reduction_ops.h}
# ALL present — headers AND implementations.
grep -c "mlx/" vendor/jaccl/*.cpp  # must return 0 for every file
```

**Commit:** `[jaccl-ffi] vendor JACCL standalone lib from ml-explore/mlx@<sha>`

#### Task A2 — Cherry-pick GID fix from PR #3468 — 10 min

**Pre-flight:** Read the GID regression (errno 22 on Apple TB — only link-local IPv6 GIDs exposed, PR #3468 still open as of 2026-05-13)

**Edit:** Apply the fix in `vendor/jaccl/rdma.cpp` (GID selection in Connection setup)

**Verify:** Syntax check — `xcrun c++ -std=c++20 -fsyntax-only -I vendor/jaccl vendor/jaccl/rdma.cpp`

**Commit:** `[jaccl-ffi] cherry-pick GID fix from ml-explore/mlx#3468`

### Phase B — C shim layer (45 min)

The shim is thin — MeshGroup's constructor and `jaccl::init()` handle all RDMA setup internally. The shim just wraps the `void*` Group API with `extern "C"` forwarding + timeout + probes.

#### Task B1 — Write jaccl_shim.h — 15 min

**Edit:** `vendor/jaccl/jaccl_shim.h`

```c
#ifndef JACCL_SHIM_H
#define JACCL_SHIM_H

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void* jaccl_group_t;

/* ── Availability + PD health ── */
bool jaccl_is_available(void);
int  jaccl_pd_budget_probe(const char* device_name);
     /* Returns remaining PD slots, or -1 on error.
        Allocates+immediately-deallocates one PD to test. */

/* ── Group lifecycle ── */
jaccl_group_t jaccl_init_mesh(
    int rank,                  /* 0 = coordinator/source, 1 = target */
    int world_size,            /* always 2 for point-to-point */
    const char* coordinator_ip,
    int coordinator_port,
    const char* devices_json_path,
    int timeout_ms             /* wallclock timeout for QP handshake */
);
/* Returns NULL on failure. Caller must check. */

int  jaccl_group_rank(jaccl_group_t g);
int  jaccl_group_size(jaccl_group_t g);

/* ── QP liveness probe ── */
int  jaccl_group_probe(jaccl_group_t g);
     /* Sends+receives 1 byte to/from peer. Returns 0 if alive,
        -1 if QP is stale (cable reseated). Caller should re-init. */

/* ── Point-to-point transfer ── */
int  jaccl_group_send(jaccl_group_t g, const void* buf, size_t len,
                      int dst, int timeout_ms);
int  jaccl_group_recv(jaccl_group_t g, void* buf, size_t len,
                      int src, int timeout_ms);
     /* Returns 0 on success, -1 on timeout, -2 on RDMA error. */

/* ── Teardown (call ONLY at process exit or confirmed cable reseat) ── */
void jaccl_group_free(jaccl_group_t g);

#ifdef __cplusplus
}
#endif

#endif /* JACCL_SHIM_H */
```

**Verify:** `xcrun cc -fsyntax-only vendor/jaccl/jaccl_shim.h` — clean C header

**Commit:** `[jaccl-ffi] add C shim header with PD probe + timeout + liveness`

#### Task B2 — Write jaccl_shim.cpp — 30 min

**Edit:** `vendor/jaccl/jaccl_shim.cpp`

The shim is ~50 LOC of `extern "C"` forwarding. All RDMA setup (dlopen, PD alloc, QP state machine, SideChannel handshake, MR registration) is handled internally by `jaccl::init()` → `MeshGroup` constructor.

The shim must:

1. **Wrap `jaccl::init(Config)`** — builds Config from the C arguments, calls init(), stores the returned `shared_ptr<Group>` as an opaque handle

2. **Forward send/recv/barrier** — cast handle back to `Group*`, call the `void*` methods directly

3. **Add wallclock timeout wrapper** — wraps each send/recv in a `std::async` + `future::wait_for` with configurable deadline

4. **Add QP liveness probe** — `jaccl_group_probe()` sends+receives 1 byte between ranks

5. **Add PD budget probe** — `jaccl_pd_budget_probe()` opens device, tries `ibv_alloc_pd`, immediately deallocs, closes device

6. **Exception safety** — all entry points wrapped in `try/catch(...)` returning error codes

**Verify:** 
```bash
xcrun c++ -std=c++20 -c -I vendor/jaccl \
  -I $(xcrun --show-sdk-path)/usr/include \
  vendor/jaccl/jaccl_shim.cpp -o /dev/null
```

**Commit:** `[jaccl-ffi] implement Group-wrapping C shim (~50 LOC)`

### Phase C — Rust FFI bindings (45 min)

#### Task C1 — Add jaccl feature to Cargo.toml — 5 min

**Edit:** `crates/cluster-monitor/Cargo.toml`

```toml
[features]
default = []
ane = ["dep:ane-runtime", "dep:objc2-io-surface", "dep:cc"]
jaccl = ["dep:cc"]

[build-dependencies]
cc = { version = "1.2", optional = true }
```

**Edit:** Root `Cargo.toml` — add `sha2 = "0.10"` to workspace dependencies for verification.

**Verify:** `cargo check --features jaccl`

**Commit:** `[jaccl-ffi] add jaccl feature gate`

#### Task C2 — Write build.rs — 20 min

**Edit:** `crates/cluster-monitor/build.rs`

```rust
#[cfg(feature = "jaccl")]
fn build_jaccl() {
    let vendor = std::path::Path::new("../../vendor/jaccl");
    cc::Build::new()
        .cpp(true)
        .std("c++20")
        .file(vendor.join("jaccl.cpp"))
        .file(vendor.join("mesh.cpp"))
        .file(vendor.join("ring.cpp"))
        .file(vendor.join("rdma.cpp"))
        .file(vendor.join("tcp.cpp"))
        .file(vendor.join("jaccl_shim.cpp"))
        .include(vendor)
        .flag("-Wno-deprecated-declarations")
        .compile("jaccl");
    // librdma.dylib is dlopen'd at runtime by rdma.cpp — no link flag needed
}
```

**Key:** ALL standalone `.cpp` files are compiled — they have zero `mlx/` dependencies (verified on upstream main). The `cc` crate produces `libjaccl.a`.

**Verify:** `cargo build --features jaccl` — compiles C++ and links

**Commit:** `[jaccl-ffi] build.rs compiles utils.cpp + shim via cc crate`

#### Task C3 — Write Rust FFI + safe wrapper — 20 min

**Edit:** `crates/cluster-monitor/src/jaccl_ffi.rs`

```rust
#[cfg(feature = "jaccl")]
pub mod jaccl {
    use std::ffi::{c_char, c_int, c_void, CString};
    use std::panic::catch_unwind;

    extern "C" {
        fn jaccl_is_available() -> bool;
        fn jaccl_pd_budget_probe(device: *const c_char) -> c_int;
        fn jaccl_init_mesh(
            rank: c_int, world_size: c_int,
            coordinator_ip: *const c_char, coordinator_port: c_int,
            devices_json: *const c_char, timeout_ms: c_int,
        ) -> *mut c_void;
        fn jaccl_group_rank(g: *mut c_void) -> c_int;
        fn jaccl_group_size(g: *mut c_void) -> c_int;
        fn jaccl_group_probe(g: *mut c_void) -> c_int;
        fn jaccl_group_send(
            g: *mut c_void, buf: *const c_void, len: usize,
            dst: c_int, timeout_ms: c_int,
        ) -> c_int;
        fn jaccl_group_recv(
            g: *mut c_void, buf: *mut c_void, len: usize,
            src: c_int, timeout_ms: c_int,
        ) -> c_int;
        fn jaccl_group_free(g: *mut c_void);
    }

    pub struct JacclGroup {
        handle: *mut c_void,
        rank: i32,
        peer_count: i32,
    }

    // SAFETY: JacclGroup is Send — the underlying MeshImpl is thread-safe
    // for sequential access (we hold a Mutex in the daemon).
    unsafe impl Send for JacclGroup {}

    impl JacclGroup {
        pub fn init(
            rank: i32, world_size: i32,
            coordinator_ip: &str, coordinator_port: i32,
            devices_json: &str, timeout_ms: i32,
        ) -> Result<Self, String> {
            let ip = CString::new(coordinator_ip).map_err(|e| e.to_string())?;
            let dev = CString::new(devices_json).map_err(|e| e.to_string())?;
            let handle = catch_unwind(|| unsafe {
                jaccl_init_mesh(
                    rank, world_size,
                    ip.as_ptr(), coordinator_port,
                    dev.as_ptr(), timeout_ms,
                )
            }).map_err(|_| "JACCL init panicked".to_string())?;
            if handle.is_null() {
                return Err("JACCL init returned null — PD exhausted or peer unreachable".into());
            }
            Ok(Self {
                handle,
                rank: unsafe { jaccl_group_rank(handle) },
                peer_count: unsafe { jaccl_group_size(handle) },
            })
        }

        pub fn rank(&self) -> i32 { self.rank }
        pub fn size(&self) -> i32 { self.peer_count }

        pub fn probe(&self) -> bool {
            catch_unwind(|| unsafe { jaccl_group_probe(self.handle) })
                .unwrap_or(-1) == 0
        }

        pub fn send(&self, buf: &[u8], dst: i32, timeout_ms: i32) -> Result<(), String> {
            let rc = catch_unwind(|| unsafe {
                jaccl_group_send(
                    self.handle, buf.as_ptr() as *const c_void,
                    buf.len(), dst, timeout_ms,
                )
            }).map_err(|_| "send panicked")?;
            match rc {
                0 => Ok(()),
                -1 => Err("send timed out".into()),
                _ => Err(format!("send failed: rc={rc}")),
            }
        }

        pub fn recv(&self, buf: &mut [u8], src: i32, timeout_ms: i32) -> Result<(), String> {
            let rc = catch_unwind(|| unsafe {
                jaccl_group_recv(
                    self.handle, buf.as_mut_ptr() as *mut c_void,
                    buf.len(), src, timeout_ms,
                )
            }).map_err(|_| "recv panicked")?;
            match rc {
                0 => Ok(()),
                -1 => Err("recv timed out".into()),
                _ => Err(format!("recv failed: rc={rc}")),
            }
        }

        pub fn is_available() -> bool {
            catch_unwind(|| unsafe { jaccl_is_available() }).unwrap_or(false)
        }

        pub fn pd_budget(device: &str) -> i32 {
            let dev = CString::new(device).unwrap_or_default();
            catch_unwind(|| unsafe { jaccl_pd_budget_probe(dev.as_ptr()) })
                .unwrap_or(-1)
        }
    }

    impl Drop for JacclGroup {
        fn drop(&mut self) {
            // WARNING: dropping a JacclGroup leaks a PD in the kernel.
            // Only drop on process exit or confirmed cable reseat.
            unsafe { jaccl_group_free(self.handle); }
        }
    }
}
```

**Verify:** `cargo build --features jaccl` — links, no undefined symbols

**Commit:** `[jaccl-ffi] Rust FFI bindings with catch_unwind + timeout + PD probe`

### Phase D — Transfer endpoint + peer protocol (2 hours)

#### Task D1 — Peer coordination protocol — 30 min

**Edit:** `src/daemon.rs` — add `POST /transfer/coordinate` endpoint

**Protocol (explicit coordinator assignment):**

1. Web app calls `POST /transfer` on source asmi with `{ model, target_hostname }`
2. Source asmi:
   - Picks a **random ephemeral port** (49152–65535) for the SideChannel coordinator
   - Tests port with `bind()` — retry if EADDRINUSE
   - Calls `POST http://<target>:9090/transfer/coordinate` with:
     ```json
     {
       "model": "DeepSeek-V4-Flash-4bit",
       "coordinator_ip": "10.1.10.70",  // LAN IP, never TB5 /30
       "coordinator_port": 52847,        // the random port
       "source_rank": 0,
       "target_rank": 1,
       "transfer_id": "uuid"
     }
     ```
3. Target asmi:
   - Validates model path doesn't already exist (or clears it)
   - Inits JACCL as rank 1, connecting to coordinator at source's IP:port
   - Responds `{ "accepted": true }` once JACCL handshake completes
4. Source asmi:
   - Inits JACCL as rank 0, listening on the coordinator port
   - Both sides are now connected

**Deadlock prevention:** Source starts SideChannel listener BEFORE calling target's coordinate endpoint. Target connects as rank 1. The TCP handshake is the synchronization point — no race.

**Concurrent transfers:** Each transfer gets its own random port. No shared state between transfers to different peers.

**Verify:** `cargo build` — endpoint compiles

**Commit:** `[jaccl-ffi] peer coordination with dynamic port + rank assignment`

#### Task D2 — File transfer loop — 30 min

**Edit:** `src/daemon.rs` — `serve_transfer_handler`

Transfer protocol over the JACCL channel:
1. Source sends header: `{ file_count, total_bytes, files: [{path, size, sha256}] }` as JSON (max 1MB)
2. For each file:
   - Source reads 64MB chunks, `send()` with 5-minute timeout per chunk
   - Target `recv()` and writes to `~/Models/.tmp.<model>.<transfer_id>/`
3. Source sends sentinel `{done: true}`
4. Target verifies SHA-256 of each received file against the manifest
5. If all match: atomic `mv .tmp → final`, respond success
6. If any mismatch: delete `.tmp`, respond with which files failed

**Verify:** `cargo build`

**Commit:** `[jaccl-ffi] file transfer loop with 64MB chunks + SHA-256 manifest`

#### Task D3 — PeerGroupCache (long-lived Groups) — 30 min

**Edit:** `src/daemon.rs`

```rust
struct PeerGroupCache {
    groups: Mutex<HashMap<String, JacclGroup>>,
}

impl PeerGroupCache {
    fn get_or_init(&self, peer: &str, ...) -> Result<&JacclGroup, String> {
        let mut map = self.groups.lock().unwrap();
        if let Some(g) = map.get(peer) {
            // QP liveness probe — if cable was reseated, QP is stale
            if g.probe() {
                return Ok(g);
            }
            // Stale — drop and re-init (costs 1 PD, but cable reseat is rare)
            tracing::warn!("QP stale for {peer}, re-initializing");
            map.remove(peer); // Drop triggers jaccl_group_free
        }
        // First transfer to this peer — init new Group
        let g = JacclGroup::init(...)?;
        map.insert(peer.to_string(), g);
        Ok(map.get(peer).unwrap())
    }
}
```

**PD budget monitoring:**
- Before init, call `JacclGroup::pd_budget("rdma_enN")` for the relevant device
- If budget == 0: return error "PD exhausted on rdma_enN — cold reboot required"
- Log remaining PD count after each init

**Verify:** `cargo build` — single-init + liveness pattern compiles

**Commit:** `[jaccl-ffi] PeerGroupCache with QP liveness + PD budget monitoring`

#### Task D4 — SSE progress streaming — 15 min

**Edit:** `src/daemon.rs`

`POST /transfer` returns `text/event-stream` with events:
```
data: {"type":"stage","stage":"preflight","detail":"PD budget: 8/11 on rdma_en5"}
data: {"type":"stage","stage":"coordinating","detail":"rank 0, port 52847"}
data: {"type":"progress","percent":12,"bytesPerSec":3800000000}
data: {"type":"stage","stage":"verifying","detail":"SHA-256 manifest check"}
data: {"type":"done","durationMs":42000}
```

**Verify:** `cargo build`

**Commit:** `[jaccl-ffi] SSE progress streaming for /transfer`

### Phase E — Web app integration (30 min)

#### Task E1 — Simplify tryJacclRdma transport — 30 min

**Edit:** `~/Projects/r1o/web/src/lib/model-transfer.ts`

Replace the current 100+ line `tryJacclRdma` (which spawns `mlx.launch`) with a single `fetch()`:

```typescript
async function* tryJacclRdma(req: TransferRequest) {
  const transport: TransferTransport = 'jaccl-rdma';
  const started = Date.now();

  yield { type: 'stage', stage: 'preflight', transport };

  // Single HTTP call to source asmi — it handles everything
  const res = await fetch(`http://${asmiHost(req.sourceHostname)}:9090/transfer`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      model: req.modelDir,
      target: req.targetHostname,
    }),
    signal: req.signal,
  });

  // Parse SSE stream — same pattern as existing spawnStreaming
  // but from HTTP instead of child_process
  ...
}
```

**Verify:** `npx tsc --noEmit` — clean

**Commit:** `[jaccl-ffi] web transport calls asmi /transfer (single HTTP call)`

## File touch matrix

| File | Lines | Notes |
|---|---|---|
| `vendor/jaccl/*.{h,cpp}` | ~2500 (vendored) | Full JACCL standalone lib, read-only except GID fix |
| `vendor/jaccl/jaccl_shim.h` | ~50 | C FFI header |
| `vendor/jaccl/jaccl_shim.cpp` | ~50 | Group-wrapping shim with timeout + probes |
| `vendor/jaccl/nlohmann_json.hpp` | vendored | Single-header JSON |
| `crates/cluster-monitor/Cargo.toml` | ~5 | jaccl feature |
| `crates/cluster-monitor/build.rs` | ~20 | cc::Build for all JACCL .cpp + shim |
| `crates/cluster-monitor/src/jaccl_ffi.rs` | ~120 | Rust FFI + safe wrapper |
| `src/daemon.rs` | ~300 | /transfer, /transfer/coordinate, PeerGroupCache, SSE |
| `Cargo.toml` | ~2 | sha2 dep |
| `web/src/lib/model-transfer.ts` | ~40 (replace ~120) | Simplified JACCL transport |

**Total:** ~600 LOC new (excluding vendored), ~120 LOC removed.

## PD budget analysis

| Scenario | PDs consumed per device | Sustainable? |
|---|---|---|
| 1 peer (point-to-point) | 1 PD + 1 QP + 1 CQ | Yes (10 remaining) |
| 3 peers (4-node mesh) | 3 PD + 3 QP + 3 CQ | Yes (8 remaining) |
| 5 peers (6-node mesh) | 5 PD + 5 QP + 5 CQ | Yes (6 remaining) |
| 10 peers (theoretical) | 10 PD + 10 QP + 10 CQ | Marginal (1 remaining) |
| Cable reseat (re-init) | +1 PD leaked per reseat | Track — warn after 5 reseats |

**Critical constraint:** max_pd=11, max_qp=11, max_cq=11 per device, max_mr=100 per device. Leaked PDs cannot be recovered (IORDMAFamily.kext bug, confirmed by RE disassembly of `tbt_dealloc_pd` → `ibv_cmd_dealloc_pd` → `execute_ioctl` — ioctl returns success but kernel retains the slot).

## Risk register

| Risk | Mitigation |
|---|---|
| PD exhaustion on init failure | Never retry init. Surface "cold reboot required". Log PD budget. |
| PD leak on cable reseat | Track reseat count per device. Warn after 5 reseats. |
| Panic in Rust FFI leaks PD | `catch_unwind` on ALL FFI calls. |
| QP stale after cable reseat | Liveness probe before each transfer. Re-init if probe fails. |
| Peer dies mid-transfer | 5-min wallclock timeout on each send/recv chunk. |
| Coordinator port conflict | Random ephemeral port + bind() test. Retry up to 5 ports. |
| Data corruption over RDMA | SHA-256 manifest verification on target. |
| GID regression (errno 22) | Cherry-pick #3468 into vendored utils.cpp. |
| Upstream API change | Pin to specific MLX commit. Re-vendor manually. |
| macOS SDK < 26.2 | Feature-gate: `jaccl` feature only compiles on macOS 26.2+. |

## Rollback strategy

| Failure point | Action |
|---|---|
| Shim won't compile | Disable `jaccl` feature flag, fallback transports still work |
| RDMA devices exhausted | Cold reboot cluster (60s power-off), asmi restarts with fresh PD budget |
| FFI segfault | `catch_unwind` catches, logs stack, falls through to tar\|nc transport |
| QP stale on all devices | PeerGroupCache re-inits; if all re-inits fail → fallback transport |
| SHA-256 mismatch | Delete partial transfer, retry once, then fallback |

## Acceptance criteria

1. `cargo build --features jaccl` produces working asmi binary
2. `POST /transfer {"model":"test-small","target":"m3u3"}` transfers a model via RDMA
3. Transfer speed > 1 GB/s (baseline: 11.7 GB/s on 100MB all_sum)
4. PD budget consumed: exactly 1 per peer, verified via `jaccl_pd_budget_probe`
5. SHA-256 manifest verified on target for every transfer
6. QP liveness probe detects cable reseat within 1 second
7. Web app drag-drop uses JACCL transport, shows "RDMA" badge
8. Fallback to tar|nc when JACCL unavailable (no RDMA, PD exhausted)
9. `POST /transfer` returns SSE stream with progress events

## Out of scope

- Multi-node broadcast (3+ nodes) — point-to-point only
- Porting JACCL to pure Rust (we wrap C++ via FFI)
- ANE integration
- Distributed inference (stays in Python/mlx.launch)
- Kext patching for PD leak fix (Apple kernel bug — filed separately)
- Cluster-wide PD monitoring dashboard (future — asmi already surfaces budget per transfer)
