# Fix plan — JACCL handshake timeout in transfer endpoint

**Date:** 2026-05-24
**Repo:** `/Users/ma/Projects/Personal/apple-smi/`
**Branch:** `feat/jaccl-native-transfer`
**Estimated time:** 1 hour

---

## Root cause

`jaccl_init_mesh_auto()` times out after 30s on both coordinator and peer. The TCP SideChannel can't complete the QP handshake. Both sides enter init, but the SideChannel constructor (which does TCP connect/listen + IB endpoint exchange) fails to connect.

**Proven so far:**
- PD probe passes (active devices with budget)
- Peer coordination works (HTTP handshake succeeds, accept_worker spawns)
- Both sides call `jaccl_init_mesh_auto()` with matching coordinator IP + port
- The timeout is in `MeshGroup::MeshGroup()` → `SideChannel::SideChannel()` → TCP connect/listen

**Likely causes (ranked by probability):**
1. **Dynamic port race:** Hub picks a port, tells m3u3 via HTTP, then both try to init. But the port binding in SideChannel might race — rank 0 needs to `listen()` BEFORE rank 1 tries to `connect()`. If rank 1 connects first (thread scheduling), it gets ECONNREFUSED and retries until timeout.
2. **Firewall:** macOS may block the dynamic port. Unlike port 9090 (whitelisted by asmi's long-running daemon), ephemeral ports may trigger the "accept incoming connections" dialog.
3. **Device mismatch:** `jaccl_init_mesh_auto` picks the "first device with PD budget." On hub that's `rdma_en3`, on m3u3 that might also be `rdma_en3` — but these are different physical devices. JACCL MeshGroup expects `device_names[peer_rank]` to name the device that CONNECTS to that peer. Auto-discovery might pick the wrong device.

## Tasks

### Task 1 — Add verbose logging to jaccl_init_mesh_auto — 15 min

**Edit:** `vendor/jaccl/jaccl_shim.cpp` — `jaccl_init_mesh_auto()`

Add stderr logging at each step:
- Which device was auto-discovered
- What coordinator address is being used
- Before/after MeshGroup constructor call
- Catch exception message if constructor throws

**Verify:** `cargo build --features jaccl`, deploy, test, read stderr

**Commit:** `[jaccl-ffi] add verbose logging to init_mesh_auto for handshake debugging`

### Task 2 — Test SideChannel TCP connectivity manually — 10 min

Before calling JACCL, verify the coordinator port is reachable:
- Hub: `nc -l <port>` 
- m3u3: `nc <hub_ip> <port>`
- If this fails, it's a firewall/routing issue, not JACCL

### Task 3 — Fix device selection for peer connectivity — 15 min

**Root cause hypothesis:** `jaccl_init_mesh_auto` picks the first device with PD budget. But for a 2-node transfer hub↔m3u3, we need the device that's physically connected to m3u3 — not just any device.

**Fix option A:** Use asmi's topology to find which `rdma_enX` connects to the peer, then pass that specific device name.

**Fix option B:** Let JACCL try all devices — modify `jaccl_init_mesh_auto` to iterate devices and try MeshGroup with each until one succeeds.

**Fix option C:** Skip auto-discovery. Write a proper devices JSON using asmi's `/topology` data (which maps `rdma_en15` ↔ `rdma_en10` etc). This is what `mlx.launch` does via the hostfile.

Decision: **Option C** — it's the most reliable. asmi already knows the topology. Generate the devices JSON dynamically from topology data, write to /tmp, pass to `jaccl_init_mesh`.

**Edit:** `src/transfer.rs` — before calling `jaccl_init_mesh_auto`, query `/topology` to find the correct device pair, generate devices JSON, use `jaccl_init_mesh` (not auto).

### Task 4 — Test end-to-end transfer — 10 min

**Verify:**
1. `curl -X POST localhost:9090/transfer -d '{"model_dir":"small-model","peer":"m3u3","direction":"send"}'`
2. SSE stream shows: preflight → coordinate → init → transfer → verify → done
3. Model files appear on m3u3
4. Speed > 1 GB/s

## Acceptance criteria

1. Transfer completes without timeout
2. SSE stream reaches `done` stage
3. Model files verified on target node
4. `cargo build --features jaccl` clean on both nodes

## Risk: PD exhaustion from failed attempts

Each failed `MeshGroup()` init burns PDs. We've already burned some this session. Check PD budget before each attempt:
```bash
/usr/local/bin/asmi-pd-probe  # or GET /rdma/health
```
If active devices show `pd_probe_raw: 0`, stop — cold reboot required.
