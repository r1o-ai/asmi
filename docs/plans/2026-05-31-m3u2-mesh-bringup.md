# Plan: Bring m3u2 into the RDMA mesh (3-node → 4-node)

**Date:** 2026-05-31
**Repo:** `/Users/ma/Projects/Personal/apple-smi` (main @ `0937531`)
**Status:** ready to execute (awaiting go — this re-converges the live 3-node mesh)

---

## Why

m3u2 is a fully capable cluster node that has been **dark all session** — not by design, just because its asmi daemon was down + running the old binary, and it was off the LAN at session start. Discovery (this session):

| Fact | Value |
|------|-------|
| RAM | **512 GB** (serving-capable; matches hub/m3u4) |
| LAN | `10.1.10.70` (reachable now; `/etc/hosts` already correct) |
| Tailscale | `100.119.116.53` (fallback path) |
| Identity | clean — `HostName=m3u2`, `LocalHostName=m3u2` (no clone issue, unlike m3u4 which had hub's identity) |
| console user | `root` → **headless → system LaunchDaemon** variant |
| Model | **already has GLM 5.1** + 1.3 TB free (→ can be a 3rd serving node) |
| TB5 cabling | **wired to hub AND m3u4** — domain-UUID match: m3u2 recept3 → hub (`1FA7E985`), m3u2 recept2 → m3u4 (`356FB55C`). m3u2 ↔ m3u3 is **NOT** cabled (only 2 active ports). |
| asmi | **daemon DOWN**; binary is the **old build** `d855679c` (no jaccl `/transfer`, old `192.168.0.x` scheme with duplicate IPs) |

## Architecture: 3-node → 4-node

Current mesh (validated, self-healing): **3 links** — hub↔m3u3, hub↔m3u4, m3u3↔m3u4.

Target mesh: **4 nodes / 5 links** — adds **hub↔m3u2** and **m3u2↔m3u4**. (m3u2↔m3u3 absent — connected graph, not full mesh; add a cable later if a full 4-mesh is wanted.)

**Key consequence:** `assign_topology_ips` derives IPs as `192.168.10.{4*i+1, +2}` per sorted link index `i`. Adding m3u2 changes the link set from 3 → 5, so `i` **re-indexes** and **every node's mesh IPs shuffle**. This is a full re-converge of the working 3-node mesh. The durability fix (`discover_stable_topology` + canonicalize) makes it self-heal, but it is a live reshuffle — hence the gate.

Do **NOT** hardcode the new IP map. Derive it from `asmi topology --format json` after m3u2 joins (the device matrix per link, and thus which `rdma_enX` gets which `/30`, comes from live discovery — see Task 6).

## Constraints (carried from this session — do not violate)

- **Coordinator IP = the RDMA hardware port's `/30` IP** (GID basis; `transfer.rs:646-650`). Never blanket-flush a linked port's IP; never substitute a LAN/169.254 coordinator. The per-port `/30` is load-bearing.
- **Re-sign the binary on EVERY node after replacing it** (`cp`/`scp` invalidates the signature → `OS_REASON_CODESIGNING` spawn-fail). `codesign --remove-signature && codesign --force --sign -`.
- **Restart asmi SEQUENTIALLY** (peers up) so each derives the same topology — simultaneous restart races (partial discovery → static fallback). retry-stable mitigates but sequential is clean.
- **PD budget:** autosetup's `mlx.distributed_config` is PD-safe (jaccl-debug Phase 2); only real `/transfer` tests consume PDs — cap at ≤2.
- **One launchd variant per node:** m3u2 is headless → system LaunchDaemon only; disable any present-but-unloaded gui agent plist.

---

## Tasks

### Task 1 — Pre-flight (no changes)
```bash
ssh ma@10.1.10.70 'echo reachable; scutil --get LocalHostName; sysctl -n hw.memsize | awk "{printf \"%.0fGB\n\",\$1/1073741824}"; [ -d ~/Models/GLM-5.1-MLX-4.8bit-INF ] && echo "GLM present"; stat -f "%Su" /dev/console'
```
**Verify:** reachable, `LocalHostName=m3u2`, 512GB, GLM present, console=root. If LAN down, fall back to Tailscale `100.119.116.53` and stop — fix LAN first.

### Task 2 — Stage the unified binary onto m3u2 (+ re-sign)
```bash
R=/Users/ma/Projects/Personal/apple-smi
scp -q "$R/target/release/asmi" ma@10.1.10.70:/tmp/asmi.unified
ssh ma@10.1.10.70 'set -e
  cp ~/.cargo/bin/asmi ~/.cargo/bin/asmi.bak-preunified-$(date +%s)
  codesign --remove-signature /tmp/asmi.unified 2>/dev/null || true
  codesign --force --sign - /tmp/asmi.unified
  /tmp/asmi.unified --version
  mv /tmp/asmi.unified ~/.cargo/bin/asmi'
```
**Verify:** `asmi --version` prints (exit 0, not 137). (If `target/release/asmi` is stale, rebuild on hub: `cargo build --release --features jaccl` first.)

### Task 3 — Install + start the system LaunchDaemon (asmi is DOWN)
```bash
ssh ma@10.1.10.70 '
  # ensure single variant: disable any gui agent plist (headless can not load it anyway)
  [ -f ~/Library/LaunchAgents/com.asmi.daemon.plist ] && mv ~/Library/LaunchAgents/com.asmi.daemon.plist{,.disabled}
  # install the system LaunchDaemon if missing (from deploy/ in the repo) and bootstrap
  sudo -n cp ~/Projects/Personal/apple-smi/deploy/com.asmi.daemon.launchdaemon.plist /Library/LaunchDaemons/com.asmi.daemon.plist 2>/dev/null || true
  sudo -n launchctl bootout system/com.asmi.daemon 2>/dev/null || true
  sudo -n launchctl bootstrap system /Library/LaunchDaemons/com.asmi.daemon.plist'
sleep 8
curl -sf -m5 http://10.1.10.70:9090/health | python3 -c "import sys,json;d=json.load(sys.stdin);print(d['hostname'],'v'+d['version'])"
curl -s -m5 -X POST http://10.1.10.70:9090/transfer -H 'Content-Type: application/json' -d '{"model_dir":"__probe__","peer":"x","direction":"send"}' | head -c 80
```
**Verify:** `/health` → `m3u2`; `/transfer` probe reaches `jaccl-rdma` (confirms jaccl build live). 1 asmi process on m3u2.

### Task 4 — Fix m3u2's stale TB5 IPs (old 0.x dups) — flush before re-converge
```bash
ssh ma@10.1.10.70 'for i in en3 en4 en5 en6; do for ip in $(ifconfig $i 2>/dev/null|awk "/inet /{print \$2}"); do sudo -n ifconfig $i delete $ip; done; done; echo flushed'
```
**Verify:** no `192.168.x` IPs linger on m3u2's TB5 ports (the autosetup will assign clean 10.x in Task 5).

### Task 5 — Coordinated 4-node re-converge (SEQUENTIAL restart)
Restart each node's asmi one at a time (peers up) so all four discover the 4-node topology and `retry-stable` derives the same 5-link map. hub/m3u3 = gui agent kickstart; m3u4/m3u2 = headless → `pkill` (KeepAlive respawn).
```bash
# m3u2 (headless): KeepAlive respawn
ssh ma@10.1.10.70 'pkill -f "asmi.*--serve"'; sleep 25
# m3u3 (gui): kickstart; sleep 25
ssh m3u3 'launchctl kickstart -k gui/$(id -u)/com.asmi.daemon'; sleep 25
# m3u4 (headless): KeepAlive respawn; sleep 25
ssh m3u4 'pkill -f "asmi.*--serve"'; sleep 25
# hub (gui): kickstart
launchctl kickstart -k gui/$(id -u)/com.asmi.daemon; sleep 30
```
**Verify:** `asmi topology --hosts hub,m3u2,m3u3,m3u4 --format json` → **5 links**, nodes `[hub,m3u2,m3u3,m3u4]`. Per-node IPs: unique, single `192.168.10.x` scheme, no dups (the durability probe pattern).

### Task 6 — Re-derive + set static_ips for the 4-node map (coordinator-IP-safe fallback)
Read the live assigned IPs (NOT hardcoded) and write them as each node's `static_ips`:
```bash
# For each node: read its current en3/en4/en5 192.168.10.x assignments, write them into
# ~/.r1o/settings.json rdma.static_ips (iface, ip, mask 255.255.255.252) via python json load/dump.
# (Same per-node stdin-piped python approach proven this session.)
```
**Verify:** each node's `static_ips` == its live linked-port IPs; cross-node /30 pairs consistent; no collisions.

### Task 7 — Functional + durability validation
```bash
# (a) extend the probe suite to 4 nodes (build, jaccl x4, health x4, IPs-clean x4, topology=5 links)
# (b) one RDMA transfer touching m3u2 (e.g. hub -> m3u2, a small model) — expect jaccl-rdma + verified
# (c) 4-node simultaneous-restart self-heal test: flush-all + restart all 4 together -> reconverges to
#     clean 5-link mesh with zero manual intervention (validates durability at 4 nodes)
```
**Verify:** probes 0 errors; transfer verified + lands; simultaneous-restart self-heals clean.

### Task 8 — Record
Update memory `asmi_rdma_mesh_ground_truth_2026_05_31`: mesh is now 4-node/5-link; m3u2 unified + system-daemon + clean identity; m3u2↔m3u3 not cabled. Note m3u2 as a 3rd GLM-5.1 serving node.

---

## Rollback

- **Revert m3u2 only:** `sudo launchctl bootout system/com.asmi.daemon` on m3u2 (stops it) + restore `~/.cargo/bin/asmi.bak-preunified-*`. The other 3 nodes re-converge to the 3-node/3-link map on their next (sequential) restart.
- **Revert IPs:** static_ips backups + a sequential restart of the 3 remaining nodes returns the known-good 3-node map (hub en3=.1/en5=.5, m3u3 en4=.2/en3=.9, m3u4 en4=.6/en3=.10).
- Nothing here is irreversible: binaries backed up per node, IPs are software-assigned, topology is hardware-discovered.

## Out of scope (deferred)
- **A — verify-skip flag** (`transfer.rs` `verify:Option<bool>`): keep deferred.
- **Full 4-mesh** (m3u2↔m3u3 direct link): needs a physical TB5 cable; current connected graph is sufficient for transfers.
- **Distributed serve across 4×512GB** (hub/m3u2/m3u4 + m3u3): separate effort; m3u2 having GLM 5.1 makes a hub+m3u2+m3u4 TP3 attractive once the mesh is 4-node.
