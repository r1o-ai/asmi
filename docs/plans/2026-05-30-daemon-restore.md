# Implementation plan — asmi daemon restore + Bonjour self-collision fix

**Date:** 2026-05-30
**Companion research:** `~/.claude/projects/-Users-ma-Projects-Tropical/memory/research_asmi_daemon_restore_2026-05-30.md`
**Working branch:** `fix/snapshot-canonicalize`
**Repo:** `/Users/ma/Projects/Personal/apple-smi`

---

## Working protocol
Apply dependency-scan before multi-file edits. Iron Laws: every fact cites the research memo; every task has a verification gate; fix root cause (Bonjour host-record collision), not symptoms.

## Architecture
Single code change (`bonjour.rs`: publish under `r1o-<host>.local.`, not the OS's `<host>.local.`) + an **operational** per-node daemon restore. The phantom topology entries are NOT edited in `state.db` — they collapse at the hub aggregator (`canonicalize_hostname`) once every node reports its canonical `LocalHostName`.

## Tech stack
Rust (asmi @ HEAD `8b2f628`, branch `fix/snapshot-canonicalize`), `mdns-sd 0.20`, macOS `launchd` (`com.asmi.daemon`, `ThrottleInterval=5`, `KeepAlive=true`), `scutil`, `codesign`.

## Tasks

### Phase A — Code fix (root cause)

#### A1 — Publish Bonjour under the unique instance name — DONE (uncommitted)
**Edit:** `src/bonjour.rs:148` → `let host_fqdn = format!("{}.local.", instance);` (was `hostname`).
**Why:** publishing the OS's own `<host>.local.` collides → mDNSResponder renames `LocalHostName` (research §root-cause).
**Verify:** `cargo build --release` exits 0; on a node, after restart, `scutil --get LocalHostName` holds canonical for ≥60s (proven on hub).
**Commit:** `[bonjour] publish A-record under r1o-<host> instance name to stop OS LocalHostName self-collision`

### Phase B — m3u2 (service not loaded + binds 127.0.0.1)
#### B1 — Fix bind + load daemon
**Pre-flight:** confirm `~/Library/LaunchAgents/com.asmi.daemon.plist` ProgramArguments has `--bind 0.0.0.0` (not `127.0.0.1`) and **no** `--cluster` (worker). Patch the plist if wrong.
**Run (on m3u2):** `launchctl bootout gui/$(id -u)/com.asmi.daemon 2>/dev/null; sleep 6; launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.asmi.daemon.plist`
**Verify:** `curl -s localhost:9090/health` → `{"hostname":"m3u2","ok":true}`; `lsof -nP -iTCP:9090 -sTCP:LISTEN` shows `asmi` on `*:9090`.

### Phase C — m3u3 (codesign/AMFI rejection of the adhoc binary)
#### C1 — Re-sign / trust the binary
**Run (on m3u3):** `codesign --force --sign - ~/.cargo/bin/asmi ~/.cargo/bin/asmi-helper; xattr -dr com.apple.quarantine ~/.cargo/bin/asmi 2>/dev/null; spctl --assess --type execute -v ~/.cargo/bin/asmi`
**Verify:** `codesign -v ~/.cargo/bin/asmi` exits 0; `~/.cargo/bin/asmi --version` runs without `Killed: 9`.
#### C2 — Clear crash-loop throttle + bootstrap
**Run:** `launchctl bootout gui/$(id -u)/com.asmi.daemon 2>/dev/null; sleep 10` (clear the 50-run throttle) `; launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.asmi.daemon.plist`
**Verify:** `launchctl print gui/$(id -u)/com.asmi.daemon | grep -E 'state|runs'` shows `state=running`; `curl localhost:9090/health` → `{"hostname":"m3u3","ok":true}`.

### Phase D — mini2 (orphan process squatting :9090)
#### D1 — Kill the orphan, let the real daemon bind
**Pre-flight:** identify the orphan: `lsof -nP -iTCP:9090 -sTCP:LISTEN` → the PID NOT managed by launchd (research: PID 14063, started 03:06, caches `mini2-3699`).
**Run (on mini2):** `kill <orphan_pid>; sleep 3; launchctl kickstart -k gui/$(id -u)/com.asmi.daemon`
**Verify:** `curl localhost:9090/health` → `{"hostname":"mini2","ok":true}` (NOT `mini2-3699`).

### Phase E — Verify the mesh self-heals
#### E1 — Confirm phantoms collapse at the aggregator
**Run (on hub):** `launchctl kickstart -k gui/$(id -u)/com.asmi.daemon; sleep 20; asmi topology`
**Verify:** node list shows only canonical `hub/m3u2/m3u3/mini2` (+ m3u4/marmac when up); **no** `hub-3`/`m3u4-2237`/`mini2-3699`; link count reflects real nodes.

## File touch matrix
| File | Δ | Notes |
|---|---|---|
| `src/bonjour.rs` | ~1 line | the only code change |
| `docs/daemon-restore-findings.md` | new | field notes |
| (ops) launchd plists per node | edit if `127.0.0.1`/`--cluster` | not in repo |

## Risk register
| Risk | Mitigation |
|---|---|
| `codesign --sign -` (adhoc) re-rejected after next reboot | proper signing identity, or a launchd `--keepalive` trust exception; track as follow-up |
| Killing orphan on mini2 disrupts gateway briefly | KeepAlive respawns the launchd daemon immediately |
| Throttle not cleared (<5s wait) | always `sleep ≥6-10s` between bootout/bootstrap |

## Rollback strategy
| Failure | Action |
|---|---|
| A node won't serve after restore | re-check plist bind/`--cluster`, codesign, throttle in that order; daemon was healthy pre-incident so config is recoverable |
| bonjour fix regresses naming | `git revert` the A1 commit; names already proven stable on hub |

## Acceptance criteria
1. m3u2, m3u3, mini2 each return `{"hostname":"<canonical>","ok":true}` on `:9090`.
2. `asmi topology` shows zero `-N` phantom nodes.
3. hub `LocalHostName` + reported name stay canonical ≥60s after restart (already proven).
4. `bonjour.rs` fix committed to `fix/snapshot-canonicalize`.

## Out of scope
- m3u4 (LAN unreachable — console needed) and marmac (offline).
- Hardening `asmi daemon deploy` to sign binaries + emit worker-specific plists (follow-up issue).
- TB5 physical cabling / JACCL link bring-up (separate, physical).
