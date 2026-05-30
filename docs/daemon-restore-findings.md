# asmi daemon — failure modes & restore (field findings, 2026-05-30)

Evidence-based notes from debugging a cluster where `topology` showed phantom
nodes (`hub-3`, `m3u4-2237`, `mini2-3699`) and worker daemons went down.

## Root cause of the phantom names: Bonjour self-collision
`bonjour.rs` published the A/AAAA record under the OS's own `<hostname>.local.`
(via `host_fqdn = format!("{}.local.", hostname)` + `enable_addr_auto()`). That
collides with the system's own mDNS record → macOS mDNSResponder renames the
host's `LocalHostName` (`hub → hub-2 → hub-3`), and the daemon (which reads
`LocalHostName` once at startup) then reports the drifted name → phantom nodes.
**Fix:** publish under the unique service-instance name `r1o-<hostname>.local.`
(consumers read the canonical name from the `host=` TXT record).

## Deploy caveats (`asmi daemon deploy`)
1. **Copies the hub plist verbatim** → would push `--cluster` to workers. Workers
   must run plain `serve --port 9090`; only the hub aggregator adds `--cluster`.
2. **Ships a locally adhoc-signed binary** → passes until a reboot, then
   Gatekeeper/AMFI rejects it (`OS_REASON_CODESIGNING`, `spctl assess → rejected`)
   → crash-loop + launchd throttle. Re-sign/trust on the target
   (`codesign --force --sign - ~/.cargo/bin/asmi`) or ship a properly-signed build.

## Restore checklist (per node)
- Canonical label is `com.asmi.daemon` (legacy `com.r1o.asmi` points at the wrong
  binary path — keep disabled).
- Plist binds `0.0.0.0` (not `127.0.0.1`) so peers/aggregator can reach it.
- `ThrottleInterval=5` + `KeepAlive=true`: **wait ≥5s between `bootout` and
  `bootstrap`**, or rapid churn leaves the job "loaded but not running."
- Hostname is read once at startup → after `scutil --set LocalHostName <name>`,
  **restart the daemon** so it re-reads. Kill any orphan ex-launchd process
  squatting `:9090` (it caches the stale name and blocks the real daemon).
- **Do not edit `state.db`** to remove phantoms (single-writer WAL; no delete
  runbook). They collapse at the hub aggregator via `canonicalize_hostname` once
  each node reports its canonical `LocalHostName`.
- Health gate: `curl :9090/health` returns `{"hostname":"<canonical>","ok":true}`.
