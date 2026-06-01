# asmi-helper deployment — install on a new node

Eliminates the per-poll `sudo powermetrics` fork-bomb that pinned
PerfPowerServices on 2026-05-30. One persistent root-owned powermetrics
exposes CPU / GPU / ANE telemetry on a Unix socket; any user-space asmi
on the same machine reads from there.

## Already deployed (no action needed)

| Node | Helper | Asmi binary (gated) | Notes |
|------|--------|----------------------|-------|
| hub  | ✓ | ✓ | Reference install |
| m3u3 | ✓ | ✓ | |
| m3u4 | ✓ | ✓ | |
| mini2 | ✓ | ✓ | Was the noisiest — 2-day-old stale zombies cleaned |

## Pending

| Node | Blocker | What's needed |
|------|---------|---------------|
| **marmac** | Laptop asleep / unreachable | Wake the laptop, run the procedure below |
| **m3u2** | SSH key rejected for both `ma@10.1.10.70` and `machiabeli@10.1.10.70`; HTTP asmi API also closed (port 9090 not reachable from cluster) | Add hub's public ED25519 key (`ssh-add -L | head -1`) to the target user's `~/.ssh/authorized_keys` on m3u2, then run the procedure below as that user |

Hub's key for the `authorized_keys` line:
```
ssh-ed25519 AAAA... ma@Marios-MacBook-Pro.local
```
(get the full key via `ssh-add -L | head -1` on hub)

## Procedure (one-time, ~2 min per node)

From **hub** (after SSH access is established), with `<NODE>` being the
target hostname:

```bash
# 1. Sync binary + plist
scp ~/.cargo/bin/asmi-helper <NODE>:/tmp/asmi-helper-new
scp ~/Projects/Personal/apple-smi/launchd/com.r1o.asmi-helper.plist <NODE>:/tmp/com.r1o.asmi-helper.plist
scp /tmp/asmi-helper-install.sh <NODE>:/tmp/asmi-helper-install.sh  # see below

# 2. Execute the install script remotely
ssh <NODE> "bash /tmp/asmi-helper-install.sh"

# 3. Verify
ssh <NODE> "nc -U /var/run/eu.r1o.asmi.sock | head -1"
# Expect: {"ane_mw":...,"cpu_mw":...,"gpu_mw":...,"cpu_percent":...,...}
```

If the node has an outdated asmi binary (md5 != hub's `1662e7314cfe5969f9d3787cab33e9f1`),
also `scp ~/.cargo/bin/asmi <NODE>:/tmp/asmi-new && ssh <NODE> "cp -f /tmp/asmi-new ~/.cargo/bin/asmi && chmod 755 ~/.cargo/bin/asmi && launchctl kickstart -k gui/$(id -u)/com.asmi.daemon"`.

## Install script (`/tmp/asmi-helper-install.sh` on hub already)

```bash
#!/bin/bash
set +e
cp -f /tmp/asmi-helper-new ~/.cargo/bin/asmi-helper
chmod 755 ~/.cargo/bin/asmi-helper
sudo cp /tmp/com.r1o.asmi-helper.plist /Library/LaunchDaemons/com.r1o.asmi-helper.plist
sudo chown root:wheel /Library/LaunchDaemons/com.r1o.asmi-helper.plist
sudo chmod 644 /Library/LaunchDaemons/com.r1o.asmi-helper.plist
sudo launchctl bootout system /Library/LaunchDaemons/com.r1o.asmi-helper.plist 2>/dev/null
sleep 1
sudo launchctl bootstrap system /Library/LaunchDaemons/com.r1o.asmi-helper.plist
```

## What you should NOT do

- Don't try to "fix" the per-poll powermetrics shellout with asmi env
  vars — the architecture is now "helper broadcasts, asmi consumes."
  Old workarounds (`ANE_POWER_CHECK=0`, etc.) are obsolete.
- Don't run `asmi-helper` manually as your user — it needs root to read
  cpu_power. The plist runs it as root via launchd.
