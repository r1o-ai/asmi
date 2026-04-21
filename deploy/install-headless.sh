#!/bin/bash
# install-headless.sh — install asmi as a system LaunchDaemon on a headless node.
#
# Use when `stat -f '%Su' /dev/console` shows `root`, i.e. the node boots
# straight to the login screen and nobody has ever signed in. See
# deploy/com.asmi.daemon.launchdaemon.plist for rationale.
#
# Run this on the TARGET node (not the orchestrator). Requires passwordless sudo.
#
# Prereqs already done: asmi binary at /Users/ma/.cargo/bin/asmi, codesigned,
# +x, and tested with `asmi --version` returning 0.

set -euo pipefail

PLIST_SOURCE="$(dirname "$0")/com.asmi.daemon.launchdaemon.plist"
PLIST_DEST="/Library/LaunchDaemons/com.asmi.daemon.plist"
USER_AGENT="$HOME/Library/LaunchAgents/com.asmi.daemon.plist"

if [[ ! -f "$PLIST_SOURCE" ]]; then
    echo "error: plist source not found at $PLIST_SOURCE" >&2
    exit 1
fi

if [[ "$(stat -f '%Su' /dev/console)" != "root" ]]; then
    echo "warning: console user is not root — this node has a user session." >&2
    echo "         You probably want the user LaunchAgent variant instead." >&2
    echo "         Press Ctrl-C within 5s to abort." >&2
    sleep 5
fi

# Move existing user-domain agent aside (if any) to prevent label collision.
if [[ -f "$USER_AGENT" ]]; then
    echo "sidelining existing user LaunchAgent"
    mv "$USER_AGENT" "${USER_AGENT}.disabled"
fi

# Kill any bare asmi process already running (manual launch, shell, etc.)
echo "clearing any bare asmi process"
pkill -9 -f '/asmi --serve' 2>/dev/null || true
sleep 2

# Install the LaunchDaemon plist with the correct ownership.
echo "installing $PLIST_DEST"
sudo cp "$PLIST_SOURCE" "$PLIST_DEST"
sudo chown root:wheel "$PLIST_DEST"
sudo chmod 644 "$PLIST_DEST"

# Bootstrap into the system domain.
echo "bootstrapping"
sudo launchctl bootstrap system "$PLIST_DEST"

# Verify.
sleep 4
echo "verifying"
if curl -s -m 5 localhost:9090/health | grep -q '"ok":true'; then
    echo "asmi LaunchDaemon healthy on $(hostname)"
    exit 0
else
    echo "asmi did not come up; check /Users/ma/Library/Logs/asmi-daemon.log" >&2
    exit 1
fi
