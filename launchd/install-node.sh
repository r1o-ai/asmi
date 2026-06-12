#!/bin/bash
# Install asmi-helper LaunchDaemon on a remote node.
# Expects /tmp/asmi-helper-new + /tmp/com.r1o.asmi-helper.plist to be pre-staged.
set +e

# 1. Move new binary into place
cp -f /tmp/asmi-helper-new ~/.cargo/bin/asmi-helper
chmod 755 ~/.cargo/bin/asmi-helper
echo "  binary: $(stat -f '%Sm %z bytes' ~/.cargo/bin/asmi-helper)"

# 2. Install plist (sudo)
sed "s|__ASMI_HOME__|$HOME|g" /tmp/com.r1o.asmi-helper.plist | sudo tee /Library/LaunchDaemons/com.r1o.asmi-helper.plist > /dev/null
sudo chown root:wheel /Library/LaunchDaemons/com.r1o.asmi-helper.plist
sudo chmod 644 /Library/LaunchDaemons/com.r1o.asmi-helper.plist
echo "  plist installed"

# 3. Bootstrap (bootout first in case of stale)
sudo launchctl bootout system /Library/LaunchDaemons/com.r1o.asmi-helper.plist 2>/dev/null
sleep 1
sudo launchctl bootstrap system /Library/LaunchDaemons/com.r1o.asmi-helper.plist 2>&1 | head -3
sleep 4

# 4. Verify
echo -n "  helper state: "
sudo launchctl print system/com.r1o.asmi-helper 2>&1 | grep -m1 "state =" | tr -s " "
echo -n "  socket: "
[ -S /var/run/eu.r1o.asmi.sock ] && echo present || echo MISSING
echo -n "  socket sample: "
nc -U /var/run/eu.r1o.asmi.sock 2>&1 | head -1 | cut -c1-120

# 5. Kill residual fork-bomb zombies + recheck
sudo pkill -f "sudo powermetrics" 2>/dev/null
sleep 1
echo -n "  sudo-powermetrics zombies after: "
pgrep -af "sudo powermetrics" | wc -l
