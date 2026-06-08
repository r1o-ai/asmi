#!/bin/bash
# Tailscale Join Script — run this on Bill's Mac
# Joins the machiabeli@ tailnet (chaco-burbot.ts.net)

set -e

echo "=== Tailscale Setup for fib0 Cluster ==="
echo ""

# 1. Install Tailscale if not present
if ! command -v tailscale &>/dev/null; then
    echo "[1/3] Installing Tailscale via Homebrew..."
    if ! command -v brew &>/dev/null; then
        echo "ERROR: Homebrew not found. Install from https://brew.sh first."
        exit 1
    fi
    brew install --cask tailscale
    echo "  → Installed. Open Tailscale.app from Applications to start the service."
    open -a Tailscale
    echo "  → Waiting 5s for Tailscale daemon to start..."
    sleep 5
else
    echo "[1/3] Tailscale already installed ✓"
fi

# 2. Check if daemon is running
if ! tailscale status &>/dev/null; then
    echo "  → Tailscale daemon not running. Opening Tailscale.app..."
    open -a Tailscale
    echo "  → Waiting 10s for daemon..."
    sleep 10
fi

# 3. Login
echo "[2/3] Logging in..."
echo "  → A browser window will open. Sign in with your fib0.ai account."
echo ""
tailscale login

# 4. Verify
echo ""
echo "[3/3] Verifying connection..."
sleep 3
tailscale status

echo ""
echo "=== Done ==="
echo "Your Tailscale IP: $(tailscale ip -4 2>/dev/null || echo 'pending')"
echo ""
echo "Test connectivity to hub:"
echo "  tailscale ping hub"
echo "  ssh ma@hub"
