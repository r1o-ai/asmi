#!/bin/bash
# Sign an asmi binary with the cluster's stable identity.
#
# macOS keys the Local Network permission (com.apple.networkextension) to the
# code identity. Ad-hoc signing (codesign -s -) derives the identity from the
# build hash, so every deploy creates a new "asmi-<hash>" identity — a fresh
# permission prompt and a stale Privacy entry per deploy (153 cleaned
# 2026-06-21, 136 more on 2026-07-06). A certificate identity with a fixed
# -i identifier keeps ONE entry forever.
#
# Usage: deploy/codesign.sh <path-to-asmi-binary>
# Sign on a machine whose login keychain holds the cert (sign as the user,
# never via sudo — root can't open the user keychain), then scp anywhere:
# signatures survive the copy.
set -euo pipefail

BINARY="${1:?usage: deploy/codesign.sh <path-to-asmi-binary>}"
IDENTITY="Apple Development: Mario A Iturrino (SDDNR9864J)"
IDENTIFIER="eu.r1o.asmi"

security find-identity -v -p codesigning | grep -q "$IDENTITY" || {
  echo "error: signing identity not in this keychain: $IDENTITY" >&2
  echo "hint: run on a machine that holds the cert (marmac/hub), then scp." >&2
  exit 1
}

codesign -f -s "$IDENTITY" -i "$IDENTIFIER" "$BINARY"

# Verify: refuse to bless anything that came out ad-hoc or mis-identified.
INFO=$(codesign -dv "$BINARY" 2>&1)
echo "$INFO" | grep -q "^Identifier=$IDENTIFIER$" || { echo "error: identifier mismatch"; echo "$INFO"; exit 1; }
echo "$INFO" | grep -q "^Signature=adhoc$" && { echo "error: still ad-hoc signed"; exit 1; }

echo "signed: $BINARY"
echo "$INFO" | grep -E "^(Identifier|TeamIdentifier)="
