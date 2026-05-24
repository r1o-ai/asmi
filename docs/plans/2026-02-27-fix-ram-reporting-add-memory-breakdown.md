# Fix RAM Reporting & Add Memory Breakdown

> **For Claude:** REQUIRED SUB-SKILL: Use executing-plans to implement this plan task-by-task.

**Goal:** Fix asmi's misleading RAM percentage (reports 96% "used" when actual app usage is 9%) by distinguishing app memory from file cache, and surface the breakdown in the r1o web dashboard.

**Architecture:** asmi's `parse_vmstat_and_memsize()` currently counts macOS speculative + inactive pages as "used memory." On Apple Silicon, the kernel aggressively caches files in unified memory — 224GB of cache on a 256GB machine is normal. The fix adds `ram_app_bytes` (active + wired + compressor) and `ram_cached_bytes` (speculative + inactive) fields to `NodeSnapshot`, changes `ram_percent` to reflect actual app usage, and propagates the breakdown through the web API to the dashboard UI.

**Tech Stack:** Rust (asmi daemon), TypeScript/Next.js (r1o web), TanStack Query, Tailwind CSS

**Repos:**
- `apple-smi/` — asmi daemon (Rust, Cargo workspace)
- `r1o/web/` — Next.js dashboard

---

## Task 1: Add memory breakdown fields to NodeSnapshot

**Files:**
- Modify: `crates/cluster-monitor/src/types.rs:17-62` (NodeSnapshot struct)

**Step 1: Add new fields to NodeSnapshot**

Add after `ram_percent` (line 43):

```rust
    // Memory breakdown (macOS vm_stat categories)
    // app = active + wired + compressor (what processes actually need)
    // cached = speculative + inactive (file cache, immediately reclaimable)
    #[serde(default)]
    pub ram_app_bytes: u64,
    #[serde(default)]
    pub ram_cached_bytes: u64,
```

The `#[serde(default)]` ensures backward compatibility — old JSON without these fields deserializes to 0.

**Step 2: Verify it compiles**

Run: `cargo check 2>&1 | head -20`
Expected: Errors in collector.rs where NodeSnapshot is constructed (missing fields). This is expected — we fix it in Task 2.

**Step 3: Fix the NodeSnapshot construction in collector.rs**

In `crates/cluster-monitor/src/collector.rs`, the `collect_via_ssh()` function constructs a `NodeSnapshot` at line 384. Add the two new fields with placeholder values (0) so it compiles:

```rust
    NodeSnapshot {
        // ... existing fields ...
        ram_used_bytes,
        ram_total_bytes,
        ram_percent,
        ram_app_bytes: 0,      // populated in Task 2
        ram_cached_bytes: 0,   // populated in Task 2
        // ... rest ...
    }
```

**Step 4: Verify it compiles cleanly**

Run: `cargo check`
Expected: Clean compile (warnings OK).

**Step 5: Commit**

```bash
git add crates/cluster-monitor/src/types.rs crates/cluster-monitor/src/collector.rs
git commit -m "feat: add ram_app_bytes and ram_cached_bytes to NodeSnapshot"
```

---

## Task 2: Fix parse_vmstat_and_memsize to return memory breakdown

**Files:**
- Modify: `crates/cluster-monitor/src/collector.rs:560-622` (parse_vmstat_and_memsize function)

**Step 1: Create a MemoryStats return struct**

Add above `parse_vmstat_and_memsize` (around line 560):

```rust
/// Breakdown of macOS memory categories from vm_stat.
#[derive(Debug, Clone, Default)]
pub struct MemoryStats {
    /// Total physical RAM from sysctl hw.memsize.
    pub total_bytes: u64,
    /// Legacy "used" = active + inactive + speculative + wired + compressor.
    /// Kept for backward compatibility. Includes file cache.
    pub used_bytes: u64,
    /// App memory = active + wired + compressor. What processes actually need.
    pub app_bytes: u64,
    /// Cached memory = speculative + inactive. File cache, immediately reclaimable.
    pub cached_bytes: u64,
}
```

**Step 2: Update parse_vmstat_and_memsize to return MemoryStats**

Change the function signature from:
```rust
pub fn parse_vmstat_and_memsize(text: &str) -> (u64, u64) {
```
to:
```rust
pub fn parse_vmstat_and_memsize(text: &str) -> MemoryStats {
```

Replace the last 3 lines (the calculation + return) with:

```rust
    let app_bytes = (active + wired + compressor) * page_size;
    let cached_bytes = (speculative + inactive) * page_size;
    let used_bytes = app_bytes + cached_bytes;

    MemoryStats {
        total_bytes,
        used_bytes,
        app_bytes,
        cached_bytes,
    }
```

**Step 3: Update the caller in collect_via_ssh**

In `collect_via_ssh()`, change the destructuring at line 170 from:

```rust
    let (resolved_hostname, ram_used_bytes, ram_total_bytes) = match &mem_res {
        Ok(r) if r.has_output() => {
            debug!(hostname, "vm_stat/sysctl OK");
            let (resolved, vmstat_text) = match r.stdout.split_once("---HOSTNAME---\n") {
                Some((h, rest)) => (Some(h.trim().to_string()), rest.to_string()),
                None => (None, r.stdout.clone()),
            };
            let (used, total) = parse_vmstat_and_memsize(&vmstat_text);
            (resolved, used, total)
        }
        Ok(r) => {
            debug!(hostname, stderr = r.stderr.as_str(), "vm_stat/sysctl empty/failed");
            (None, 0, 0)
        }
        Err(e) => {
            warn!(hostname, error = %e, "vm_stat/sysctl command error");
            (None, 0, 0)
        }
    };
```

to:

```rust
    let (resolved_hostname, mem_stats) = match &mem_res {
        Ok(r) if r.has_output() => {
            debug!(hostname, "vm_stat/sysctl OK");
            let (resolved, vmstat_text) = match r.stdout.split_once("---HOSTNAME---\n") {
                Some((h, rest)) => (Some(h.trim().to_string()), rest.to_string()),
                None => (None, r.stdout.clone()),
            };
            (resolved, parse_vmstat_and_memsize(&vmstat_text))
        }
        Ok(r) => {
            debug!(hostname, stderr = r.stderr.as_str(), "vm_stat/sysctl empty/failed");
            (None, MemoryStats::default())
        }
        Err(e) => {
            warn!(hostname, error = %e, "vm_stat/sysctl command error");
            (None, MemoryStats::default())
        }
    };
```

**Step 4: Update ram_percent and NodeSnapshot construction**

Change the ram_percent calculation (line 378) and NodeSnapshot (line 384):

```rust
    // ram_percent reflects actual app usage (excludes file cache)
    let ram_percent = if mem_stats.total_bytes > 0 {
        (mem_stats.app_bytes as f64 / mem_stats.total_bytes as f64) * 100.0
    } else {
        0.0
    };

    NodeSnapshot {
        hostname: canonical_hostname,
        online: true,
        timestamp: Utc::now(),
        chip_model: None,
        serial_number: None,
        model_name: None,
        cpu_watts: power.cpu_mw,
        gpu_watts: power.gpu_mw,
        ane_watts: power.ane_mw,
        cpu_percent: power.cpu_percent,
        gpu_percent: power.gpu_percent,
        ram_used_bytes: mem_stats.used_bytes,
        ram_total_bytes: mem_stats.total_bytes,
        ram_percent,
        ram_app_bytes: mem_stats.app_bytes,
        ram_cached_bytes: mem_stats.cached_bytes,
        cpu_temp_c: None,
        gpu_temp_c: None,
        processes,
        top_tasks: Vec::new(),
        rdma: rdma_status,
        interface_ips,
    }
```

**Step 5: Verify it compiles**

Run: `cargo check`
Expected: Clean compile.

**Step 6: Commit**

```bash
git add crates/cluster-monitor/src/collector.rs
git commit -m "fix: ram_percent now reflects app usage, not file cache"
```

---

## Task 3: Update tests

**Files:**
- Modify: `crates/cluster-monitor/src/collector.rs:807-833` (test_parse_vmstat_and_memsize)

**Step 1: Update the vmstat test**

Replace `test_parse_vmstat_and_memsize` with:

```rust
    #[test]
    fn test_parse_vmstat_and_memsize() {
        let vmstat = include_str!("../testdata/vmstat.txt");
        let sysctl = include_str!("../testdata/sysctl-hw.txt");
        let combined = format!("{vmstat}\n---MEMSIZE---\n{sysctl}");

        let stats = parse_vmstat_and_memsize(&combined);

        // Total should be 549755813888 (512 GiB)
        assert_eq!(stats.total_bytes, 549_755_813_888, "total_bytes");

        // App = active + wired + compressor = 11058419 + 1434855 + 422 = 12493696 pages
        let expected_app_pages: u64 = 11_058_419 + 1_434_855 + 422;
        let expected_app = expected_app_pages * 16384;
        assert_eq!(stats.app_bytes, expected_app, "app_bytes");

        // Cached = speculative + inactive = 133882 + 11287075 = 11420957 pages
        let expected_cached_pages: u64 = 133_882 + 11_287_075;
        let expected_cached = expected_cached_pages * 16384;
        assert_eq!(stats.cached_bytes, expected_cached, "cached_bytes");

        // Used = app + cached (backward compat)
        assert_eq!(stats.used_bytes, expected_app + expected_cached, "used_bytes = app + cached");

        // Sanity: app should be much smaller than total
        assert!(stats.app_bytes < stats.total_bytes / 2, "app_bytes should be < 50% of total");

        eprintln!(
            "Parsed: app={:.1} GiB, cached={:.1} GiB, used={:.1} GiB, total={:.1} GiB",
            stats.app_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            stats.cached_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            stats.used_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
            stats.total_bytes as f64 / (1024.0 * 1024.0 * 1024.0),
        );
    }
```

**Step 2: Update the custom page size test**

```rust
    #[test]
    fn test_parse_vmstat_custom_page_size() {
        let text = "Mach Virtual Memory Statistics: (page size of 4096 bytes)\n\
                     Pages active:                               100.\n\
                     ---MEMSIZE---\n\
                     8589934592\n";
        let stats = parse_vmstat_and_memsize(text);
        assert_eq!(stats.total_bytes, 8_589_934_592);
        // Only active pages — no inactive/speculative/wired/compressor
        assert_eq!(stats.app_bytes, 100 * 4096);  // active goes into app (with wired+compressor=0)
        assert_eq!(stats.cached_bytes, 0);
        assert_eq!(stats.used_bytes, 100 * 4096);
    }
```

**Step 3: Run tests**

Run: `cargo test -p asmi-core -- test_parse_vmstat 2>&1`
Expected: Both tests PASS.

**Step 4: Run full test suite**

Run: `cargo test 2>&1`
Expected: All tests PASS.

**Step 5: Commit**

```bash
git add crates/cluster-monitor/src/collector.rs
git commit -m "test: update vmstat tests for memory breakdown"
```

---

## Task 4: Build and deploy asmi to all nodes

**Files:**
- None modified (deployment step)

**Step 1: Build release binary**

Run: `cargo build --release 2>&1 | tail -5`
Expected: `Finished release` with no errors.

**Step 2: Deploy to all nodes**

The asmi binary lives at `target/release/asmi`. Deploy to each node and restart the daemon:

```bash
# Copy binary to each node
for node in m3u1 m3u3 m4m1; do
  scp target/release/asmi ${node}:~/bin/asmi
done
# m3u2 (local hub) — copy directly
cp target/release/asmi ~/bin/asmi

# Restart daemons
for node in m3u1 m3u3 m4m1; do
  ssh $node "pkill -f 'asmi.*daemon' 2>/dev/null; sleep 1; nohup ~/bin/asmi daemon > /tmp/asmi-daemon.log 2>&1 &"
done
# Local (m3u2)
pkill -f 'asmi.*daemon' 2>/dev/null; sleep 1; nohup ~/bin/asmi daemon > /tmp/asmi-daemon.log 2>&1 &
```

**Step 3: Verify new fields on each node**

```bash
for node in m3u1 m3u2 m3u3; do
  echo "=== $node ==="
  curl -s http://${node}.local:9090/metrics | python3 -c "
import json,sys; d=json.load(sys.stdin)
app=d.get('ram_app_bytes',0)/1e9
cached=d.get('ram_cached_bytes',0)/1e9
pct=d.get('ram_percent',0)
total=d.get('ram_total_bytes',0)/1e9
print(f'  ram_percent={pct:.1f}% app={app:.1f}GB cached={cached:.1f}GB total={total:.0f}GB')
"
done
```

Expected: `ram_percent` should be single-digit % for idle nodes, `cached` should be large.

---

## Task 5: Update r1o web types and API

**Files (all in `r1o/web/`):**
- Modify: `src/lib/asmi-client.ts:17-31` (AsmiSnapshot type)
- Modify: `src/types/metrics.ts:35-53` (PowerMetricsSnapshot type)
- Modify: `src/app/api/cluster/powermetrics/[hostname]/route.ts:54-68` (snapshot mapping)

**Step 1: Add fields to AsmiSnapshot**

In `src/lib/asmi-client.ts`, add to the `AsmiSnapshot` interface after `ram_percent`:

```typescript
  ram_app_bytes?: number;
  ram_cached_bytes?: number;
```

**Step 2: Add fields to PowerMetricsSnapshot**

In `src/types/metrics.ts`, add to the `PowerMetricsSnapshot` interface after `ram_total_gb`:

```typescript
  /** App memory in GB (active + wired + compressor — what processes need) */
  ram_app_gb?: number;
  /** Cached memory in GB (speculative + inactive — file cache, reclaimable) */
  ram_cached_gb?: number;
```

**Step 3: Map new fields in powermetrics route**

In `src/app/api/cluster/powermetrics/[hostname]/route.ts`, add to the snapshot object after `ram_total_gb`:

```typescript
    ram_app_gb: data.ram_app_bytes
      ? Math.round(data.ram_app_bytes / 1_073_741_824 * 10) / 10
      : undefined,
    ram_cached_gb: data.ram_cached_bytes
      ? Math.round(data.ram_cached_bytes / 1_073_741_824 * 10) / 10
      : undefined,
```

Also fix `ram_percent` to prefer the asmi-calculated value (which now uses app_bytes):

The existing line `ram_percent: Math.round(data.ram_percent * 10) / 10,` is already correct — asmi now sends the fixed value. No change needed.

**Step 4: Type check**

Run: `npx tsc --noEmit`
Expected: Clean.

**Step 5: Commit**

```bash
git add src/lib/asmi-client.ts src/types/metrics.ts src/app/api/cluster/powermetrics/\\[hostname\\]/route.ts
git commit -m "feat: surface ram_app_gb and ram_cached_gb from asmi"
```

---

## Task 6: Update dashboard UI to show memory breakdown

**Files (all in `r1o/web/`):**
- Modify: `src/components/topology/NodePowerCharts.tsx` (RAM gauge)
- Modify: `src/components/topology/NodeMetricsBadges.tsx` (RAM badge text)

**Step 1: Find where RAM is displayed**

Search for `ram_percent` or `ram_used_gb` in the topology components to find the exact rendering locations.

**Step 2: Update NodeMetricsBadges to show cached**

Where the RAM badge currently shows something like `245 / 256 GB (96%)`, change to show:
- Main: `24 / 256 GB (9%)` — app usage
- Subtitle: `224 GB cached` — file cache

Use `ram_app_gb` if available, fall back to `ram_used_gb`.

**Step 3: Update NodePowerCharts RAM gauge**

If there's a circular gauge or progress bar for RAM, make it display `ram_app_gb / ram_total_gb` as the primary fill, with an optional lighter segment for cached. If the gauge is simple (just a percent), use the corrected `ram_percent`.

**Step 4: Verify in browser**

Open `http://localhost:59408` and check:
- Node detail panel RAM values are single-digit % for idle nodes
- Cached memory is visible somewhere in the UI
- m3u1 should show higher app usage (Qwen3.5 is loaded — ~208GB footprint)

**Step 5: Commit**

```bash
git add src/components/topology/NodePowerCharts.tsx src/components/topology/NodeMetricsBadges.tsx
git commit -m "feat: show app vs cached memory breakdown in dashboard"
```

---

## Verification Checklist

After all tasks:

- [ ] `cargo test` passes in apple-smi
- [ ] asmi daemon on m3u1 reports ~40% ram_percent (Qwen3.5 loaded = ~208GB app)
- [ ] asmi daemon on m3u3 reports ~9% ram_percent (no models loaded)
- [ ] Web dashboard shows accurate RAM for all nodes
- [ ] Cached memory is visible in the UI
- [ ] `npx tsc --noEmit` passes in r1o/web
