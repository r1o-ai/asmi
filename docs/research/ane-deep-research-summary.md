# ANE Deep Research Summary

> Canonical reference for the asmi ANE research effort, March 2026.

---

## 1. Executive Summary

**asmi** (apple-smi) is a Rust-based system monitoring and inference orchestration tool for Apple Silicon clusters. It exposes hardware telemetry, model management, and distributed inference coordination as a lightweight HTTP daemon. The ANE (Apple Neural Engine) represents a massively underutilized compute resource on Apple Silicon -- 19 TFLOPS at 2-3W per die -- but Apple provides no public API for direct dispatch. asmi aims to be the first tool to expose both ANE telemetry and direct ANE compute as a network service, using private APIs that bypass CoreML's 2-4x overhead.

This research effort, conducted across multiple sessions in late February and early March 2026, mapped the full ANE hardware landscape across our 5-node cluster (8 ANE dies, ~92 TFLOPS total), discovered Egor Bokhan's `ane` Rust crate that achieves direct ANE dispatch with a working GPT-2 forward pass at 45.6 tok/s, reverse-engineered M3 Ultra's dual-ANE topology via ioreg, and identified MoE (Mixture of Experts) models as the ideal target architecture for ANE inference given the ~119 compile budget constraint.

The key insight is that ANE's fixed compile budget makes dense models impractical beyond ~7B parameters, but MoE models -- where only a fraction of parameters are active per token -- sidestep this entirely. Combined with speculative decoding strategies proven by Apple and SqueezeBits, and the cluster's 1.5TB unified memory pool connected via RDMA, asmi is positioned to deliver ANE-accelerated distributed inference that no other tool provides.

---

## 2. Cluster Hardware

### Node Inventory

| Node | Chip | RAM | ANE Dies | ANE TFLOPS | Memory BW | Role |
|------|------|-----|----------|------------|-----------|------|
| m3u1 | M3 Ultra | 512 GB | 2 (ane0 + ane1) | ~18 | 800 GB/s | Compute |
| m3u2 | M3 Ultra | 512 GB | 2 (ane0 + ane1) | ~18 | 800 GB/s | Coordinator |
| m3u3 | M3 Ultra | 256 GB | 2 (ane0 + ane1) | ~18 | 800 GB/s | Compute |
| m4m1 | M4 Max | 128 GB | 1 | 19 | 546 GB/s | Compute |
| m4m-b-1 | M4 Max | 128 GB | 1 | 19 | 546 GB/s | Dev/Build |

**Totals:** 5 nodes, 1.5 TB RAM, 8 ANE dies, ~92 TFLOPS ANE (FP16), ~3.5 TB/s aggregate memory bandwidth.

### Interconnect

- RDMA mesh via Thunderbolt 5 (TB5)
- 5 of 6 links active (missing m3u1 <-> m3u3 -- cable needed)
- JACCL-ready 3-node subsets identified via `asmi topology`
- Topology discovered live by wrapping `mlx.distributed_config --dot`

---

## 3. Bokhan's ANE Crate Discovery

**Repository:** [github.com/computer-graphics-tools/ane](https://github.com/computer-graphics-tools/ane)
**Author:** Egor Bokhan | **Language:** Rust | **License:** MIT | **Date:** Feb 15, 2026

This is the most significant ANE research artifact found. It provides a pure-Rust, MPSGraph-inspired symbolic graph API that compiles directly to ANE hardware via Apple's private `_ANEInMemoryModel` API, completely bypassing CoreML.

### Architecture

```
Rust Graph API -> MIL text (string building) -> _ANEInMemoryModel -> ANE hardware
                                                       |
                                              IOSurface zero-copy I/O
```

### Key Characteristics

- **No Python dependency** -- MIL generation is pure Rust string building, no coremltools
- **IOSurface I/O** -- Zero-copy via `TensorData` (fp32 external, fp16 internal conversion)
- **Working GPT-2** -- 124M params, 45.6 tok/s on M4 Max
- **Compile strategy** -- 48 compiles for GPT-2: 12 layers x 2 ops x 2 phases (prefill + decode)
- **Separate prefill/decode executables** per layer
- **KV cache** -- IOSurface-backed, CPU-written, shape `[embedding_dim, 1, max_seq_len]`
- **LayerNorm** -- Uses `add(-mean)` workaround because ANE subtraction is unreliable
- **GELU** -- Full tanh approximation built from primitives

### Why This Matters for asmi

Bokhan's crate proves direct ANE dispatch from Rust is feasible and performant. asmi can adapt this approach for larger models, especially MoE architectures, without the 2-4x CoreML overhead.

---

## 4. ANE Hardware Constraints

### Specification Table

| Constraint | Value | Notes |
|-----------|-------|-------|
| Compile budget | ~119 per process | Hard limit, leaks -- not per-ANE, per-process |
| On-chip SRAM | ~32 MB | Per ANE die |
| Peak compute | 19 TFLOPS FP16 (M4) | ~9 TFLOPS per M3 Max die in Ultra |
| Power draw | 2-3W under load | Hard power gating: 0mW when idle |
| Energy efficiency | 6.6 TFLOPS/W | Exceptional compared to GPU |
| Queue depth | 127 in-flight | Requests to ANE hardware |
| Conv vs matmul | Conv 3x faster | ANE prefers convolution formulation |
| Graph depth sweet spot | 32+ ops | 94% utilization at this depth |
| Single-op overhead | 0.095ms dispatch | Only 30% utilization for single ops |
| Min spatial width | 64 | Hardware constraint (Bokhan discovery) |
| Subtraction | Unreliable | Use `add(x, mul(mean, -1))` instead |
| Weight baking | At compile time | Weights compiled in, cannot swap at runtime |
| M5 ANE | Same H16 family | Unchanged from M4 |

### Compile Budget Implications

The ~119 compile limit is the most critical constraint. Each compiled graph segment consumes one slot, and these slots leak (never reclaimed until process restart). This means:

- **Dense 70B model:** Would need ~840 compiles (too many) with naive 1-layer-per-compile
- **4-layer fused chunks:** 70B / 4 layers per chunk = ~20 prefill + 20 decode = 40 compiles (fits)
- **MoE 35B-A3B:** Only active experts compiled, ~24 compiles for 3B active (easily fits)

---

## 5. Dual ANE on M3 Ultra

### ioreg Findings (verified on m3u1)

The M3 Ultra contains two separate ANE hardware units, confirmed via `ioreg`:

```
ane0@C9400000 -> RTBuddy(ANE)  -> H11ANE
ane1@C9400000 -> RTBuddy(ANE1) -> H11ANE1
```

### Key Details

| Property | Value |
|----------|-------|
| Chip ID | T6031 (AppleT6031ANEHAL) |
| ANE count | ANEDevicePropertyNumANEs = 2 |
| DMA | Each die has own DART (DMA Address Remapping Table) |
| Load balancing | H1xANELoadBalancer auto-distributes work |
| Direct targeting | H1xANELoadBalancerDirectPathClient -- possible per-die targeting |

### Pipeline Parallelism Opportunity

With two ANE dies per Ultra, pipeline parallelism is possible within a single node:
- ane0 runs layers 0-N/2 (prefill phase)
- ane1 runs layers N/2-N (decode phase)
- Or: ane0 handles draft model, ane1 handles verification (speculative decoding)

The `DirectPathClient` class suggests Apple has internal support for targeting specific dies, though the API is undocumented. Probing this is a priority research item.

---

## 6. Speculative Decoding Strategies

Four proven approaches for accelerating inference on Apple Silicon using ANE:

### 6.1 Mirror Speculative Decoding (Apple Research)

- **Speedup:** 2.8-5.8x over autoregressive
- **Method:** GPU and NPU run in parallel -- NPU generates draft tokens, GPU verifies
- **Key insight:** NPU and GPU share unified memory, so KV cache transfer is free

### 6.2 QuantSpec (Apple Research)

- **Speedup:** ~2.5x
- **Method:** Self-speculative with 4-bit quantized draft of the same model
- **Acceptance rate:** >90% (same architecture, similar representations)
- **Advantage:** No separate draft model needed

### 6.3 SqueezeBits Yetter

- **Method:** ANE handles prefill, GPU handles decode
- **Status:** Production-proven on iPhone
- **Key insight:** Unified memory enables zero-copy KV cache handoff between ANE and GPU

### 6.4 Apple On-Device Setup

- **Configuration:** 3B base model + 300M draft model (visible in ioreg cryptexes)
- **Observation:** Apple ships this configuration for Siri/system intelligence
- **Relevance:** Validates the small-draft-model approach at production scale

---

## 7. MoE Models as Ideal ANE Targets

Mixture of Experts models are uniquely suited to ANE's constraints:

### Why MoE Works

1. **Compile budget savings** -- Only active experts need compilation; inactive experts consume zero compile slots
2. **Small active parameter count** -- A 35B-total model with 3B active fits comfortably in ANE SRAM and compile budget
3. **Router on CPU** -- The tiny gate/router network runs on CPU, dispatching only the selected expert(s) to ANE
4. **Natural chunking** -- Each expert is already an isolated compute graph

### Compile Budget Math

| Model | Architecture | Active Params | Estimated Compiles | Fits Budget? |
|-------|-------------|---------------|-------------------|-------------|
| Qwen 3.5-35B-A3B | MoE | 3B | ~24 | Yes |
| Qwen 3.5-122B-A10B | MoE | 10B | ~48 | Yes |
| Qwen 3.5-397B-A17B | MoE | 17B | ~80 | Tight but yes |
| Llama 70B (dense) | Dense | 70B | ~40 (4-layer chunks) | Yes with chunking |
| Llama 405B (dense) | Dense | 405B | ~100+ | Marginal |

---

## 8. March 2026 Model Landscape

### Qwen 3.5 Family (Released Feb-Mar 2026)

The Qwen 3.5 release is particularly relevant because of its MoE variants:

| Model | Total Params | Active Params | 4-bit Size | Cluster Fit |
|-------|-------------|---------------|------------|-------------|
| Qwen 3.5-0.8B | 0.8B | 0.8B (dense) | ~0.5 GB | Any node |
| Qwen 3.5-35B-A3B | 35B | 3B | ~17 GB | Any single node |
| Qwen 3.5-122B-A10B | 122B | 10B | ~61 GB | Any single node |
| Qwen 3.5-397B-A17B | 397B | 17B | ~200 GB | 1 Ultra (4-bit) |

### Other Notable Models

| Model | Active Params | 4-bit Size | Cluster Fit |
|-------|--------------|------------|-------------|
| GLM-5 | 44B (of 744B) | ~370 GB | Full cluster |
| Llama 70B | 70B (dense) | ~35 GB | 1 Ultra |
| Llama 405B | 405B (dense) | ~200 GB | 3+ nodes via RDMA |

### M5 Silicon Direction

- **ANE unchanged** -- Same H16 family as M4, no architectural improvements
- **New: Neural Accelerators in GPU cores** -- Metal 4 exposes Tensor APIs in each GPU core
- **Apple's public strategy:** Focus on GPU for AI workloads
- **asmi's differentiation:** Private-API ANE access becomes more unique as Apple deprioritizes public ANE tooling

---

## 9. Partitioning Strategy

### Optimal Chunking: 4 Layers per Fused Chunk

Based on compile budget analysis and utilization measurements:

| Chunk Size | Ops per Chunk | Utilization | Compiles for 80-layer Model |
|------------|--------------|-------------|----------------------------|
| 1 layer | ~8 ops | ~60% | 160 (over budget) |
| 2 layers | ~16 ops | ~80% | 80 |
| 4 layers | ~32 ops | ~94% | 40 |
| 8 layers | ~64 ops | ~97% | 20 |

4-layer chunks hit the sweet spot: 94% utilization while keeping compile count well within the ~119 budget, even accounting for separate prefill and decode executables.

### Three-Tier Execution Model

```
Tier 1: Compile-time (startup)
  - Bake weights into ANE programs
  - Pre-compile all layer chunks (prefill + decode variants)
  - Cost: 10-30 seconds, amortized across session

Tier 2: Prefill (per-prompt)
  - Process full input sequence
  - Longer sequences, higher throughput
  - Can use wider spatial dimensions

Tier 3: Decode (per-token)
  - Autoregressive single-token generation
  - Latency-critical path
  - Narrower, faster compiled programs
```

### KV Cache: The Real Bottleneck

The performance bottleneck is not ANE compute but IOSurface I/O for KV cache:
- Each layer reads/writes KV cache via IOSurface lock/memcpy/unlock
- For long sequences, this dominates latency
- **Mitigations:** Sliding window attention, stateful models (cache inside compiled graph), grouped-query attention

### ANEMLL RMSNorm Hack

ANE has native LayerNorm but not RMSNorm. The ANEMLL project discovered a workaround:

```
RMSNorm(x) = LayerNorm(concat([x, -x]))
```

By concatenating `[x, -x]`, the mean becomes zero, making LayerNorm equivalent to RMSNorm. This avoids implementing RMSNorm from primitives on hardware that lacks a square root op.

---

## 10. Implementation Status

### What Exists

| Component | Status | Location |
|-----------|--------|----------|
| IOReport ANE power monitoring | Planned (implementation doc ready) | `docs/plans/2026-03-02-ane-monitoring-and-compute.md` |
| ANE compute bridge (ObjC FFI) | Planned (implementation doc ready) | Same plan, Path B |
| Bokhan crate (reference) | Cloned, builds, runs | `/tmp/ane-bokhan` on m4m-b-1 |
| `GET /ane` endpoint | Planned | Plan Task A3 |
| `POST /ane/eval` endpoint | Scaffolded in plan | Plan Task B3 |
| Feature flag (`--features ane`) | Planned | Plan Task B1 |
| Cluster topology | Implemented | `src/topology.rs` |

### asmi ANE Endpoints (Planned)

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/ane` | GET | ANE power, power-gated status (production) |
| `/ane/status` | GET | ANE compute subsystem availability (experimental) |
| `/ane/eval` | POST | Submit MIL program to ANE (experimental, future) |

---

## 11. ANE Training (maderix/ANE Discovery)

**Repository:** [github.com/maderix/ANE](https://github.com/maderix/ANE) | **Language:** Objective-C | **License:** MIT
**Local clone:** `/tmp/ane-analysis/` on m4m-b-1

This is the most significant ANE research finding after Bokhan's crate. maderix achieved **backpropagation on ANE** -- forward and backward passes running on the Neural Engine, with weight gradient accumulation on CPU. No CoreML training APIs, no Metal, no GPU -- pure ANE compute.

### Training Results

**Model:** Stories110M (12 layers, dim=768, hidden=2048, 12 heads, 32K vocab)
**Hardware:** M4 (single ANE die)

| Optimization | ms/step | ANE Utilization | TFLOPS |
|---|---|---|---|
| Baseline (vDSP transpose) | 33.5 | 3.1% | 0.49 |
| Channel-first layout | 20.3 | 5.2% | 0.82 |
| vDSP vectorized RMSNorm | 14.2 | 7.4% | 1.17 |
| GCD async cblas overlap | 11.4 | 9.2% | 1.46 |
| ANE RMSNorm fusion | 11.4 | 9.2% | 1.46 |
| Wo^T fusion (7→6 kernels) | 11.4 | 9.2% | 1.46 |
| Deferred cblas wait | **9.3** | **11.2%** | **1.78** |

### 6-Kernel Training Architecture

Each transformer layer uses 6 ANE kernels per training step:

| Kernel | Function | Weights Baked |
|---|---|---|
| `kFwdAttn` | RMSNorm + QKV projection + SDPA + output projection | Wq, Wk, Wv, Wo, rms1, causal mask |
| `kFwdFFN` | RMSNorm + SwiGLU FFN (W1, W3, SiLU, W2) | W1, W2, W3, rms2 |
| `kFFNBwd` | FFN backward (W2^T + SiLU_bwd + W1^T + W3^T) | W2^T, W1^T, W3^T |
| `kSdpaBwd1` | Wo^T + SDPA backward part 1 (dV, probs, dp) | Wo^T, causal mask |
| `kSdpaBwd2` | SDPA backward part 2 (softmax grad, dQ, dK) | (weight-free) |
| `kQKVb` | QKV backward (Wq^T + Wk^T + Wv^T → dx) | Wq^T, Wk^T, Wv^T |

**Division of labor:**
- **ANE:** All linear projections (forward + backward dx), SDPA, fused RMSNorm
- **CPU:** RMSNorm backward, residual connections, loss, dW gradient accumulation (cblas_sgemm), Adam optimizer, RoPE, SiLU activation

### Key Optimizations

1. **Channel-first `[1,C,1,S]` layout** -- Matches ANE IOSurface format directly, eliminates all transpose overhead (33.5ms → 20.3ms)
2. **Forward taps** -- Q, K, V, attention scores, hidden states exposed via concat outputs, avoiding CPU recompute in backward pass
3. **GCD async cblas overlap** -- dW gradient sgemms run on serial dispatch queue in parallel with ANE evals
4. **Deferred cblas wait** -- Wait pushed into next step's forward pass for maximum overlap
5. **ANE RMSNorm fusion** -- RMSNorm folded into forward kernels as MIL ops (reduce_sum + pow + mul)
6. **Wo^T fusion** -- Output projection backward merged into SDPA backward kernel (7→6 kernels)

### Compile Budget Management (exec() restart)

The compile budget (~119 per process) leaks even with the private API. maderix's solution:

```
if (compile_count + TOTAL_WEIGHT_KERNELS > MAX_COMPILES) {
    save_checkpoint()    // All weights, Adam states, step counter
    execl(argv[0])       // Fork new process, budget resets
    // New process resumes from checkpoint
}
```

**12 layers × 5 weight-bearing kernels = 60 compiles per weight update.** With the weight-free sdpaBwd2 shared across layers (12 more), total ~72 per batch. Budget allows ~1-2 weight updates before restart.

**Checkpoint format:** `BLZT` magic, version 2, ~440MB (110M params × 4 bytes + Adam m/v states).

### SDPA Causal Masking Workaround

**Critical finding:** ANE hardware **ignores `attn_mask`** in SDPA ops. The mask parameter is accepted syntactically but has no effect on output. Workaround:

```
Decompose SDPA into:
  1. Q @ K^T  (matmul on ANE)
  2. + mask + softmax  (add + softmax on ANE, mask baked as weight)
  3. scores @ V  (matmul on ANE)
```

### Training vs Inference: Weights as Inputs

For inference (Bokhan), weights are baked into compiled programs via BLOBFILE constants (conv ops). For training, maderix switches to **matmul ops** where weights are passed as **input tensors**:

- **Inference:** `conv(weight = BLOBFILE("weights.bin"), x = input)` -- weight is immutable constant
- **Training:** `matmul(x = weight_tensor, y = input_tensor)` -- weight is mutable input via IOSurface

This enables gradient computation: the same kernel structure can compute both forward pass output and backward pass input gradients (dx) by feeding transposed weights (W^T).

---

## 12. SRAM Probing Results

Using `/tmp/ane-analysis/sram_probe.m` and `sram_bench.m`, maderix probed ANE's on-chip SRAM limits:

### Working Memory Formula

```
Total working set = weights + input activations + output activations
                  = (ch × ch × 2) + (ch × sp × 2) + (ch × sp × 2) bytes
```

### SRAM Cliff Detection

| Channels | Spatial | Weight (MB) | Total Working Set (MB) | Result |
|---|---|---|---|---|
| 256 | 64 | 0.13 | 0.26 | OK |
| 512 | 64 | 0.5 | 0.77 | OK |
| 1024 | 64 | 2.0 | 2.5 | OK |
| 2048 | 64 | 8.0 | 8.5 | OK |
| 3072 | 64 | 18.0 | 18.8 | OK |
| 4096 | 64 | 32.0 | 33.0 | OK |
| 5120 | 64 | 50.0 | 51.3 | **Cliff** (severe slowdown) |
| 6144 | 64 | 72.0 | 73.5 | **Fail** |
| 8192 | 32 | 128.0 | 129.0 | **Fail** |

**Estimated ANE SRAM:** ~100-120 MB (not the ~32MB often cited). The cliff between 4096ch (33MB working set) and 5120ch (51MB) suggests the actual boundary is in the 40-50MB range for a single kernel, but tiled execution may extend to ~100MB.

### Bandwidth Measurements

- **Peak sustained bandwidth:** 150-250 GB/s (30-40% of nominal 800 GB/s on M3 Ultra)
- **IOSurface lock/unlock overhead** is significant for small batches
- **Compile time >> eval time** (budget management is more critical than dispatch optimization)

---

## 13. ANE Private API Reference

Consolidated from both Bokhan and maderix implementations:

### Core Classes

| Class | Purpose |
|---|---|
| `_ANEInMemoryModelDescriptor` | Creates model descriptor from MIL text + weight blobs |
| `_ANEInMemoryModel` | Compiled ANE kernel wrapper |
| `_ANERequest` | I/O binding (IOSurface inputs/outputs) |
| `_ANEIOSurfaceObject` | IOSurface wrapper for tensor data |
| `_ANEClient` | Shared connection to ANE driver (alternative API path) |
| `_ANEModel` | Model handle for _ANEClient path |

### Compilation Flow

```
1. MIL text (NSData) + weight dict → _ANEInMemoryModelDescriptor.modelWithMILText:weights:optionsPlist:
2. Descriptor → _ANEInMemoryModel.inMemoryModelWithDescriptor:
3. Model → compileWithQoS:options:error:  (QoS 21 = USER_INTERACTIVE)
4. Model → loadWithQoS:options:error:
5. Model → evaluateWithQoS:options:request:error:  (with _ANERequest)
6. Model → unloadWithQoS:error:
```

### Weight Blob Format

```
Offset 0-63:     Global header (magic: 0x01 @ offset 0, 0x02 @ offset 4)
Per chunk:
  Offset +0:     Magic: 0xDEADBEEF
  Offset +4:     Version: 0x01
  Offset +8:     Data size (uint32)
  Offset +16:    Data offset (uint32, absolute)
  Offset +64:    FP16 weight data
Chunk stride:    64 + (out_ch × in_ch × 2)
```

### IOSurface Tensor I/O

```
IOSurfaceCreate({
    kIOSurfaceWidth: total_bytes,     // NOT logical dimensions
    kIOSurfaceHeight: 1,
    kIOSurfaceBytesPerElement: 1,
    kIOSurfaceBytesPerRow: total_bytes,
    kIOSurfaceAllocSize: total_bytes,
    kIOSurfacePixelFormat: 0
})

// Read: IOSurfaceLock(surf, kIOSurfaceLockReadOnly, NULL) → memcpy → IOSurfaceUnlock
// Write: IOSurfaceLock(surf, 0, NULL) → memcpy → IOSurfaceUnlock
```

**Critical:** ANE I/O is byte-level linear, not shaped. Shape semantics exist only in MIL text. The ANE tensor format is `[1, channels, 1, spatial]` (fp16 internal, fp32 external with cast ops).

### MIL Program Requirements

```
program(1.3)
[buildInfo = dict<string, string>({{
    "coremlc-component-MIL", "3510.2.1",
    "coremlc-version", "3505.4.1",
    "coremltools-version", "9.0"
}})]
{
    func main<ios18>(tensor<fp32, [1, ch, 1, sp]> x) {
        // cast fp32→fp16, compute, cast fp16→fp32
    } -> (output);
}
```

Target must be `<ios18>`. Weight references use `BLOBFILE(path = string("@model_path/weights/name.bin"), offset = uint64(N))`.

---

## 14. Next Steps / Open Questions

### High Priority

1. **Implement IOReport ANE power monitoring** -- Sudoless ANE power via `"Energy Model"` channel (plan ready, Task A1-A3)
2. **Adapt Bokhan crate for MoE** -- Extend graph API to support expert routing and selective compilation
3. **Benchmark Qwen 3.5-35B-A3B on ANE** -- 3B active params should fit single ANE die; measure tok/s vs MLX GPU baseline
4. **Port maderix training loop to Rust** -- Adapt 6-kernel architecture for Bokhan's MIL API; would enable fine-tuning on ANE

### Medium Priority

5. **Probe DirectPathClient** -- Determine if `H1xANELoadBalancerDirectPathClient` allows targeting individual ANE dies on M3 Ultra
6. **RDMA pipeline parallelism** -- Cross-node ANE inference: node A runs layers 0-N, ships activations via TB5 RDMA to node B for layers N-2N
7. **KV cache optimization** -- Sliding window attention, stateful compiled models, grouped-query attention to reduce IOSurface overhead
8. **Measure actual SRAM capacity** -- maderix's probe suggests 40-50MB per kernel, not ~32MB as commonly cited. Need finer-grained sweep.

### Research Questions

9. **Can compiled ANE programs be serialized and shared across nodes?** -- Would eliminate redundant compilation on identical hardware
10. **What is the actual DirectPathClient API surface?** -- Needs runtime probing with `class-dump` or Hopper
11. **Does the compile budget leak apply to _ANEInMemoryModel differently than CoreML?** -- Both Bokhan and maderix use the private API; both report the ~119 limit
12. **Can ANEMLL's chunking strategy (Embed, FFN chunks, LM head) be combined with Bokhan's MIL approach?** -- Best of both worlds: ANEMLL's proven model partitioning with Bokhan's lower-overhead dispatch
13. **Can the exec() restart strategy be adapted for long inference sessions?** -- Periodic checkpoint + restart to reclaim compile budget
14. **What is the actual ANE SDPA mask behavior?** -- maderix reports it's ignored; need to verify on M3 Ultra (different silicon)

---

## Key References

| Resource | URL / Path | Notes |
|----------|-----------|-------|
| Bokhan ANE crate | github.com/computer-graphics-tools/ane | Rust, MIT, direct ANE dispatch |
| maderix/ANE | github.com/maderix/ANE | Training on ANE, 6-kernel backprop, ObjC |
| maderix/ANE (local) | `/tmp/ane-analysis/` | Full source with training/, benchmarks, SRAM probes |
| ANEMLL | github.com/ANEMLL/ANEMLL | CoreML-based ANE inference, RMSNorm hack |
| Mirror Speculative Decoding | Apple Research paper | GPU+NPU parallel, 2.8-5.8x |
| QuantSpec | Apple Research paper | Self-speculative, 4-bit draft |
| SqueezeBits Yetter | squeezebits.com | ANE prefill + GPU decode, production |
| asmi ANE plan | `docs/plans/2026-03-02-ane-monitoring-and-compute.md` | Full implementation plan |
| asmi topology | `src/topology.rs` | RDMA mesh discovery |
| asmi harness | `.claude/harness/` | progress.txt + features.json |

---

*Last updated: 2026-03-04 (added maderix training findings, SRAM probing, private API reference, harness setup)*
