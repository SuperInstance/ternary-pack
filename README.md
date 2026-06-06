# ternary-pack

Experimental bit-packing of ternary {-1,0,+1} values for GPU memory efficiency. Tests 2-bit packing, XNOR+popcount matmul, and SIMD lane mapping.

## Why This Matters

# ternary-pack
Experimental bit-packing of ternary {-1, 0, +1} values for GPU memory.
Each ternary value maps to 2 bits: -1→00, 0→01, +1→10 (11 unused).
This gives 16 values per u32 register — 16× density over FP32.

## The Five-Layer Stack

This crate is part of the **Oxide Stack** — a distributed GPU runtime built on five layers:

```
┌─────────────────┐
│  cudaclaw        │  Persistent GPU kernels, warp consensus, SmartCRDT
├─────────────────┤
│  cuda-oxide      │  Flux → MIR → Pliron → NVVM → PTX compiler
├─────────────────┤
│  flux-core       │  Bytecode VM + A2A agent protocol
├─────────────────┤
│  pincher         │  "Vector DB as runtime, LLM as compiler"
├─────────────────┤
│  open-parallel   │  Async runtime (tokio fork)
└─────────────────┘
```

The key insight: **ternary values {-1, 0, +1} map directly to GPU compute**. They pack 16× denser than FP32, enable XNOR+popcount matmul, and conservation laws become compile-time checks.

## Design

Every value in this crate follows **ternary algebra** (Z₃):

| Value | Meaning | GPU Analog |
|-------|---------|------------|
| +1 | Positive / Active / Healthy | Warp vote yes |
| 0 | Neutral / Pending / Balanced | Warp vote abstain |
| -1 | Negative / Failed / Overloaded | Warp vote no |

This isn't arbitrary — ternary is the natural encoding for:
1. **BitNet b1.58** (Microsoft) — ternary LLMs at 60% less power
2. **GPU warp voting** — hardware ballot returns ternary consensus
3. **Conservation laws** — {-1, 0, +1} preserves quantity

## Key Types

```rust
pub enum Trit
pub fn from_i8
pub fn to_i8
pub fn to_bits
pub struct PackedTrits
pub fn new
pub fn pack
pub fn unpack
pub fn len
pub fn is_empty
pub fn storage_bytes
pub fn density_vs_f32
```

## Usage

```toml
[dependencies]
ternary-pack = "0.1.0"
```

```rust
use ternary_pack::*;
// See src/lib.rs tests for complete working examples
```

## Testing

```bash
git clone https://github.com/SuperInstance/ternary-pack.git
cd ternary-pack
cargo test    # 10 tests
```

## Stats

| Metric | Value |
|--------|-------|
| Tests | 10 |
| Lines of Rust | 284 |
| Public API | 24 items |

## License

Apache-2.0
