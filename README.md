# ternary-pack

Bit-packing ternary values into `u32` registers for 16× FP32 memory density.

## Why This Exists

A single FP32 weight takes 32 bits. A ternary weight takes 2 bits (−1, 0, +1 → 2 bits each). That's 16 ternary weights per `u32` register — a 16× density improvement. On GPU, this means you can load 16× more weights per memory transaction, and matmul becomes XNOR + popcount instead of FMA. This crate implements the packing/unpacking, a GPU register abstraction (`GpuTritReg`), ternary matmul, and Z₃ algebra verification.

## Architecture

### Core Types

- **`Trit`** — Enum: `MinusOne`, `Zero`, `One`. The fundamental ternary digit.
- **`PackedTrits`** — A packed array of trits stored in `Vec<u32>`. Each `u32` holds 16 trits.
- **`GpuTritReg`** — A single `u32` register holding exactly 16 trits, with element-wise operations.
- **`Z3Verifier`** — Static methods verifying Z₃ ring axioms (closure, associativity, distributivity).

### Key Operations

- **pack/unpack**: Convert between `Vec<Trit>` and packed `u32` representation.
- **ternary_matmul**: Matrix multiplication using ternary addition (Z₃ arithmetic).
- **xnor_popcount**: Binary similarity metric between two registers — the GPU primitive.

## Usage

```rust
use ternary_pack::{Trit, PackedTrits, GpuTritReg};

// Pack ternary weights
let weights: Vec<Trit> = vec![Trit::One, Trit::Zero, Trit::MinusOne, /* ... 13 more */];
let packed = PackedTrits::pack(&weights);
println!("Storage: {} bytes vs {} bytes FP32", packed.storage_bytes(), weights.len() * 4);
println!("Density: {:.1}× vs FP32", packed.density_vs_f32());

// GPU register operations
let reg_a = GpuTritReg::new([Trit::One; 16]);
let reg_b = GpuTritReg::new([Trit::MinusOne; 16]);
let similarity = reg_a.xnor_popcount(&reg_b);

// Verify Z₃ algebra
assert!(Z3Verifier::verify_tadd_closure());
assert!(Z3Verifier::verify_tmul_closure());
assert!(Z3Verifier::verify_distributivity());
```

### Ternary Matmul

```rust
use ternary_pack::{Trit, ternary_matmul};

let a: Vec<Trit> = /* 2×3 matrix */;
let b: Vec<Trit> = /* 3×2 matrix */;
let result = ternary_matmul(&a, &b, 2, 3, 2); // rows_a, cols_a, cols_b
```

## API Reference

| Method | Returns | Description |
|--------|---------|-------------|
| `Trit::from_i8(v)` | `Option<Trit>` | Convert i8 to Trit |
| `Trit::to_i8(self)` | `i8` | Trit to i8 (-1, 0, 1) |
| `PackedTrits::pack(values)` | `PackedTrits` | Pack trits into u32 array |
| `packed.unpack()` | `Vec<Trit>` | Unpack back to trits |
| `packed.len()` | `usize` | Number of trits |
| `packed.storage_bytes()` | `usize` | Actual memory used |
| `packed.density_vs_f32()` | `f64` | Density ratio (16.0 ideal) |
| `GpuTritReg::new(values)` | `GpuTritReg` | Create 16-trit register |
| `reg.get(idx)` | `Trit` | Read one trit |
| `reg.xnor_popcount(other)` | `i32` | Binary similarity metric |
| `Z3Verifier::verify_*` | `bool` | Algebra axiom checks |

## The Deeper Idea

The 2-bit encoding (`−1→00, 0→01, +1→10`) is chosen so that XNOR of two packed registers followed by popcount gives a **ternary dot product approximation** in a single hardware instruction on most GPUs. This is the same insight behind XNOR-Net and BinaryConnect, but extended to three values. The unused `11` pattern can serve as a "don't care" marker for sparse or masked operations.

## Related Crates

- **ternary-compress** — higher-level compression (RLE, sparse, dictionary) for ternary data
- **ternary-inference-sim** — simulated inference using packed ternary weights
- **ternary-hotswap-inference** — live model swapping with ternary tensors
