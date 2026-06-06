# ternary-pack

**Experimental 2-bit packing of ternary {−1, 0, +1} values for GPU memory — 16× density over FP32 with Z₃ algebraic verification.**

[![License: Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](./LICENSE)

## Background

Ternary neural networks — where weights are constrained to {−1, 0, +1} — have emerged as a promising approach to reducing memory and compute costs in deep learning. The key insight: if weights are ternary, you don't need 32-bit floats. Each weight fits in 2 bits (4 possible encodings, 3 used), giving 16 values per `u32` register — a 16× density improvement over FP32.

But packing is only half the problem. The other half is *algebra*: ternary arithmetic over ℤ/3ℤ must be closed (results stay in {−1, 0, +1}), associative, and distributive. If these properties break during packing/unpacking, the entire computation is unsound.

`ternary-pack` provides:
- A 2-bit encoding scheme (−1→00, 0→01, +1→10) with pack/unpack operations.
- A `GpuTritReg` type representing 16 trits in one `u32`.
- Ternary matrix multiplication.
- A `Z3Verifier` that exhaustively verifies Z₃ ring axioms over all 9 pairwise combinations.

## How It Works

### 2-Bit Encoding

| Trit | Value | Binary |
|------|-------|--------|
| −1   | `NegOne` | `0b00` |
|  0   | `Zero`   | `0b01` |
| +1   | `PosOne` | `0b10` |

The encoding `0b11` is unused and maps to `Zero` on decode (defensive).

### Packing

```rust
let values: Vec<Trit> = /* ... */;
let packed = PackedTrits::pack(&values);
// 1024 trits → 256 bytes (16× density vs FP32)
assert_eq!(packed.density_vs_f32(), 16.0);
```

`PackedTrits` stores values in a `Vec<u32>` where each `u32` holds 16 trits. The `pack`/`unpack` round-trip is verified to be lossless.

### GPU Register Type

`GpuTritReg` packs 16 trits into a single `u32`:
```rust
let reg = GpuTritReg::new(values);  // [Trit; 16] → u32
let val = reg.get(7);               // Extract trit at index 7
let product = reg.xnor_popcount(&other); // SIMD-friendly dot product
```

### Ternary Matrix Multiplication

`PackedTrits::ternary_matmul(a, b, rows_a, cols_a, cols_b)` computes the standard matrix product using ternary arithmetic. Each element of the result is an `i32` (the unclamped sum of ternary products), suitable for batch normalization downstream.

### Z₃ Verification

`Z3Verifier` provides static methods that exhaustively test all 3×3 = 9 input pairs:

- `verify_tadd_closure()` — Addition is closed in Z₃.
- `verify_tmul_closure()` — Multiplication is closed in Z₃.
- `verify_tadd_associative()` — (a + b) + c = a + (b + c).
- `verify_distributivity()` — a × (b + c) = a×b + a×c.

## Experimental Results

The test suite confirms:
- **Lossless round-trip**: Pack 64 values, unpack them, get exactly the same sequence.
- **16× density**: 1024 trits occupy 256 bytes (`density_vs_f32() = 16.0`).
- **Z₃ completeness**: All four ring axioms pass for all 9³ = 729 combinations (associativity, distributivity).
- **Addition table**: `−1 + −1 = +1`, `+1 + +1 = −1`, `−1 + +1 = 0` (correct Z₃ arithmetic).
- **Multiplication table**: `−1 × −1 = +1`, `+1 × −1 = −1`, `0 × +1 = 0`.
- **Matrix multiply**: 2×2 ternary matrix product gives correct results.

## Impact

Memory is often the bottleneck in neural network inference. By packing ternary weights at 2 bits each, `ternary-pack` enables:
- 16× more weights per GPU register compared to FP32.
- XNOR+popcount kernels for ternary dot products (amenable to SIMD).
- Formal verification that the packing doesn't violate Z₃ algebra.

## Use Cases

1. **Ternary Neural Network Inference** — Pack quantized {−1, 0, +1} weights into `PackedTrits`, ship to GPU, and compute dot products using `xnor_popcount` for fast inference.
2. **Memory-Efficient Embeddings** — Store ternary embeddings (e.g., from ternary BERT) at 2 bits per dimension instead of 32 bits, enabling 16× larger embedding tables in the same VRAM.
3. **FPGA Ternary Accelerators** — The 2-bit encoding maps directly to LUT-based ternary ALUs on FPGAs, where each trit is a 2-bit bus.
4. **Algebraic Testing** — Use `Z3Verifier` as a compile-time or test-time assertion that custom ternary operations respect ring axioms.

## Open Questions

1. **GPU kernel integration** — Can the 2-bit packing be directly consumed by CUDA/HIP kernels without unpacking? XNOR+popcount works for binary; does a ternary analog exist?
2. **Sparse packing** — For mostly-zero ternary vectors, would a run-length encoding combined with 2-bit packing be more efficient than dense packing?
3. **Endianness** — The current packing is little-endian (bit position = index × 2). Should big-endian packing be supported for network-order environments?

## Connection to Oxide Stack

`ternary-pack` is the memory efficiency layer of the ternary fleet. It bridges the abstract Z₃ arithmetic of `ternary-core` with the physical realities of GPU memory layout. Compiled strategies from `ternary-compiler` can be packed for deployment; ternary signals from `ternary-signals` can be stored compactly for archival. In the broader vision, `ternary-pack` enables the "ternary everywhere" model: compute in ternary, store in ternary, communicate in ternary — all at minimal memory cost.
