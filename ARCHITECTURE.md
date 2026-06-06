# Architecture — ternary-pack

> *Internal design, data flow, and extension points.*

## Overview

This crate implements ternary {-1, 0, +1} logic for the `pack` domain.
It is one of ~160 ternary crates in the SuperInstance fleet, all sharing Z₃ arithmetic
from [ternary-core](https://github.com/SuperInstance/ternary-core).

The ternary principle: **0 is not nothing** — it is the "neutral" or "abstain" state,
distinct from both positive and negative. This three-state encoding is more expressive
than binary for systems that need to represent an off-ramp or undecided state.

## Source Structure

1 Rust source file(s) in `src/`:

## Core Types

- **`PackedTrits`** — primary data structure
- **`GpuTritReg`** — primary data structure
- **`Z3Verifier`** — primary data structure

## Key Functions

- `from_i8()`
- `to_i8()`
- `to_bits()`
- `new()`
- `pack()`
- `unpack()`
- `len()`
- `is_empty()`

## Data Flow

```
Input → ternary_pack::transform → Ternary {-1,0,+1} → Output
```

## Design Principles

1. **Zero-dependency where possible** — keep the trust chain minimal
2. **Ternary by default** — all operations expose or consume {-1, 0, +1}
3. **No hidden state** — pure functions over explicit parameters
4. **Fail closed** — errors return safe defaults (typically 0/neutral)

## Ternary Mapping

| Value | Meaning |
|-------|---------|
| +1 | Active / positive / true / signal on |
| 0  | Neutral / undecided / empty / passive |
| -1 | Inactive / negative / false / signal off |

## Cross-Repo References

- [ternary-core](https://github.com/SuperInstance/ternary-core) — shared traits
- [ternary-types](https://github.com/SuperInstance/ternary-types) — type-level encodings
