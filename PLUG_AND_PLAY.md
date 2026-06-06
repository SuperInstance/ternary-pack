# PLUG_AND_PLAY — Pack

> Efficient packing/mining of ternary values

## 🚀 Quick Start

Add to your `Cargo.toml`:

```toml
[dependencies]
ternary-pack = { git = "https://github.com/SuperInstance/ternary-pack" }
```

Use in your code:

```rust
use ternary_pack::{PackedTrits, Trit};

let mut packed = PackedTrits::new(1024);
packed.set(0, Trit::Pos);
packed.set(1, Trit::Neg);
```

## 🔗 Integration

This crate is part of the [SuperInstance ternary fleet](https://github.com/SuperInstance). It uses the canonical `Ternary` type from `ternary-types` for cross-crate compatibility.

## 📄 License

MIT
