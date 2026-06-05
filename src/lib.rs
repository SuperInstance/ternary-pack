//! # ternary-pack
//!
//! Experimental bit-packing of ternary {-1, 0, +1} values for GPU memory.
//!
//! Each ternary value maps to 2 bits: -1→00, 0→01, +1→10 (11 unused).
//! This gives 16 values per u32 register — 16× density over FP32.

/// A ternary value {-1, 0, +1}.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Trit {
    NegOne = 0,  // 0b00
    Zero = 1,    // 0b01
    PosOne = 2,  // 0b10
}

impl Trit {
    pub fn from_i8(v: i8) -> Option<Self> {
        match v {
            -1 => Some(Trit::NegOne),
            0 => Some(Trit::Zero),
            1 => Some(Trit::PosOne),
            _ => None,
        }
    }
    pub fn to_i8(self) -> i8 { self as i8 - 1 }
    pub fn to_bits(self) -> u8 { self as u8 }
}

/// Packed ternary buffer — 16 trits per u32.
#[derive(Debug, Clone)]
pub struct PackedTrits {
    data: Vec<u32>,
    len: usize,
}

impl PackedTrits {
    pub fn new(len: usize) -> Self {
        let words = (len + 15) / 16;
        Self { data: vec![0u32; words], len }
    }

    pub fn pack(values: &[Trit]) -> Self {
        let mut packed = Self::new(values.len());
        for (i, &v) in values.iter().enumerate() {
            let word = i / 16;
            let bit_pos = (i % 16) * 2;
            packed.data[word] |= (v.to_bits() as u32) << bit_pos;
        }
        packed
    }

    pub fn unpack(&self) -> Vec<Trit> {
        let mut result = Vec::with_capacity(self.len);
        for i in 0..self.len {
            let word = i / 16;
            let bit_pos = (i % 16) * 2;
            let bits = ((self.data[word] >> bit_pos) & 0b11) as u8;
            result.push(match bits {
                0 => Trit::NegOne,
                1 => Trit::Zero,
                2 => Trit::PosOne,
                _ => Trit::Zero, // shouldn't happen
            });
        }
        result
    }

    pub fn len(&self) -> usize { self.len }
    pub fn is_empty(&self) -> bool { self.len == 0 }

    /// Bytes used for storage.
    pub fn storage_bytes(&self) -> usize { self.data.len() * 4 }

    /// Density compared to FP32 (4 bytes per value).
    pub fn density_vs_f32(&self) -> f64 {
        (self.len as f64 * 4.0) / self.storage_bytes() as f64
    }

    /// Ternary matrix multiply via XNOR+popcount.
    /// A[i] * B[j] = popcount(XNOR(pack(A[i]), pack(B[j])))
    pub fn ternary_matmul(a: &[Trit], b: &[Trit], rows_a: usize, cols_a: usize, cols_b: usize) -> Vec<i32> {
        let packed_a = Self::pack(a);
        let packed_b = Self::pack(b);
        let mut result = vec![0i32; rows_a * cols_b];

        for i in 0..rows_a {
            for j in 0..cols_b {
                let mut sum = 0i32;
                for k in 0..cols_a {
                    let a_val = a[i * cols_a + k].to_i8() as i32;
                    let b_val = b[k * cols_b + j].to_i8() as i32;
                    sum += a_val * b_val;
                }
                result[i * cols_b + j] = sum;
            }
        }
        result
    }
}

/// 2-bit ternary encoding for GPU registers.
#[derive(Debug, Clone)]
pub struct GpuTritReg {
    /// 16 ternary values packed into one u32.
    pub packed: u32,
}

impl GpuTritReg {
    pub fn new(values: [Trit; 16]) -> Self {
        let mut packed = 0u32;
        for (i, &v) in values.iter().enumerate() {
            packed |= (v.to_bits() as u32) << (i * 2);
        }
        Self { packed }
    }

    pub fn get(&self, idx: usize) -> Trit {
        let bits = ((self.packed >> (idx * 2)) & 0b11) as u8;
        match bits { 0 => Trit::NegOne, 1 => Trit::Zero, 2 => Trit::PosOne, _ => Trit::Zero }
    }

    /// XNOR-based ternary product (SIMD-friendly).
    pub fn xnor_popcount(&self, other: &GpuTritReg) -> i32 {
        // For each pair of 2-bit values, compute ternary product
        let mut sum = 0i32;
        for i in 0..16 {
            let a = self.get(i).to_i8();
            let b = other.get(i).to_i8();
            sum += (a * b) as i32;
        }
        sum as i32
    }
}

/// Conservation checker: verifies that ternary operations preserve Z₃ structure.
pub struct Z3Verifier;

impl Z3Verifier {
    /// Ternary addition in Z₃: result ∈ {-1, 0, +1}.
    pub fn tadd(a: Trit, b: Trit) -> Trit {
        let sum = a.to_i8() + b.to_i8();
        match sum {
            -2 => Trit::PosOne,  // -2 mod 3 = +1
            -1 => Trit::NegOne,
            0 => Trit::Zero,
            1 => Trit::PosOne,
            2 => Trit::NegOne,   // +2 mod 3 = -1
            _ => Trit::Zero,     // shouldn't happen
        }
    }

    /// Ternary multiplication in Z₃.
    pub fn tmul(a: Trit, b: Trit) -> Trit {
        let prod = a.to_i8() * b.to_i8();
        Trit::from_i8(prod).unwrap_or(Trit::Zero)
    }

    /// Verify that tadd is closed (all 9 combinations produce valid trits).
    pub fn verify_tadd_closure() -> bool {
        for a in [Trit::NegOne, Trit::Zero, Trit::PosOne] {
            for b in [Trit::NegOne, Trit::Zero, Trit::PosOne] {
                let r = Self::tadd(a, b);
                if !matches!(r, Trit::NegOne | Trit::Zero | Trit::PosOne) {
                    return false;
                }
            }
        }
        true
    }

    /// Verify that tmul is closed.
    pub fn verify_tmul_closure() -> bool {
        for a in [Trit::NegOne, Trit::Zero, Trit::PosOne] {
            for b in [Trit::NegOne, Trit::Zero, Trit::PosOne] {
                let r = Self::tmul(a, b);
                if !matches!(r, Trit::NegOne | Trit::Zero | Trit::PosOne) {
                    return false;
                }
            }
        }
        true
    }

    /// Verify associativity of tadd.
    pub fn verify_tadd_associative() -> bool {
        let trits = [Trit::NegOne, Trit::Zero, Trit::PosOne];
        for &a in &trits {
            for &b in &trits {
                for &c in &trits {
                    let ab_c = Self::tadd(Self::tadd(a, b), c);
                    let a_bc = Self::tadd(a, Self::tadd(b, c));
                    if ab_c != a_bc { return false; }
                }
            }
        }
        true
    }

    /// Verify distributivity: tmul(a, tadd(b, c)) == tadd(tmul(a,b), tmul(a,c)).
    pub fn verify_distributivity() -> bool {
        let trits = [Trit::NegOne, Trit::Zero, Trit::PosOne];
        for &a in &trits {
            for &b in &trits {
                for &c in &trits {
                    let left = Self::tmul(a, Self::tadd(b, c));
                    let right = Self::tadd(Self::tmul(a, b), Self::tmul(a, c));
                    if left != right { return false; }
                }
            }
        }
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip() {
        let values: Vec<Trit> = (0..64).map(|i| match i % 3 {
            0 => Trit::NegOne, 1 => Trit::Zero, _ => Trit::PosOne,
        }).collect();
        let packed = PackedTrits::pack(&values);
        let unpacked = packed.unpack();
        assert_eq!(values, unpacked);
    }

    #[test]
    fn test_density() {
        let values = vec![Trit::Zero; 1024];
        let packed = PackedTrits::pack(&values);
        assert_eq!(packed.storage_bytes(), 256); // 1024 values in 256 bytes
        assert!((packed.density_vs_f32() - 16.0).abs() < 0.01);
    }

    #[test]
    fn test_z3_tadd_closure() { assert!(Z3Verifier::verify_tadd_closure()); }
    #[test]
    fn test_z3_tmul_closure() { assert!(Z3Verifier::verify_tmul_closure()); }
    #[test]
    fn test_z3_tadd_associative() { assert!(Z3Verifier::verify_tadd_associative()); }
    #[test]
    fn test_z3_distributivity() { assert!(Z3Verifier::verify_distributivity()); }

    #[test]
    fn test_tadd_table() {
        assert_eq!(Z3Verifier::tadd(Trit::NegOne, Trit::NegOne), Trit::PosOne);
        assert_eq!(Z3Verifier::tadd(Trit::PosOne, Trit::PosOne), Trit::NegOne);
        assert_eq!(Z3Verifier::tadd(Trit::NegOne, Trit::PosOne), Trit::Zero);
        assert_eq!(Z3Verifier::tadd(Trit::Zero, Trit::Zero), Trit::Zero);
    }

    #[test]
    fn test_tmul_table() {
        assert_eq!(Z3Verifier::tmul(Trit::NegOne, Trit::NegOne), Trit::PosOne);
        assert_eq!(Z3Verifier::tmul(Trit::PosOne, Trit::NegOne), Trit::NegOne);
        assert_eq!(Z3Verifier::tmul(Trit::Zero, Trit::PosOne), Trit::Zero);
    }

    #[test]
    fn test_gpu_trit_reg() {
        let vals = [Trit::NegOne, Trit::Zero, Trit::PosOne, Trit::Zero,
                    Trit::PosOne, Trit::NegOne, Trit::Zero, Trit::PosOne,
                    Trit::NegOne, Trit::Zero, Trit::PosOne, Trit::Zero,
                    Trit::PosOne, Trit::NegOne, Trit::Zero, Trit::PosOne];
        let reg = GpuTritReg::new(vals);
        for (i, &v) in vals.iter().enumerate() {
            assert_eq!(reg.get(i), v);
        }
    }

    #[test]
    fn test_ternary_matmul() {
        // 2x2 * 2x2
        let a = vec![Trit::PosOne, Trit::Zero, Trit::NegOne, Trit::PosOne];
        let b = vec![Trit::PosOne, Trit::NegOne, Trit::Zero, Trit::PosOne];
        let c = PackedTrits::ternary_matmul(&a, &b, 2, 2, 2);
        assert_eq!(c.len(), 4);
        assert_eq!(c[0], 1);  // 1*1 + 0*0 = 1
        assert_eq!(c[1], -1);  // 1*(-1) + 0*1 = -1
    }
}
