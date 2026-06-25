//! MinHash over word-shingles — mirror of src/curate/minhash.py (SC6).
//!
//! Permutation params (a, b) are loaded from the shared models/minhash.json that
//! Python exports, so both runtimes hash identically without reproducing NumPy's
//! RNG. crc32fast computes the same CRC-32/ISO-HDLC as Python's zlib.crc32, and the
//! `(a*h + b) % prime` arithmetic stays within u64 (a < 2^31, h < 2^32 ⇒ < 2^63).

use std::collections::HashSet;
use std::path::Path;

use serde::Deserialize;

const PRIME: u64 = 4294967291; // largest prime < 2^32
const MAXHASH: u64 = 4294967295; // 2^32 - 1, empty-doc sentinel

#[derive(Deserialize)]
pub struct MinHasher {
    num_perm: usize,
    shingle_size: usize,
    a: Vec<u64>,
    b: Vec<u64>,
}

impl MinHasher {
    pub fn from_json(path: &Path) -> anyhow::Result<Self> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    fn shingle_hashes(&self, text: &str) -> HashSet<u64> {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        let k = self.shingle_size;
        let mut set = HashSet::new();
        if words.len() < k {
            if !words.is_empty() {
                set.insert(crc32fast::hash(words.join(" ").as_bytes()) as u64);
            }
        } else {
            for w in words.windows(k) {
                set.insert(crc32fast::hash(w.join(" ").as_bytes()) as u64);
            }
        }
        set
    }

    pub fn signature(&self, text: &str) -> Vec<u64> {
        let shingles = self.shingle_hashes(text);
        if shingles.is_empty() {
            return vec![MAXHASH; self.num_perm];
        }
        let hs: Vec<u64> = shingles.into_iter().collect();
        (0..self.num_perm)
            .map(|i| hs.iter().map(|&h| (self.a[i] * h + self.b[i]) % PRIME).min().unwrap())
            .collect()
    }

    /// Estimated Jaccard = fraction of permutations whose minima agree.
    pub fn jaccard(a: &[u64], b: &[u64]) -> f32 {
        let eq = a.iter().zip(b).filter(|(x, y)| x == y).count();
        eq as f32 / a.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hasher() -> MinHasher {
        let p = Path::new(env!("CARGO_MANIFEST_DIR")).join("../models/minhash.json");
        MinHasher::from_json(&p).expect("minhash.json")
    }

    #[test]
    fn matches_python_reference() {
        // First 6 minima for this exact string, computed by the Python MinHasher
        // (src/curate/minhash.py). Equality proves CRC32 + shingling + arithmetic
        // all match across runtimes — the cross-language parity guarantee.
        let sig = hasher().signature("the quick brown fox jumps over the lazy dog and runs away fast");
        assert_eq!(&sig[..6], &[50383417, 776864844, 1395673139, 77063379, 823443672, 2500240318]);
    }

    #[test]
    fn jaccard_self_is_one() {
        let h = hasher();
        let s = h.signature("mitochondria is the powerhouse of the cell and makes atp energy");
        assert_eq!(MinHasher::jaccard(&s, &s), 1.0);
    }
}
