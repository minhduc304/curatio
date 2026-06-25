//! Online HNSW dedup — mirror of src/curate/dedup.py. TDD §4.5, §16.
//!
//! Same M / ef / cosine space + single-threaded insertion order as hnswlib so
//! decisions are deterministic and match the Python reference (SC6). OQ4 resolved:
//! `hnsw_rs` (not `instant-distance`, which builds once and can't insert online
//! the way an incremental dedup stream needs).
//!
//! Embed v4 vectors are L2-normalized (OQ1), so cosine distance = 1 - dot and the
//! 0.90 threshold applies directly.

use hnsw_rs::prelude::*; // re-exports the anndists Distance trait + Hnsw

use crate::minhash::MinHasher;

const M: usize = 16;
const MAX_LAYER: usize = 16;
const EF_CONSTRUCTION: usize = 200;
const EF_SEARCH: usize = 64;

/// Cosine distance for already-L2-normalized vectors (OQ1): 1 - dot. Skips the
/// per-eval norm recomputation that hnsw_rs's built-in DistCosine pays, which
/// dominated the Rust hot path. sim = 1 - distance = dot, matching hnswlib's
/// cosine space on normalized input (parity, SC6).
#[derive(Clone)]
struct DistNormDot;

impl Distance<f32> for DistNormDot {
    fn eval(&self, a: &[f32], b: &[f32]) -> f32 {
        1.0 - a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }
}

pub struct OnlineDedup<'a> {
    index: Hnsw<'a, f32, DistNormDot>,
    threshold: f32,
    surface_threshold: f32,
    signatures: Vec<Vec<u64>>, // per-element MinHash, indexed by the inserted data id
    size: usize,
}

impl<'a> OnlineDedup<'a> {
    pub fn new(threshold: f32, surface_threshold: f32, max_elements: usize) -> Self {
        let index = Hnsw::new(M, max_elements.max(1), MAX_LAYER, EF_CONSTRUCTION, DistNormDot {});
        Self { index, threshold, surface_threshold, signatures: Vec::new(), size: 0 }
    }

    /// Return Some(similarity) if near-duplicate (reject); None if kept + added.
    ///
    /// A cosine candidate is only rejected if the matched neighbour also clears the
    /// surface-Jaccard threshold — distinct-but-topically-similar docs are kept
    /// (mirror of src/curate/dedup.py; holds precision as the index grows).
    pub fn check_and_add(&mut self, emb: &[f32], sig: &[u64]) -> Option<f32> {
        if self.size > 0 {
            let neighbours = self.index.search(emb, 1, EF_SEARCH);
            if let Some(nb) = neighbours.first() {
                let sim = 1.0 - nb.distance;
                if sim >= self.threshold
                    && MinHasher::jaccard(sig, &self.signatures[nb.d_id]) >= self.surface_threshold
                {
                    return Some(sim);
                }
            }
        }
        self.index.insert((emb, self.size));
        self.signatures.push(sig.to_vec());
        self.size += 1;
        None
    }
}
