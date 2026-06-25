//! Quality classifier — loads quality_model.json, does the dot product. TDD §4.4, FR9.
//!
//! prob = sigmoid(coef . emb + intercept); reject if prob < threshold. The dot
//! product accumulates in f64 so the keep/reject decision is independent of
//! summation order across runtimes (parity, SC6). The Python benchmark reference
//! also accumulates in f64 (eval/bench.py).

use std::path::Path;

use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub struct QualityModel {
    pub coef: Vec<f32>,
    pub intercept: f32,
    pub embed_dim: usize,
    pub threshold: f32,
}

impl QualityModel {
    pub fn from_json(path: &Path) -> anyhow::Result<Self> {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    }

    /// prob = sigmoid(coef . emb + intercept).
    pub fn predict_proba(&self, emb: &[f32]) -> f64 {
        let z: f64 = self
            .coef
            .iter()
            .zip(emb)
            .map(|(c, e)| *c as f64 * *e as f64)
            .sum::<f64>()
            + self.intercept as f64;
        1.0 / (1.0 + (-z).exp())
    }
}
