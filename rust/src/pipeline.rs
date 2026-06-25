//! Stage orchestration + serde mirrors of src/models.py. See TDD §3, §4.8.

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RawDoc {
    pub id: String,
    pub text: String,
    #[serde(default)]
    pub source: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StageResult {
    pub doc_id: String,
    pub decision: String, // "keep" | "reject"
    pub stage: String,
    pub reason: Option<String>,
    pub quality_score: Option<f32>,
    pub dedup_similarity: Option<f32>,
}

// TODO: run a RawDoc through heuristics -> quality -> dedup -> sink (TDD §4.8).
