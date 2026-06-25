//! C16 — offline benchmark path. Reads a pre-embedded Parquet (Embed API excluded),
//! runs heuristics -> quality -> online dedup, and reports per-doc keep/reject
//! decisions + throughput. eval/bench.py diffs the decisions against the Python
//! reference (parity gate, SC6) and compares docs/sec (SC4/SC5/SC7).

use std::fs::File;
use std::path::Path;
use std::time::Instant;

use arrow::array::{Array, BinaryArray, StringArray};
use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
use serde::Serialize;

use crate::dedup::OnlineDedup;
use crate::heuristics;
use crate::minhash::MinHasher;
use crate::quality::QualityModel;

struct DocRow {
    id: String,
    text: String,
    emb: Vec<f32>,
}

#[derive(Serialize)]
struct Decision {
    id: String,
    decision: &'static str, // "keep" | "reject"
    stage: &'static str,    // "heuristic" | "quality" | "dedup" | "kept"
    reason: Option<&'static str>,
}

#[derive(Serialize)]
struct BenchOutput {
    workers: usize,
    docs: usize,
    elapsed_sec: f64,
    docs_per_sec: f64,
    p50_latency_us: f64,
    decisions: Vec<Decision>,
}

/// Read id / text / emb(binary, 1024xf32 LE) rows from the pre-embedded Parquet.
fn read_parquet(path: &Path) -> anyhow::Result<Vec<DocRow>> {
    let file = File::open(path)?;
    let reader = ParquetRecordBatchReaderBuilder::try_new(file)?.build()?;
    let mut rows = Vec::new();
    for batch in reader {
        let batch = batch?;
        let ids = col::<StringArray>(&batch, "id")?;
        let texts = col::<StringArray>(&batch, "text")?;
        let embs = col::<BinaryArray>(&batch, "emb")?;
        for i in 0..batch.num_rows() {
            let bytes = embs.value(i);
            let emb = bytes
                .chunks_exact(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            rows.push(DocRow {
                id: ids.value(i).to_string(),
                text: texts.value(i).to_string(),
                emb,
            });
        }
    }
    Ok(rows)
}

fn col<'a, T: 'static>(
    batch: &'a arrow::record_batch::RecordBatch,
    name: &str,
) -> anyhow::Result<&'a T> {
    batch
        .column_by_name(name)
        .and_then(|c| c.as_any().downcast_ref::<T>())
        .ok_or_else(|| anyhow::anyhow!("missing/mistyped column {name}"))
}

/// Run the curation hot path over one shard, returning decisions + per-doc latency (us).
fn run_shard(
    rows: &[DocRow],
    model: &QualityModel,
    hasher: &MinHasher,
    dedup_threshold: f32,
    surface_threshold: f32,
) -> (Vec<Decision>, Vec<f64>) {
    let mut dedup = OnlineDedup::new(dedup_threshold, surface_threshold, rows.len());
    let mut decisions = Vec::with_capacity(rows.len());
    let mut latencies = Vec::with_capacity(rows.len());

    for row in rows {
        let t = Instant::now();
        let decision = classify(row, model, hasher, &mut dedup);
        latencies.push(t.elapsed().as_secs_f64() * 1e6);
        decisions.push(decision);
    }
    (decisions, latencies)
}

fn classify(
    row: &DocRow,
    model: &QualityModel,
    hasher: &MinHasher,
    dedup: &mut OnlineDedup,
) -> Decision {
    let id = row.id.clone();
    if let Some(reason) = heuristics::check(&row.text) {
        return Decision { id, decision: "reject", stage: "heuristic", reason: Some(reason) };
    }
    let prob = model.predict_proba(&row.emb);
    if prob < model.threshold as f64 {
        return Decision { id, decision: "reject", stage: "quality", reason: Some("low_quality") };
    }
    if dedup.check_and_add(&row.emb, &hasher.signature(&row.text)).is_some() {
        return Decision { id, decision: "reject", stage: "dedup", reason: Some("near_duplicate") };
    }
    Decision { id, decision: "keep", stage: "kept", reason: None }
}

fn p50(mut xs: Vec<f64>) -> f64 {
    if xs.is_empty() {
        return 0.0;
    }
    xs.sort_by(|a, b| a.partial_cmp(b).unwrap());
    xs[xs.len() / 2]
}

pub fn run(
    input: &Path,
    model_path: &Path,
    minhash_path: &Path,
    workers: usize,
    dedup_threshold: f32,
    surface_threshold: f32,
) -> anyhow::Result<()> {
    let rows = read_parquet(input)?;
    let model = QualityModel::from_json(model_path)?;
    let hasher = MinHasher::from_json(minhash_path)?;
    let n = rows.len();
    if let Some(first) = rows.first() {
        anyhow::ensure!(
            first.emb.len() == model.embed_dim,
            "embedding dim {} != model embed_dim {}",
            first.emb.len(),
            model.embed_dim
        );
    }

    let (decisions, latencies, elapsed) = if workers <= 1 {
        let t0 = Instant::now();
        let (d, l) = run_shard(&rows, &model, &hasher, dedup_threshold, surface_threshold);
        (d, l, t0.elapsed().as_secs_f64())
    } else {
        // Shard contiguously across N OS threads; each keeps its own dedup index
        // (partition-local dedup, the FR11 design). Wall clock of the parallel
        // section drives docs/sec.
        let chunk = n.div_ceil(workers);
        let shards: Vec<&[DocRow]> = rows.chunks(chunk).collect();
        let t0 = Instant::now();
        let results: Vec<(Vec<Decision>, Vec<f64>)> = std::thread::scope(|scope| {
            let handles: Vec<_> = shards
                .iter()
                .map(|shard| {
                    scope.spawn(|| run_shard(shard, &model, &hasher, dedup_threshold, surface_threshold))
                })
                .collect();
            handles.into_iter().map(|h| h.join().unwrap()).collect()
        });
        let elapsed = t0.elapsed().as_secs_f64();
        let mut decisions = Vec::with_capacity(n);
        let mut latencies = Vec::with_capacity(n);
        for (d, l) in results {
            decisions.extend(d);
            latencies.extend(l);
        }
        (decisions, latencies, elapsed)
    };

    let out = BenchOutput {
        workers,
        docs: n,
        elapsed_sec: elapsed,
        docs_per_sec: if elapsed > 0.0 { n as f64 / elapsed } else { 0.0 },
        p50_latency_us: p50(latencies),
        decisions,
    };
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
