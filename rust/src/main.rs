//! C15 — Rust/Axum hot-path curation worker. See TDD §4.8.
//!
//! Two modes:
//!   `Curatio-worker bench --input <parquet> [--model <json>] [--workers N] [--dedup-threshold T]`
//!       offline benchmark over pre-embedded docs (Embed API excluded); prints a
//!       decisions+throughput JSON for eval/bench.py (parity gate + SC4/SC5/SC7).
//!   `Curatio-worker --workers N`
//!       live worker: consume `raw-docs`, run the pipeline, serve Axum metrics.
//!       (Wired in the dashboard phase; TDD §4.8, §5.)

mod bench;
mod dedup;
mod heuristics;
mod metrics;
mod minhash;
mod pipeline;
mod quality;
mod sink;

use std::path::PathBuf;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    dotenvy::dotenv().ok();
    // Logs to stderr at WARN so bench mode's JSON owns stdout cleanly (hnsw_rs is
    // chatty at INFO).
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr)
        .with_max_level(tracing::Level::WARN)
        .init();

    let args: Vec<String> = std::env::args().collect();
    if args.get(1).map(String::as_str) == Some("bench") {
        return run_bench(&args[2..]);
    }

    // TODO: live mode — parse --workers; subscribe to raw-docs; run pipeline per
    //       partition; serve axum metrics on :8000 (TDD §4.8, §5).
    todo!("live worker — dashboard phase (TDD §4.8)")
}

fn run_bench(args: &[String]) -> anyhow::Result<()> {
    let mut input = PathBuf::from("eval/results/bench_input.parquet");
    let mut model = PathBuf::from("models/quality_model.json");
    let mut minhash = PathBuf::from("models/minhash.json");
    let mut workers: usize = 1;
    let mut dedup_threshold: f32 = 0.90;
    let mut surface_threshold: f32 = 0.5;

    let mut i = 0;
    while i < args.len() {
        match args[i].as_str() {
            "--input" => input = PathBuf::from(&args[i + 1]),
            "--model" => model = PathBuf::from(&args[i + 1]),
            "--minhash" => minhash = PathBuf::from(&args[i + 1]),
            "--workers" => workers = args[i + 1].parse()?,
            "--dedup-threshold" => dedup_threshold = args[i + 1].parse()?,
            "--surface-threshold" => surface_threshold = args[i + 1].parse()?,
            other => anyhow::bail!("unknown bench arg: {other}"),
        }
        i += 2;
    }
    bench::run(&input, &model, &minhash, workers, dedup_threshold, surface_threshold)
}
