"""Pydantic data model — single source of truth. See TDD §3.

Mirrored as serde structs on the Rust side (rust/src/pipeline.rs).
"""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

Stage = Literal["heuristic", "quality", "dedup", "kept"]
RejectReason = Literal[
    "too_short",
    "too_long",
    "symbol_ratio",
    "short_lines",
    "boilerplate",
    "wrong_language",
    "low_quality",
    "near_duplicate",
]


class RawDoc(BaseModel):
    id: str
    text: str
    source: str = "fineweb-sample"
    meta: dict[str, str] = {}


class StageResult(BaseModel):
    doc_id: str
    decision: Literal["keep", "reject"]
    stage: Stage
    reason: RejectReason | None = None
    quality_score: float | None = None
    dedup_similarity: float | None = None
    embed_dim: int | None = None


class CuratedDoc(BaseModel):
    id: str
    text: str
    quality_score: float
    source: str


class Label(BaseModel):
    """Command A judge output, cached to disk (paid once)."""

    doc_id: str
    edu_score: int  # 1..5
    rationale: str


class QualityModel(BaseModel):
    """Exported, language-neutral classifier — both runtimes load this (FR9)."""

    coef: list[float]  # length == embed_dim
    intercept: float
    embed_dim: int
    threshold: float  # default decision cutoff


class FunnelStats(BaseModel):
    ingested: int
    kept: int
    rejected_by_stage: dict[Stage, int]
    rejected_by_reason: dict[RejectReason, int]
    retention_rate: float
    docs_per_sec: float  # non-API processing throughput
    embed_calls: int
    chat_calls: int
    est_cost_usd: float
    p50_stage_latency_ms: dict[Stage, float]


class CurationConfig(BaseModel):
    embed_dim: Literal[256, 512, 1024, 1536] = 1024
    embed_batch_size: int = 96
    quality_threshold: float = 0.5
    dedup_cosine_threshold: float = 0.90  # calibrated by eval PR sweep: recall 0.92 @ precision 1.0 (was 0.92 → recall 0.855)
    # Content-aware dedup (see eval/stress.py): MinHash routing co-locates near-dups
    # under scale-out; surface verification confirms a cosine candidate is a true
    # near-dup (Jaccard ≥ threshold), not just topically similar.
    minhash_num_perm: int = 128
    minhash_shingle_size: int = 5
    lsh_band_rows: int = 1  # eval/stress.py sweep: rows=1 maximizes co-location (P=8 recall 0.79 vs 0.68 @ rows=2)
    dedup_surface_threshold: float = 0.5
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    hnsw_ef_search: int = 64
    min_chars: int = 200
    max_chars: int = 100_000
    max_symbol_ratio: float = 0.10
    target_language: str = "en"
    replay_rate_per_sec: int = 0  # 0 = as fast as backpressure allows


class BenchResult(BaseModel):
    runtime: Literal["python", "rust"]
    workers: int
    docs: int
    docs_per_sec: float
    p50_latency_us: float
    decisions_match_reference: bool  # parity gate (NFR5/SC6)


class EvalReport(BaseModel):
    config: CurationConfig
    dedup_precision: float
    dedup_recall: float
    classifier_auc: float
    funnel: FunnelStats
    benchmarks: list[BenchResult]
    n_injected_dups: int
    duration_sec: float
    timestamp: datetime
