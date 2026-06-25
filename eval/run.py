"""C12 — curation-quality eval. See TDD §7.

Three measurable claims, all offline + deterministic (no Kafka):
  1. Dedup precision/recall on the injected near-duplicate set (SC1, SC2).
  2. Distilled classifier AUC on a held-out split of the Command A labels (SC3).
  3. Funnel integrity over a raw HF sample (ingested == kept + Σ rejected).

CACHE-ONLY: this harness never calls the Embed API — it reuses embeddings already
in the disk cache (populated by the live curation run + the label/train step) and
silently skips any doc whose vector isn't cached. This keeps the eval reproducible
and, critically, spends zero Cohere quota (the trial key is near its 1000-call/month
cap). Regenerating the cache with a production key would let it run cold.

Writes eval/results/eval_report.json (EvalReport) + plot_data.json (chart inputs).
Exits non-zero if any of SC1-SC3 misses its bar.
"""
from __future__ import annotations

import json
import time
from collections import Counter
from datetime import datetime

import numpy as np
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

from eval import inject
from src.config import settings
from src.curate.dedup import OnlineDedup
from src.curate.heuristics import HeuristicFilter
from src.curate.quality import QualityClassifier
from src.embedding.service import EmbeddingService
from src.models import CurationConfig, EvalReport, FunnelStats, RawDoc

RESULTS_DIR = settings.data_dir.parent / "eval" / "results"
DEDUP_THRESHOLDS = [0.80, 0.84, 0.88, 0.90, 0.92, 0.94, 0.96, 0.98]
FUNNEL_DECISIONS = 200  # docs to fully classify for the funnel
FUNNEL_STREAM_CAP = 300  # bound the HF stream window
EMBED_USD_PER_1M_TOKENS = 0.12  # Embed v4 list price; tokens ≈ chars / 4


def _cached_embeddings(embedder: EmbeddingService, texts: list[str]) -> tuple[np.ndarray, list[int]]:
    """Return (embeddings, kept_idx) using ONLY cache hits — never calls the API."""
    vecs: list[list[float]] = []
    kept: list[int] = []
    for i, text in enumerate(texts):
        cached = embedder.cache.get(embedder._key(text))
        if cached is not None:
            vecs.append(cached)  # type: ignore[arg-type]
            kept.append(i)
    return np.asarray(vecs, dtype=np.float32), kept


# --- 1. Dedup P/R (SC1, SC2) -------------------------------------------------
def _dedup_pr(docs: list[inject.EvalDoc], embeddings: np.ndarray, threshold: float) -> tuple[float, float, int]:
    """Order-independent P/R: a doc is a true dup iff its group was seen earlier."""
    cfg = CurationConfig().model_copy(update={"dedup_cosine_threshold": threshold})
    dedup = OnlineDedup(cfg, max_elements=len(docs) + 1)
    seen: set[str] = set()
    total_dups = all_dropped = dropped_true = 0
    for d, emb in zip(docs, embeddings):
        is_dup = d.group in seen
        sim = dedup.check_and_add(d.doc_id, emb)
        total_dups += is_dup
        if sim is not None:
            all_dropped += 1
            dropped_true += is_dup
        seen.add(d.group)
    precision = dropped_true / all_dropped if all_dropped else 1.0
    recall = dropped_true / total_dups if total_dups else 0.0
    return precision, recall, total_dups


def eval_dedup(embedder: EmbeddingService) -> tuple[float, float, int, list[dict]]:
    docs = inject.build()
    embeddings, kept = _cached_embeddings(embedder, [d.text for d in docs])
    docs = [docs[i] for i in kept]  # keep only docs whose vector is cached
    curve = [
        {"threshold": thr, "precision": p, "recall": r}
        for thr in DEDUP_THRESHOLDS
        for p, r, _ in [_dedup_pr(docs, embeddings, thr)]
    ]
    op = CurationConfig().dedup_cosine_threshold
    precision, recall, n_dups = _dedup_pr(docs, embeddings, op)
    return precision, recall, n_dups, curve


# --- 2. Classifier AUC (SC3) -------------------------------------------------
def eval_classifier(embedder: EmbeddingService) -> tuple[float, list[float], list[float]]:
    with open(settings.cache_dir / "label_cache.json") as f:
        records = json.load(f)
    texts, y = [], []
    for r in records:
        s = r["edu_score"]
        if s >= 4:
            y.append(1)
        elif s <= 2:
            y.append(0)
        else:
            continue
        texts.append(r["text"])
    X, kept = _cached_embeddings(embedder, texts)
    y_arr = np.asarray([y[i] for i in kept])
    Xtr, Xte, ytr, yte = train_test_split(X, y_arr, test_size=0.25, stratify=y_arr, random_state=42)
    clf = LogisticRegression(max_iter=1000, class_weight="balanced").fit(Xtr, ytr)
    probs = clf.predict_proba(Xte)[:, 1]
    auc = float(roc_auc_score(yte, probs))
    return auc, probs[yte == 1].tolist(), probs[yte == 0].tolist()


# --- 3. Funnel integrity -----------------------------------------------------
def eval_funnel(embedder: EmbeddingService) -> FunnelStats:
    config = CurationConfig()
    heuristics = HeuristicFilter(config)
    quality = (
        QualityClassifier.from_json(str(settings.model_path)) if settings.model_path.exists() else None
    )
    dedup = OnlineDedup(config, max_elements=FUNNEL_DECISIONS + 1)

    by_stage: Counter[str] = Counter()
    by_reason: Counter[str] = Counter()
    lat: dict[str, list[float]] = {"heuristic": [], "quality": [], "dedup": []}
    ingested = kept = embed_calls = embed_chars = streamed = 0

    ds = load_dataset(settings.hf_slice, split="train", streaming=True)
    for row in ds:
        if ingested >= FUNNEL_DECISIONS or streamed >= FUNNEL_STREAM_CAP:
            break
        streamed += 1
        text = row.get("text") or ""
        if not text:
            continue
        doc = RawDoc(id=str(row.get("id", streamed)), text=text)

        t = time.perf_counter()
        reason = heuristics.check(doc)
        lat["heuristic"].append((time.perf_counter() - t) * 1e3)
        if reason is not None:
            ingested += 1
            by_stage["heuristic"] += 1
            by_reason[reason] += 1
            continue

        cached = embedder.cache.get(embedder._key(doc.text))
        if cached is None:
            continue  # cache-only: skip uncached passers without spending quota
        embedding = np.asarray(cached, dtype=np.float32)
        ingested += 1
        embed_calls += 1
        embed_chars += len(doc.text)

        if quality is not None:
            t = time.perf_counter()
            score = quality.predict_proba(embedding)
            lat["quality"].append((time.perf_counter() - t) * 1e3)
            if score < config.quality_threshold:
                by_stage["quality"] += 1
                by_reason["low_quality"] += 1
                continue

        t = time.perf_counter()
        sim = dedup.check_and_add(doc.id, embedding)
        lat["dedup"].append((time.perf_counter() - t) * 1e3)
        if sim is not None:
            by_stage["dedup"] += 1
            by_reason["near_duplicate"] += 1
            continue
        kept += 1

    assert ingested == kept + sum(by_stage.values()), "funnel counts do not conserve"
    nonapi_s = sum(sum(v) for v in lat.values()) / 1e3
    return FunnelStats(
        ingested=ingested,
        kept=kept,
        rejected_by_stage=dict(by_stage),  # type: ignore[arg-type]
        rejected_by_reason=dict(by_reason),  # type: ignore[arg-type]
        retention_rate=kept / ingested if ingested else 0.0,
        docs_per_sec=ingested / nonapi_s if nonapi_s else 0.0,  # non-API stages only
        embed_calls=embed_calls,
        chat_calls=0,
        est_cost_usd=embed_chars / 4 / 1e6 * EMBED_USD_PER_1M_TOKENS,
        p50_stage_latency_ms={k: float(np.percentile(v, 50)) for k, v in lat.items() if v},  # type: ignore[arg-type]
    )


def main() -> None:
    t0 = time.perf_counter()
    config = CurationConfig()
    embedder = EmbeddingService(config.embed_dim, config.embed_batch_size)

    print("[1/3] dedup P/R on injected near-duplicates ...")
    precision, recall, n_dups, pr_curve = eval_dedup(embedder)
    print("[2/3] classifier AUC on held-out Command A labels ...")
    auc, pos_scores, neg_scores = eval_classifier(embedder)
    print("[3/3] funnel integrity over raw HF sample ...")
    funnel = eval_funnel(embedder)

    report = EvalReport(
        config=config,
        dedup_precision=precision,
        dedup_recall=recall,
        classifier_auc=auc,
        funnel=funnel,
        benchmarks=[],  # filled by eval/bench.py (Thu-Fri)
        n_injected_dups=n_dups,
        duration_sec=time.perf_counter() - t0,
        timestamp=datetime.now(),
    )
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(RESULTS_DIR / "eval_report.json", "w") as f:
        f.write(report.model_dump_json(indent=2))
    with open(RESULTS_DIR / "plot_data.json", "w") as f:
        json.dump(
            {
                "dedup_pr": pr_curve,
                "operating_threshold": config.dedup_cosine_threshold,
                "quality_scores": {"pos": pos_scores, "neg": neg_scores},
                "nonapi_docs_per_sec": funnel.docs_per_sec,
            },
            f,
            indent=2,
        )

    checks = [
        ("SC1 dedup recall", recall, 0.90),
        ("SC2 dedup precision", precision, 0.95),
        ("SC3 classifier AUC", auc, 0.85),
    ]
    print(f"\n{'='*52}\nCuratio eval — {report.timestamp:%Y-%m-%d %H:%M}")
    print(
        f"funnel: {funnel.ingested} ingested -> {funnel.kept} kept "
        f"({funnel.retention_rate:.0%}); rejects {dict(funnel.rejected_by_stage)}"
    )
    print(f"non-API throughput: {funnel.docs_per_sec:,.0f} docs/sec; n_injected_dups={n_dups}")
    print("-" * 52)
    ok = True
    for name, val, bar in checks:
        passed = val >= bar
        ok &= passed
        print(f"  {'PASS' if passed else 'FAIL'}  {name:22s} {val:.3f}  (>= {bar})")
    print("=" * 52)
    print(f"report -> {RESULTS_DIR / 'eval_report.json'}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
