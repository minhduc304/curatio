"""Stress tests for the two dedup limitations — and the content-aware fix.

The headline dedup numbers (precision 1.00, recall 0.92) are measured on a tiny,
single-index corpus. This script puts real numbers on where they break, then on the
MinHash fix (routing + surface verification, see src/curate/minhash.py):

  1. Precision is N-limited. 1.00 holds only because the corpus is small; as the index
     grows, distinct docs drift within the 0.90 cosine threshold. Surface verification
     keeps the embedding-similar-but-distinct ones.

  2. Scale-out lowers recall. The multi-worker path dedups per partition; keying by
     doc_id scatters a near-dup group across partitions. MinHash-band routing
     co-locates the group so recall holds.

Cache-only (zero API quota). Run: `python -m eval.stress`.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Callable

import numpy as np

from eval import inject
from eval.run import _cached_embeddings
from src.config import settings
from src.curate.dedup import OnlineDedup
from src.curate.minhash import MinHasher
from src.embedding.service import EmbeddingService
from src.models import CurationConfig

CFG = CurationConfig()
THRESHOLD = CFG.dedup_cosine_threshold  # 0.90


def _route_id(doc_id: str, _sig: np.ndarray, p: int) -> int:
    return int(hashlib.md5(doc_id.encode()).hexdigest(), 16) % p


def _route_minhash(_doc_id: str, sig: np.ndarray, p: int) -> int:
    return int(hashlib.md5(sig[: CFG.lsh_band_rows].tobytes()).hexdigest(), 16) % p


def _partitioned(
    docs: list[inject.EvalDoc],
    embs: np.ndarray,
    sigs: list[np.ndarray],
    p: int,
    route: Callable[[str, np.ndarray, int], int],
    verify: bool,
) -> tuple[float, float]:
    """Global recall + precision with p partition-local indexes."""
    idxs = [OnlineDedup(CFG, max_elements=len(docs) + 1) for _ in range(p)]
    seen: set[str] = set()
    total = dropped_true = all_dropped = 0
    for d, emb, sig in zip(docs, embs, sigs):
        part = route(d.doc_id, sig, p)
        is_dup = d.group in seen
        sim = idxs[part].check_and_add(d.doc_id, emb, sig if verify else None)
        total += is_dup
        if sim is not None:
            all_dropped += 1
            dropped_true += is_dup
        seen.add(d.group)
    recall = dropped_true / total if total else 0.0
    precision = dropped_true / all_dropped if all_dropped else 1.0
    return recall, precision


def precision_vs_n(embedder: EmbeddingService, hasher: MinHasher) -> None:
    """Exp 1 — nearest-prior cosine grows with N; surface verification removes the
    distinct-doc collisions it produces."""
    with open(settings.cache_dir / "label_cache.json") as f:
        records = json.load(f)
    texts = [r["text"] for r in records]
    embs, kept = _cached_embeddings(embedder, texts)
    texts = [texts[i] for i in kept]
    embs = embs.astype(np.float32)  # Embed v4 is L2-normalized → cosine == dot
    n = len(embs)
    sims = embs @ embs.T

    print(f"\n[1] Precision is N-limited — {n} distinct organic FineWeb docs")
    print("    nearest-prior cosine as the index grows (collisions = distinct docs >= 0.90):")
    print(f"    {'N':>5} {'p50':>7} {'p95':>7} {'p99':>7} {'max':>7} {'collisions':>11}")
    for cut in [50, 100, 200, 300, n]:
        if cut < 2 or cut > n:
            continue
        nn = np.array([sims[i, :i].max() for i in range(1, cut)])
        print(f"    {cut:>5} {np.percentile(nn, 50):>7.3f} {np.percentile(nn, 95):>7.3f} "
              f"{np.percentile(nn, 99):>7.3f} {nn.max():>7.3f} {int((nn >= THRESHOLD).sum()):>11}")

    # Surface verification on the same distinct corpus: how many 0.90 collisions are
    # dropped (= false positives) with cosine-only vs cosine+verification.
    sigs = [hasher.signature(t) for t in texts]
    fp = {}
    for verify in (False, True):
        idx = OnlineDedup(CFG, max_elements=n + 1)
        flagged = []
        for i, (emb, sig) in enumerate(zip(embs, sigs)):
            if idx.check_and_add(str(i), emb, sig if verify else None) is not None:
                flagged.append(i)
        fp[verify] = flagged
    print(f"    false-positive drops among distinct docs:  cosine-only = {len(fp[False])}"
          f"   cosine+verification = {len(fp[True])}")
    for i in fp[False]:
        j = int(sims[i, :i].argmax())
        print(f"      flagged doc {i}: cosine {sims[i, j]:.3f} to prior doc {j}, "
              f"but surface Jaccard {hasher.jaccard(sigs[i], sigs[j]):.3f} "
              f"→ {'kept (distinct)' if i not in fp[True] else 'still dropped'}")
    print("    -> cosine-only flags distinct-but-similar docs as N grows; verification keeps them.")


def recall_vs_partitions(embedder: EmbeddingService, hasher: MinHasher) -> None:
    """Exp 2 — global recall as partition count P grows: doc_id routing (before) vs
    MinHash-band routing + verification (after)."""
    docs = inject.build()
    embs, kept = _cached_embeddings(embedder, [d.text for d in docs])
    docs = [docs[i] for i in kept]
    sigs = [hasher.signature(d.text) for d in docs]
    n_dups = sum(d.is_variant for d in docs)

    print("\n[2] Scale-out recall — partition-local indexes (key = doc_id vs MinHash band)")
    print(f"    {len(docs)} docs, {n_dups} true near-dups, threshold {THRESHOLD}, "
          f"band_rows={CFG.lsh_band_rows}")
    print(f"    {'partitions':>11} {'recall (doc_id)':>16} {'recall (MinHash)':>17} {'precision':>11}")
    for p in [1, 2, 4, 8]:
        before, _ = _partitioned(docs, embs, sigs, p, _route_id, verify=False)
        after, after_p = _partitioned(docs, embs, sigs, p, _route_minhash, verify=True)
        print(f"    {p:>11} {before:>16.3f} {after:>17.3f} {after_p:>11.3f}")
    print("    -> doc_id routing scatters near-dup groups; MinHash routing co-locates them,")
    print("       so recall holds under scale-out (precision stays 1.0 via verification).")


def main() -> None:
    embedder = EmbeddingService()
    hasher = MinHasher(CFG.minhash_num_perm, CFG.minhash_shingle_size)
    precision_vs_n(embedder, hasher)
    recall_vs_partitions(embedder, hasher)
    print()


if __name__ == "__main__":
    main()
