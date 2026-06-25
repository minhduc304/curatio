"""C6 — online HNSW near-duplicate detection. See TDD §4.5.

Index grows as the stream flows; each doc is compared against everything kept
before it. Parity (NFR5): fixed M/ef + single-threaded insertion order is
deterministic and matches the Rust impl.

Embed v4 returns L2-normalized vectors (verified OQ1), so cosine space needs no
extra normalization and the 0.92 threshold applies directly.
"""
from __future__ import annotations

import hnswlib
import numpy as np

from src.curate.minhash import MinHasher
from src.models import CurationConfig


class OnlineDedup:
    def __init__(self, config: CurationConfig, max_elements: int) -> None:
        self.config = config
        self.index = hnswlib.Index(space="cosine", dim=config.embed_dim)
        self.index.init_index(
            max_elements=max_elements,
            ef_construction=config.hnsw_ef_construction,
            M=config.hnsw_m,
        )
        self.index.set_ef(config.hnsw_ef_search)
        self.size = 0
        # Per-element MinHash signature, indexed by the internal label we assign
        # (== self.size at insert). None when a doc is added without a signature.
        self.signatures: list[np.ndarray | None] = []

    def check_and_add(
        self, doc_id: str, embedding: np.ndarray, signature: np.ndarray | None = None
    ) -> float | None:
        """Return similarity if near-duplicate (reject); None if kept + added.

        When `signature` is given, a cosine candidate is only rejected if the matched
        neighbour also clears the surface-Jaccard threshold — distinct-but-topically-
        similar docs (the false positives that emerge as the index grows) are kept.
        With signature=None this is the original cosine-only decision (back-compat).
        """
        if self.size > 0:
            labels, distances = self.index.knn_query(embedding, k=1)
            sim = 1.0 - float(distances[0][0])
            if sim >= self.config.dedup_cosine_threshold:
                neighbor_sig = self.signatures[int(labels[0][0])] if self.signatures else None
                if signature is None or neighbor_sig is None:
                    return sim  # cosine-only decision
                if MinHasher.jaccard(signature, neighbor_sig) >= self.config.dedup_surface_threshold:
                    return sim  # confirmed near-duplicate
                # else: embedding-similar but surface-distinct → false positive, keep
        self.index.add_items(embedding, self.size)
        self.signatures.append(signature)  # aligns with the internal label == self.size
        self.size += 1
        return None
