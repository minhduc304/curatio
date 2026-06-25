"""C2 — Curator: drain raw-docs, run stages, commit offsets. See TDD §4, §4.7.

Stage order: heuristics -> embed -> quality -> dedup (TDD §4.2-§4.5). The quality
stage is active only once models/quality_model.json exists (produced by the paid,
one-time label.judge + label.train run); until then docs skip straight to dedup.

enable.auto.commit=False; the offset is committed only after a doc reaches a
terminal state (rejected or kept) => at-least-once, no silent loss (NFR4).
"""
from __future__ import annotations

import signal
import time

from confluent_kafka import Consumer

from src.config import settings
from src.curate.dedup import OnlineDedup
from src.curate.heuristics import HeuristicFilter
from src.curate.minhash import MinHasher
from src.curate.quality import QualityClassifier
from src.embedding.service import EmbeddingService
from src.metrics.store import MetricsStore
from src.models import CuratedDoc, CurationConfig, RawDoc, StageResult
from src.sink.writer import Sink


class Curator:
    def __init__(self, metrics: MetricsStore | None = None) -> None:
        # metrics is set when the curator runs embedded in the FastAPI app (shared
        # in-process store feeds the dashboard); None when run standalone (prints).
        self.metrics = metrics
        self.config = CurationConfig()
        self.heuristics = HeuristicFilter(self.config)
        self.embedder = EmbeddingService(self.config.embed_dim, self.config.embed_batch_size)
        self.hasher = MinHasher(self.config.minhash_num_perm, self.config.minhash_shingle_size)
        self.dedup = OnlineDedup(self.config, max_elements=settings.sample_size)
        self.sink = Sink()
        self.quality = (
            QualityClassifier.from_json(str(settings.model_path))
            if settings.model_path.exists()
            else None
        )
        self.consumer = Consumer(
            {
                "bootstrap.servers": settings.kafka_bootstrap,
                "group.id": "Curatio-curator",
                "auto.offset.reset": "earliest",
                "enable.auto.commit": False,
            }
        )

    def run(self, install_signals: bool = True) -> None:
        # Override any inherited SIG_IGN (background jobs) so the loop stops cleanly
        # and the sink buffer is flushed — confluent-kafka's C poll won't surface a
        # default KeyboardInterrupt reliably. Skipped when embedded in a worker
        # thread (signals can only be installed from the main thread); the FastAPI
        # lifespan flips self._running instead.
        self._running = True
        if install_signals:
            for sig in (signal.SIGINT, signal.SIGTERM):
                signal.signal(sig, lambda *_: setattr(self, "_running", False))

        self.consumer.subscribe([settings.raw_topic])
        try:
            while self._running:
                msg = self.consumer.poll(1.0)
                if msg is None:
                    continue
                if msg.error():
                    continue  # transient; offset not committed, will retry
                self._handle(RawDoc.model_validate_json(msg.value()))
                self.consumer.commit(msg)
        finally:
            self.sink.close()
            self.consumer.close()

    def _handle(self, doc: RawDoc) -> None:
        t0 = time.perf_counter()
        reason = self.heuristics.check(doc)
        if reason is not None:
            self._reject(StageResult(doc_id=doc.id, decision="reject", stage="heuristic", reason=reason), t0, 0)
            return

        embedding = self.embedder.embed([doc.text])[0]
        chars = len(doc.text)

        quality_score: float | None = None
        if self.quality is not None:
            quality_score = self.quality.predict_proba(embedding)
            if quality_score < self.config.quality_threshold:
                self._reject(
                    StageResult(
                        doc_id=doc.id,
                        decision="reject",
                        stage="quality",
                        reason="low_quality",
                        quality_score=quality_score,
                    ),
                    t0,
                    chars,
                )
                return

        # Surface verification: confirm a cosine candidate is a true near-dup (high
        # MinHash-Jaccard), not just topically similar — holds precision as the index
        # grows (README "Limitations").
        sim = self.dedup.check_and_add(doc.id, embedding, self.hasher.signature(doc.text))
        if sim is not None:
            self._reject(
                StageResult(
                    doc_id=doc.id,
                    decision="reject",
                    stage="dedup",
                    reason="near_duplicate",
                    dedup_similarity=sim,
                ),
                t0,
                chars,
            )
            return

        # quality_score is None until the classifier model is trained; 1.0 = "passed
        # all active filters, quality stage not yet enabled" (TDD §4.4).
        self.sink.write_kept(
            CuratedDoc(
                id=doc.id,
                text=doc.text,
                quality_score=quality_score if quality_score is not None else 1.0,
                source=doc.source,
            )
        )
        self._emit(
            StageResult(
                doc_id=doc.id,
                decision="keep",
                stage="kept",
                quality_score=quality_score,
                embed_dim=len(embedding),
            ),
            t0,
            chars,
        )

    def _reject(self, result: StageResult, t0: float, chars: int) -> None:
        self.sink.write_rejected(result)
        self._emit(result, t0, chars)

    def _emit(self, result: StageResult, t0: float, chars: int) -> None:
        if self.metrics is not None:
            self.metrics.record(result, (time.perf_counter() - t0) * 1e3, embedded_chars=chars)
        else:
            print(result.model_dump_json())


if __name__ == "__main__":
    Curator().run()
