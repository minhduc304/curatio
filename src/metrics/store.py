"""C10 — counters + percentile latency + score histogram. See TDD §12, §5.

Aggregates per-stage counts, a rolling per-stage latency list, and a quality-score
histogram into a live payload, snapshotted on demand and pushed over the metrics
WebSocket (~500ms). Thread-safe: the curator thread writes via record(); the API
event loop reads via snapshot()/live_payload().
"""
from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict
from statistics import median

from src.models import FunnelStats, StageResult

EMBED_USD_PER_1M_TOKENS = 0.12  # Embed v4 list price; tokens ≈ chars / 4
_QUALITY_BINS = 20  # histogram resolution over [0, 1]


class MetricsStore:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._ingested = 0
        self._kept = 0
        self._by_stage: Counter[str] = Counter()
        self._by_reason: Counter[str] = Counter()
        self._embed_calls = 0
        self._embed_chars = 0
        self._lat: dict[str, list[float]] = defaultdict(list)
        self._quality_hist = [0] * _QUALITY_BINS
        self._t0 = time.perf_counter()

    def record(self, result: StageResult, stage_latency_ms: float, embedded_chars: int = 0) -> None:
        """One call per doc at its terminal stage. embedded_chars > 0 iff the doc
        reached the Embed stage (i.e., passed heuristics) — drives cost + embed_calls."""
        with self._lock:
            self._ingested += 1
            if result.decision == "keep":
                self._kept += 1
            else:
                self._by_stage[result.stage] += 1
                if result.reason:
                    self._by_reason[result.reason] += 1
            if embedded_chars:
                self._embed_calls += 1
                self._embed_chars += embedded_chars
            self._lat[result.stage].append(stage_latency_ms)
            if result.quality_score is not None:
                b = min(int(result.quality_score * _QUALITY_BINS), _QUALITY_BINS - 1)
                self._quality_hist[b] += 1

    def snapshot(self) -> FunnelStats:
        with self._lock:
            return self._funnel_locked()

    def live_payload(self) -> dict[str, object]:
        """FunnelStats + a quality-score histogram (additive keys; a backend that
        omits them just renders an empty histogram — keeps the contract agnostic)."""
        with self._lock:
            funnel = self._funnel_locked()
            hist = [
                {"bin": round((i + 0.5) / _QUALITY_BINS, 3), "count": c}
                for i, c in enumerate(self._quality_hist)
            ]
        payload = funnel.model_dump()
        payload["quality_hist"] = hist
        return payload

    def _funnel_locked(self) -> FunnelStats:
        elapsed = max(time.perf_counter() - self._t0, 1e-9)
        return FunnelStats(
            ingested=self._ingested,
            kept=self._kept,
            rejected_by_stage=dict(self._by_stage),  # type: ignore[arg-type]
            rejected_by_reason=dict(self._by_reason),  # type: ignore[arg-type]
            retention_rate=self._kept / self._ingested if self._ingested else 0.0,
            docs_per_sec=self._ingested / elapsed,
            embed_calls=self._embed_calls,
            chat_calls=0,
            est_cost_usd=self._embed_chars / 4 / 1e6 * EMBED_USD_PER_1M_TOKENS,
            p50_stage_latency_ms={k: median(v) for k, v in self._lat.items() if v},  # type: ignore[misc]
        )
