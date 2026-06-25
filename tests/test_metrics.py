"""MetricsStore aggregation + the FastAPI metrics contract (no Kafka, no Cohere).

The app is started with Curatio_CONSUMER=0 so no curator thread spins up; the module
MetricsStore is fed directly to simulate a run.
"""
from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from src.metrics.store import MetricsStore
from src.models import StageResult


def _kept(doc_id: str, score: float) -> StageResult:
    return StageResult(doc_id=doc_id, decision="keep", stage="kept", quality_score=score)


def _reject(doc_id: str, stage: str, reason: str, score: float | None = None) -> StageResult:
    return StageResult(doc_id=doc_id, decision="reject", stage=stage, reason=reason, quality_score=score)


def test_store_aggregates_funnel() -> None:
    s = MetricsStore()
    s.record(_reject("a", "heuristic", "too_short"), 0.2, embedded_chars=0)
    s.record(_reject("b", "quality", "low_quality", 0.3), 1.1, embedded_chars=500)
    s.record(_kept("c", 0.8), 1.5, embedded_chars=600)
    s.record(_kept("d", 0.6), 1.4, embedded_chars=700)

    snap = s.snapshot()
    assert snap.ingested == 4
    assert snap.kept == 2
    assert snap.rejected_by_stage == {"heuristic": 1, "quality": 1}
    assert snap.rejected_by_reason == {"too_short": 1, "low_quality": 1}
    assert snap.retention_rate == 0.5
    assert snap.embed_calls == 3  # b, c, d reached embed; a did not
    assert snap.est_cost_usd > 0
    assert "kept" in snap.p50_stage_latency_ms


def test_store_quality_histogram() -> None:
    s = MetricsStore()
    s.record(_kept("c", 0.82), 1.0, embedded_chars=100)  # bin 16
    s.record(_reject("b", "quality", "low_quality", 0.05), 1.0, embedded_chars=100)  # bin 1
    hist = {h["bin"]: h["count"] for h in s.live_payload()["quality_hist"]}  # type: ignore[index]
    assert sum(hist.values()) == 2
    assert any(b > 0.8 and hist[b] == 1 for b in hist)  # high-quality doc landed in a high bin


@pytest.fixture()
def client() -> TestClient:
    os.environ["Curatio_CONSUMER"] = "0"  # serve API without the curator thread
    from src.api import app as app_module

    app_module.metrics = MetricsStore()  # fresh store per test
    with TestClient(app_module.app) as c:
        c._store = app_module.metrics  # type: ignore[attr-defined]
        yield c


def test_health(client: TestClient) -> None:
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert set(body) == {"status", "kafka_reachable", "cohere_reachable"}


def test_stats_reflects_store(client: TestClient) -> None:
    client._store.record(_kept("c", 0.8), 1.0, embedded_chars=600)  # type: ignore[attr-defined]
    body = client.get("/stats").json()
    assert body["ingested"] == 1
    assert body["kept"] == 1
    assert "quality_hist" in body


def test_ws_pushes_payload(client: TestClient) -> None:
    client._store.record(_kept("c", 0.8), 1.0, embedded_chars=600)  # type: ignore[attr-defined]
    with client.websocket_connect("/ws/metrics") as ws:
        msg = ws.receive_json()
    assert msg["ingested"] == 1
    assert "retention_rate" in msg
