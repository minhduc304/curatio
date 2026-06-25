"""C11 — FastAPI metrics surface. See TDD §5.

Contract is identical to the Rust/Axum worker so the React dashboard is
backend-agnostic: GET /stats, GET /ws/metrics (WS) + /sse/metrics, GET /health.

Like the Rust worker (one process consumes Kafka AND serves metrics), the curator
loop runs in a background thread inside this process, feeding a shared in-memory
MetricsStore. Set Curatio_CONSUMER=0 to serve the API without the consumer (tests,
or when a curator is driven externally).
"""
from __future__ import annotations

import asyncio
import json
import os
import random
import threading
import time
from contextlib import asynccontextmanager
from collections.abc import AsyncIterator

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse

from src.config import settings
from src.metrics.store import MetricsStore
from src.models import RejectReason, StageResult

BROADCAST_INTERVAL_S = 0.5
RESULTS_DIR = settings.data_dir.parent / "eval" / "results"

# Module-level singleton: the curator (or fake-feed) thread writes, handlers read.
metrics = MetricsStore()


def _start_curator(app: FastAPI) -> None:
    """Run the curator loop in a daemon thread, sharing the module MetricsStore.

    WARNING: this embeds Embed v4 — a live producer flood will spend Cohere quota on
    every cache miss. For a quota-free demo use Curatio_FAKE_FEED=1 instead.
    """
    from src.curate.consumer import Curator  # deferred: pulls in Kafka/Cohere clients

    curator = Curator(metrics=metrics)
    app.state.curator = curator
    thread = threading.Thread(target=curator.run, kwargs={"install_signals": False}, daemon=True)
    thread.start()


def _fake_feed(stop: threading.Event) -> None:
    """Quota-free offline demo: replay the real eval funnel (held-out quality scores +
    the real reject mix) as synthetic StageResults at ~30 docs/sec, plus a small set of
    injected junk/duplicate docs so every funnel stage is visibly working. Drives the
    live dashboard without Kafka or any Embed API call."""
    try:
        funnel = json.loads((RESULTS_DIR / "eval_report.json").read_text())["funnel"]
        scores = json.loads((RESULTS_DIR / "plot_data.json").read_text())["quality_scores"]
        h = funnel["rejected_by_stage"].get("heuristic", 13)
        q = funnel["rejected_by_stage"].get("quality", 101)
        k = funnel["kept"]
        pos, neg = scores["pos"] or [0.7], scores["neg"] or [0.3]
    except Exception:
        h, q, k, pos, neg = 13, 101, 86, [0.7], [0.3]

    reasons: list[RejectReason] = ["too_short", "symbol_ratio", "short_lines", "boilerplate"]
    # Base pool = the real eval-funnel proportions (what the committed charts show).
    pool: list[tuple[str, RejectReason | None]] = (
        [("heuristic", None)] * h + [("quality", "low_quality")] * q + [("kept", None)] * k
    )
    # Demo-only: the real FineWeb slice is already cleaned, so the heuristic and dedup
    # stages barely fire. Inject synthetic junk + duplicate docs so every funnel stage
    # is visible in the live demo; the committed eval charts stay un-injected.
    injected: dict[tuple[str, RejectReason], int] = {
        ("heuristic", "too_short"): 9,
        ("heuristic", "symbol_ratio"): 8,
        ("heuristic", "short_lines"): 8,
        ("heuristic", "boilerplate"): 7,
        ("heuristic", "wrong_language"): 6,
        ("dedup", "near_duplicate"): 14,
    }
    for key, n in injected.items():
        pool += [key] * n

    rng = random.Random(42)
    while not stop.is_set():
        stage, reason = rng.choice(pool)
        if stage == "heuristic":
            res = StageResult(doc_id="demo", decision="reject", stage="heuristic",
                              reason=reason or rng.choice(reasons))
            chars = 0  # rejected before the embed call
        elif stage == "dedup":
            res = StageResult(doc_id="demo", decision="reject", stage="dedup",
                              reason="near_duplicate", dedup_similarity=rng.uniform(0.91, 0.99))
            chars = rng.randint(400, 3000)  # passed embed, then matched a prior doc
        elif stage == "quality":
            res = StageResult(doc_id="demo", decision="reject", stage="quality",
                              reason="low_quality", quality_score=rng.choice(neg))
            chars = rng.randint(400, 3000)
        else:
            res = StageResult(doc_id="demo", decision="keep", stage="kept", quality_score=rng.choice(pos))
            chars = rng.randint(400, 3000)
        metrics.record(res, rng.uniform(0.2, 1.5), embedded_chars=chars)
        time.sleep(1 / 30)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    app.state.curator = None
    app.state.stop = threading.Event()
    if os.environ.get("Curatio_FAKE_FEED") == "1":
        threading.Thread(target=_fake_feed, args=(app.state.stop,), daemon=True).start()
    elif os.environ.get("Curatio_CONSUMER", "1") != "0":
        _start_curator(app)
    try:
        yield
    finally:
        app.state.stop.set()
        if app.state.curator is not None:
            app.state.curator._running = False  # loop exits within one poll timeout


app = FastAPI(title="Curatio metrics", lifespan=lifespan)


def _kafka_reachable() -> bool:
    try:
        from confluent_kafka.admin import AdminClient

        md = AdminClient({"bootstrap.servers": settings.kafka_bootstrap}).list_topics(timeout=1.0)
        return md is not None
    except Exception:
        return False


@app.get("/health")
def health() -> dict[str, object]:
    # cohere_reachable is key-presence only — we never spend a live call here (the
    # trial key is quota-capped).
    return {
        "status": "ok",
        "kafka_reachable": _kafka_reachable(),
        "cohere_reachable": bool(settings.cohere_api_key.get_secret_value()),
    }


@app.get("/stats")
def stats() -> dict[str, object]:
    """Current FunnelStats snapshot (+ quality histogram for the dashboard)."""
    return metrics.live_payload()


@app.websocket("/ws/metrics")
async def ws_metrics(ws: WebSocket) -> None:
    await ws.accept()
    try:
        while True:
            await ws.send_json(metrics.live_payload())
            await asyncio.sleep(BROADCAST_INTERVAL_S)
    except WebSocketDisconnect:
        pass


@app.get("/sse/metrics")
async def sse_metrics() -> StreamingResponse:
    async def gen() -> AsyncIterator[str]:
        while True:
            yield f"data: {json.dumps(metrics.live_payload())}\n\n"
            await asyncio.sleep(BROADCAST_INTERVAL_S)

    return StreamingResponse(gen(), media_type="text/event-stream")
