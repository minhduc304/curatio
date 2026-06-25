"""C8 — offline Command A edu-quality labeling (one-time, cached). See TDD §4.4.

Sample the first N heuristic-passing docs from the HF slice; Command A scores each
1-5 for educational/informational value; cache to .cache/label_cache.json. Batched
(K docs/call) and paced under the 20 req/min trial limit. Resumable: a re-run loads
the cache and labels only the docs that are still missing, so quota is never re-spent.
Never runs in the demo path.
"""
from __future__ import annotations

import argparse
import json
import time

import cohere
from cohere.errors import TooManyRequestsError
from datasets import load_dataset
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings
from src.curate.heuristics import HeuristicFilter
from src.models import CurationConfig, RawDoc

N_LABEL = 600  # heuristic-passing docs to label (user-chosen budget)
BATCH_K = 8  # docs per Command A call
MAX_DOC_CHARS = 2000  # truncate each doc to bound prompt tokens
REQ_PER_MIN = 20  # trial chat rate cap
_MIN_INTERVAL = 60.0 / REQ_PER_MIN + 0.5  # pacing floor between calls, with margin

_SYSTEM = (
    "You are a strict data-quality annotator building a pretraining corpus. Rate each "
    "web document for educational and informational value on an integer 1-5 scale:\n"
    "1 = no value: spam, boilerplate, navigation, link lists, or incoherent text.\n"
    "2 = little value: mostly promotional or shallow, minimal real information.\n"
    "3 = some value: contains useful information but not especially educational.\n"
    "4 = educational: coherent and informative, like a good article, tutorial, or "
    "reference page.\n"
    "5 = highly educational: comprehensive, well-structured, textbook- or "
    "encyclopedia-quality.\n"
    'Return JSON {"scores": [{"id": <int>, "score": <1-5>}, ...]} with one entry per '
    "document id provided. Score every document; output nothing else."
)

_RESPONSE_FORMAT = {
    "type": "json_object",
    "json_schema": {
        "type": "object",
        "properties": {
            "scores": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "score": {"type": "integer"},
                    },
                    "required": ["id", "score"],
                },
            }
        },
        "required": ["scores"],
    },
}


def _cache_path() -> str:
    return str(settings.cache_dir / "label_cache.json")


def _load_cache() -> list[dict]:
    try:
        with open(_cache_path()) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


def _save_cache(records: list[dict]) -> None:
    settings.cache_dir.mkdir(parents=True, exist_ok=True)
    with open(_cache_path(), "w") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)


def _collect_passing(n: int) -> list[RawDoc]:
    """Stream the HF slice and return the first n docs that pass the heuristics."""
    heuristics = HeuristicFilter(CurationConfig())
    ds = load_dataset(settings.hf_slice, split="train", streaming=True)
    docs: list[RawDoc] = []
    seen = 0
    for row in ds:
        seen += 1
        text = row.get("text") or ""
        if not text:
            continue
        doc = RawDoc(id=str(row.get("id", seen)), text=text)
        if heuristics.check(doc) is None:
            docs.append(doc)
            if len(docs) >= n:
                break
    print(f"collected {len(docs)} heuristic-passing docs (streamed {seen})")
    return docs


class Judge:
    def __init__(self) -> None:
        self.client = cohere.ClientV2(api_key=settings.cohere_api_key.get_secret_value())
        self._last_call = 0.0

    def _pace(self) -> None:
        wait = _MIN_INTERVAL - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)
        self._last_call = time.monotonic()

    @retry(
        retry=retry_if_exception_type(TooManyRequestsError),
        wait=wait_exponential(multiplier=2, max=70),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _score_batch(self, batch: list[RawDoc]) -> dict[int, int]:
        """Return {batch_index: edu_score} for the docs in this batch."""
        lines = [f"[{i}] {doc.text[:MAX_DOC_CHARS]}" for i, doc in enumerate(batch)]
        self._pace()
        resp = self.client.chat(
            model=settings.chat_model,
            messages=[
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": "\n\n".join(lines)},
            ],
            response_format=_RESPONSE_FORMAT,
            temperature=0,
            seed=42,
        )
        payload = json.loads(resp.message.content[0].text)
        out: dict[int, int] = {}
        for entry in payload.get("scores", []):
            idx, score = entry.get("id"), entry.get("score")
            if isinstance(idx, int) and isinstance(score, int) and 1 <= score <= 5:
                out[idx] = score
        return out

    def run(self, n: int = N_LABEL, batch_k: int = BATCH_K) -> None:
        records = _load_cache()
        labeled = {r["doc_id"] for r in records}
        docs = _collect_passing(n)
        todo = [d for d in docs if d.id not in labeled]
        print(f"{len(labeled)} already labeled; {len(todo)} to label this run")

        for start in range(0, len(todo), batch_k):
            batch = todo[start : start + batch_k]
            scores = self._score_batch(batch)
            for i, doc in enumerate(batch):
                if i in scores:
                    records.append(
                        {"doc_id": doc.id, "text": doc.text, "edu_score": scores[i], "rationale": ""}
                    )
            _save_cache(records)  # incremental: survive interruption / quota exhaustion
            print(f"labeled {min(start + batch_k, len(todo))}/{len(todo)}")

        hist = {s: 0 for s in range(1, 6)}
        for r in records:
            hist[r["edu_score"]] = hist.get(r["edu_score"], 0) + 1
        print(f"done: {len(records)} labels cached at {_cache_path()}")
        print("score histogram (1..5):", hist)


def run() -> None:
    parser = argparse.ArgumentParser(description="Command A edu-quality labeling")
    parser.add_argument("--n", type=int, default=N_LABEL, help="docs to label")
    parser.add_argument("--batch", type=int, default=BATCH_K, help="docs per chat call")
    args = parser.parse_args()
    Judge().run(n=args.n, batch_k=args.batch)


if __name__ == "__main__":
    run()
