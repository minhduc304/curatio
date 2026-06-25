"""C7 — Parquet writer + sampled rejected log. See TDD §4.6.

Kept -> CuratedDoc buffered, flushed to data/clean-docs/part-*.parquet. Parquet
can't append in place, so each flush writes a complete, self-contained part file
(valid footer) — a partial run still leaves DuckDB-readable output (glob the dir).
Rejected -> sampled JSONL for inspection.
"""
from __future__ import annotations

import pyarrow as pa
import pyarrow.parquet as pq

from src.config import settings
from src.models import CuratedDoc, StageResult


class Sink:
    def __init__(self, flush_size: int = 256, rejected_sample_rate: int = 10) -> None:
        self.kept_dir = settings.data_dir / "clean-docs"
        self.kept_dir.mkdir(parents=True, exist_ok=True)
        self.rejected_path = settings.data_dir / "rejected-sample.jsonl"
        self.flush_size = flush_size
        self.rejected_sample_rate = rejected_sample_rate
        self._buf: list[CuratedDoc] = []
        self._part = 0
        self._rejected_seen = 0

    def write_kept(self, doc: CuratedDoc) -> None:
        self._buf.append(doc)
        if len(self._buf) >= self.flush_size:
            self.flush()

    def write_rejected(self, result: StageResult) -> None:
        self._rejected_seen += 1
        if self._rejected_seen % self.rejected_sample_rate == 0:
            with self.rejected_path.open("a") as f:
                f.write(result.model_dump_json() + "\n")

    def flush(self) -> None:
        if not self._buf:
            return
        table = pa.Table.from_pylist([d.model_dump() for d in self._buf])
        pq.write_table(table, self.kept_dir / f"part-{self._part:05d}.parquet")
        self._part += 1
        self._buf.clear()

    def close(self) -> None:
        self.flush()
