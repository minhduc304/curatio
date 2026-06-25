"""C1 — replay a fixed HF slice into the raw-docs topic. See TDD §4.1.

Streams `settings.hf_slice`, sends up to `sample_size` docs to `raw_topic`. Messages
are keyed by a MinHash-band of the *content* (not the doc id), so near-duplicate docs
land on the same partition and partition-local dedup catches them under scale-out
(see README "Limitations"). The band value is ~uniform, so spread stays even.
"""
from __future__ import annotations

from confluent_kafka import Producer as KafkaProducer
from datasets import load_dataset

from src.config import settings
from src.curate.minhash import MinHasher
from src.models import CurationConfig, RawDoc


class Producer:
    def __init__(self) -> None:
        self.producer = KafkaProducer({"bootstrap.servers": settings.kafka_bootstrap})
        cfg = CurationConfig()
        self.hasher = MinHasher(cfg.minhash_num_perm, cfg.minhash_shingle_size)
        self.band_rows = cfg.lsh_band_rows

    def run(self) -> None:
        ds = load_dataset(settings.hf_slice, split="train", streaming=True)
        sent = 0
        for row in ds:
            if sent >= settings.sample_size:
                break
            text = row.get("text") or ""
            if not text:
                continue
            doc = RawDoc(id=str(row.get("id", sent)), text=text)
            band_key = self.hasher.band_key(self.hasher.signature(doc.text), self.band_rows)
            self.producer.produce(
                settings.raw_topic,
                key=band_key,
                value=doc.model_dump_json().encode(),
            )
            sent += 1
            if sent % 1000 == 0:
                self.producer.poll(0)  # serve delivery callbacks, drain queue
                print(f"produced {sent}")
        self.producer.flush()
        print(f"done: produced {sent} docs to {settings.raw_topic}")


if __name__ == "__main__":
    Producer().run()
