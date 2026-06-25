"""C4 — batched Embed v4 with disk cache, retry, rate-limit awareness. See TDD §4.3.

co.embed(model="embed-v4.0", input_type="clustering", output_dimension=embed_dim).
Cache key = sha256(text) + ":" + dim (diskcache). Retry on TooManyRequestsError with
exponential backoff to 70s; sustained throttling propagates so the consumer can pause
its Kafka poll (backpressure, TDD §4.7).
"""
from __future__ import annotations

import hashlib

import cohere
import numpy as np
from cohere.errors import TooManyRequestsError  # OQ: verify path against installed SDK
from diskcache import Cache
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import settings


class EmbeddingService:
    def __init__(self, embed_dim: int = 1024, batch_size: int = 96) -> None:
        self.embed_dim = embed_dim
        self.batch_size = batch_size
        self.client = cohere.ClientV2(api_key=settings.cohere_api_key.get_secret_value())
        self.cache = Cache(str(settings.cache_dir / "embeddings"))

    def _key(self, text: str) -> str:
        return f"{hashlib.sha256(text.encode()).hexdigest()}:{self.embed_dim}"

    @retry(
        retry=retry_if_exception_type(TooManyRequestsError),
        wait=wait_exponential(multiplier=2, max=70),
        stop=stop_after_attempt(6),
        reraise=True,
    )
    def _embed_api(self, texts: list[str]) -> list[list[float]]:
        resp = self.client.embed(
            model=settings.embed_model,
            texts=texts,
            input_type="clustering",
            output_dimension=self.embed_dim,
            embedding_types=["float"],
        )
        return resp.embeddings.float  # type: ignore[union-attr]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), embed_dim) float32 array; cache-aware, batched."""
        out: list[list[float] | None] = [None] * len(texts)
        misses: list[int] = []
        for i, text in enumerate(texts):
            cached = self.cache.get(self._key(text))
            if cached is not None:
                out[i] = cached  # type: ignore[assignment]
            else:
                misses.append(i)

        for start in range(0, len(misses), self.batch_size):
            idxs = misses[start : start + self.batch_size]
            vectors = self._embed_api([texts[i] for i in idxs])
            for i, vec in zip(idxs, vectors):
                self.cache.set(self._key(texts[i]), vec)
                out[i] = vec

        return np.asarray(out, dtype=np.float32)
