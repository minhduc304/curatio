"""C6b — MinHash signatures over word-shingles for content-aware dedup.

Two uses (see README "Limitations"):
  - routing: key the stream by an LSH band of the signature so near-duplicate docs
    co-locate on one partition — restores recall under scale-out.
  - verification: confirm an embedding-cosine candidate with surface Jaccard, so
    distinct-but-topically-similar docs aren't dropped — holds precision at scale.

Pure, deterministic integer arithmetic (no per-process hash salt, no float) so the
Rust worker can mirror it for parity. Universal hashing kept within uint64: with
a < 2^31 and a 32-bit shingle hash, a*h+b < 2^63, so no overflow before the mod.
"""
from __future__ import annotations

import zlib

import numpy as np

_PRIME = np.uint64(4294967291)  # largest prime < 2**32
_MAXHASH = np.uint64(4294967295)  # 2**32 - 1, empty-doc sentinel


class MinHasher:
    def __init__(self, num_perm: int = 128, shingle_size: int = 5, seed: int = 42) -> None:
        self.num_perm = num_perm
        self.shingle_size = shingle_size
        rng = np.random.RandomState(seed)
        self.a = rng.randint(1, 1 << 31, size=num_perm).astype(np.uint64)
        self.b = rng.randint(0, 1 << 31, size=num_perm).astype(np.uint64)

    def _shingle_hashes(self, text: str) -> set[int]:
        """Deterministic 32-bit CRC of each word k-gram (crc32 mirrors trivially in Rust)."""
        words = text.lower().split()
        k = self.shingle_size
        if len(words) < k:
            grams = [" ".join(words)] if words else []
        else:
            grams = [" ".join(words[i : i + k]) for i in range(len(words) - k + 1)]
        return {zlib.crc32(g.encode()) for g in grams}

    def signature(self, text: str) -> np.ndarray:
        """num_perm-length uint64 MinHash signature."""
        hs = self._shingle_hashes(text)
        if not hs:
            return np.full(self.num_perm, _MAXHASH, dtype=np.uint64)
        h = np.fromiter(hs, dtype=np.uint64, count=len(hs))
        hashed = (self.a[:, None] * h[None, :] + self.b[:, None]) % _PRIME  # (num_perm, |shingles|)
        return hashed.min(axis=1)

    @staticmethod
    def jaccard(a: np.ndarray, b: np.ndarray) -> float:
        """Estimated Jaccard = fraction of permutations whose minima agree."""
        return float(np.mean(a == b))

    def band_key(self, sig: np.ndarray, rows: int) -> bytes:
        """LSH routing key: the first `rows` minima. Docs sharing them co-locate."""
        return sig[:rows].tobytes()

    def params(self) -> dict[str, object]:
        """Permutation params for the shared models/minhash.json the Rust worker loads
        (so both runtimes hash identically without reproducing NumPy's RNG)."""
        return {
            "num_perm": self.num_perm,
            "shingle_size": self.shingle_size,
            "a": self.a.tolist(),
            "b": self.b.tolist(),
        }


if __name__ == "__main__":
    # Export the default-config params for the Rust worker. Deterministic (fixed
    # seed); re-run only if num_perm/shingle_size/seed change.
    import json
    from pathlib import Path

    out = Path("models/minhash.json")
    out.write_text(json.dumps(MinHasher().params()))
    print(f"wrote {out}")
