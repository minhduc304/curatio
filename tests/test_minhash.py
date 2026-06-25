"""MinHash load-bearing assumption: variants stay similar, unrelated docs don't."""
from __future__ import annotations

import json
from pathlib import Path

from eval import inject
from src.curate.minhash import MinHasher

_DOC = (
    "The mitochondria is the powerhouse of the cell. It produces ATP through "
    "oxidative phosphorylation. This process is essential for cellular energy. "
    "Without it, complex life as we know it could not exist on this planet."
)
_OTHER = (
    "Quarterly revenue rose twelve percent on strong cloud demand. The board "
    "approved a dividend increase. Analysts expect margins to expand next year "
    "as the company scales its data-center footprint across new regions."
)


def test_identical_text_jaccard_one() -> None:
    h = MinHasher()
    assert h.jaccard(h.signature(_DOC), h.signature(_DOC)) == 1.0


def test_unrelated_docs_low_jaccard() -> None:
    h = MinHasher()
    assert h.jaccard(h.signature(_DOC), h.signature(_OTHER)) < 0.1


def test_exported_params_match_seed() -> None:
    """The committed models/minhash.json (loaded by the Rust worker) must equal the
    seeded Python params — guards cross-runtime parity against any RNG drift."""
    committed = json.loads(Path("models/minhash.json").read_text())
    assert committed == MinHasher().params()


def test_inject_variants_stay_similar() -> None:
    """Every variant type in eval/inject.py must keep high surface overlap with its
    original — this is what lets MinHash routing co-locate them and verification
    confirm them as true near-dups."""
    h = MinHasher()
    base = h.signature(_DOC)
    for variant in (inject._truncate(_DOC), inject._paraphrase(_DOC), inject._wrap(_DOC)):
        assert h.jaccard(base, h.signature(variant)) >= 0.5
