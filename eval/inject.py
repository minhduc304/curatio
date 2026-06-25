"""Build the labeled near-duplicate set for dedup eval. See TDD §7.2.

Take N held-out clean docs (from the cached Command A label set); per doc emit two
near-dup variants: truncation (first 70%) and either a light paraphrase (sentence
rotation) or a nav/footer wrap. Ground truth is group membership: within each group
{original + variants}, whichever the online index sees first is legitimately kept;
every later member is a true near-duplicate that should be dropped. This framing is
order-independent, so the stream can be shuffled (TDD §7.2).
"""
from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass

from src.config import settings

N_ORIGINALS = 200
SEED = 42

_NAV = "Home  News  Topics  About  Contact  Login  Search\n\n"
_FOOT = "\n\nShare this page  Subscribe for updates  Back to top  Print"


@dataclass
class EvalDoc:
    doc_id: str
    text: str
    group: str  # source-original id; variants share their parent's group
    is_variant: bool


def _truncate(text: str) -> str:
    return text[: int(len(text) * 0.7)]


def _paraphrase(text: str) -> str:
    """Light paraphrase: rotate sentence order (same tokens, reordered)."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return " ".join(parts[1:] + parts[:1])


def _wrap(text: str) -> str:
    """Boilerplate wrap with innocuous nav/footer (no heuristic boilerplate markers)."""
    return _NAV + text + _FOOT


def _n_sentences(text: str) -> int:
    return len(re.split(r"(?<=[.!?])\s+", text.strip()))


def _load_originals(n: int) -> list[dict]:
    with open(settings.cache_dir / "label_cache.json") as f:
        records = json.load(f)
    return records[:n]


def build(n: int = N_ORIGINALS) -> list[EvalDoc]:
    """Return originals + 2 near-dup variants each, shuffled (seeded)."""
    docs: list[EvalDoc] = []
    for r in _load_originals(n):
        oid, text = r["doc_id"], r["text"]
        docs.append(EvalDoc(oid, text, oid, False))
        docs.append(EvalDoc(f"{oid}::trunc", _truncate(text), oid, True))
        second = _paraphrase(text) if _n_sentences(text) >= 2 else _wrap(text)
        docs.append(EvalDoc(f"{oid}::v2", second, oid, True))
    random.Random(SEED).shuffle(docs)
    return docs
