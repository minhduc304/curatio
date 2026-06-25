"""C3 — C4/Gopher-style rule filters (no API). See TDD §4.2.

Pure functions; mirrored exactly in rust/src/heuristics.rs for parity (NFR5).
Rules are checked in a fixed order; the first failure's reason is returned.
"""
from __future__ import annotations

import py3langid as langid  # drop-in fork of langid: identical classifications, ~6x faster

from src.models import CurationConfig, RawDoc, RejectReason

# Lock langid to a yes/no English decision space would over-trigger; we keep the
# full model and compare the top label to the target language below.

_BOILERPLATE_MARKERS = (
    "lorem ipsum",
    "all rights reserved",
    "terms of service",
    "enable javascript",
)


class HeuristicFilter:
    def __init__(self, config: CurationConfig) -> None:
        self.config = config

    def check(self, doc: RawDoc) -> RejectReason | None:
        """Return a reject reason, or None if the doc passes."""
        text = doc.text
        n = len(text)

        if n < self.config.min_chars:
            return "too_short"
        if n > self.config.max_chars:
            return "too_long"

        # Symbol ratio: non-alphanumeric, non-whitespace chars over total. Counts
        # punctuation/markup but not spaces, so ordinary prose passes.
        symbols = sum(1 for c in text if not c.isalnum() and not c.isspace())
        if symbols / n > self.config.max_symbol_ratio:
            return "symbol_ratio"

        # Mean line length: nav bars / link lists have many tiny lines.
        lines = text.splitlines() or [text]
        mean_line_len = sum(len(ln) for ln in lines) / len(lines)
        if mean_line_len < 20:
            return "short_lines"

        lowered = text.lower()
        if any(marker in lowered for marker in _BOILERPLATE_MARKERS):
            return "boilerplate"

        lang, _ = langid.classify(text)
        if lang != self.config.target_language:
            return "wrong_language"

        return None
