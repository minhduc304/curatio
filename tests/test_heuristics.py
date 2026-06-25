"""Pure-function tests for the heuristic stage (deterministic, no API). TDD §11."""
from __future__ import annotations

import pytest

from src.curate.heuristics import HeuristicFilter
from src.models import CurationConfig, RawDoc


@pytest.fixture
def hf() -> HeuristicFilter:
    return HeuristicFilter(CurationConfig())


def test_rejects_too_short(hf: HeuristicFilter) -> None:
    assert hf.check(RawDoc(id="1", text="short")) == "too_short"


def test_passes_clean_doc(hf: HeuristicFilter) -> None:
    clean = "This is a sufficiently long, clean English paragraph. " * 10
    assert hf.check(RawDoc(id="2", text=clean)) is None
