"""Online dedup tests — deterministic, no API. TDD §4.5, §11."""
from __future__ import annotations

import numpy as np
import pytest

from src.curate.dedup import OnlineDedup
from src.models import CurationConfig


@pytest.fixture
def dedup() -> OnlineDedup:
    return OnlineDedup(CurationConfig(), max_elements=100)


def _unit(vec: list[float]) -> np.ndarray:
    v = np.asarray(vec, dtype=np.float32)
    return (v / np.linalg.norm(v)).reshape(1, -1)


def test_first_doc_is_kept(dedup: OnlineDedup) -> None:
    emb = _unit([1.0] + [0.0] * (dedup.config.embed_dim - 1))
    assert dedup.check_and_add("a", emb) is None
    assert dedup.size == 1


def test_near_identical_is_rejected(dedup: OnlineDedup) -> None:
    base = np.random.default_rng(0).standard_normal(dedup.config.embed_dim).astype(np.float32)
    a = (base / np.linalg.norm(base)).reshape(1, -1)
    near = base + 1e-3 * np.random.default_rng(1).standard_normal(dedup.config.embed_dim).astype(np.float32)
    near = (near / np.linalg.norm(near)).reshape(1, -1)

    assert dedup.check_and_add("a", a) is None
    sim = dedup.check_and_add("b", near)
    assert sim is not None and sim >= dedup.config.dedup_cosine_threshold
    assert dedup.size == 1  # duplicate not added


def test_orthogonal_doc_is_kept(dedup: OnlineDedup) -> None:
    d = dedup.config.embed_dim
    a = _unit([1.0] + [0.0] * (d - 1))
    b = _unit([0.0, 1.0] + [0.0] * (d - 2))
    assert dedup.check_and_add("a", a) is None
    assert dedup.check_and_add("b", b) is None
    assert dedup.size == 2
