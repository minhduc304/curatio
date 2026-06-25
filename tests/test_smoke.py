"""Smoke tests: imports + one live Embed v4 call. TDD §11.

load_dotenv() runs before the skipif so a .env COHERE_API_KEY is visible at
collection time (known gotcha carried from the RAG project).
"""
from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()


def test_imports() -> None:
    import src.api.app  # noqa: F401
    import src.curate.consumer  # noqa: F401
    import src.curate.dedup  # noqa: F401
    import src.curate.heuristics  # noqa: F401
    import src.curate.quality  # noqa: F401
    import src.embedding.service  # noqa: F401
    import src.models  # noqa: F401
    import src.source.producer  # noqa: F401


@pytest.mark.skipif(not os.environ.get("COHERE_API_KEY"), reason="no COHERE_API_KEY")
def test_live_embed() -> None:
    import cohere
    from cohere.errors import TooManyRequestsError

    co = cohere.ClientV2(api_key=os.environ["COHERE_API_KEY"])
    try:
        resp = co.embed(
            model="embed-v4.0",
            texts=["hello world"],
            input_type="clustering",
            output_dimension=256,
            embedding_types=["float"],
        )
    except TooManyRequestsError:
        pytest.skip("Cohere quota/rate limit exhausted (trial key)")
    assert len(resp.embeddings.float[0]) == 256
