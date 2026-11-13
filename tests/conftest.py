import uuid
from pathlib import Path

import pytest

# Fast in-memory hash embedder (0ms, no network, no torch model download)
from docchat.embeddings import HashEmbedder

# Recording spy fake LLM (no API key required, deterministic response)
from docchat.llm import FakeLLM

# The facade pipeline we just built
from docchat.pipeline import DocChat

# Absolute path to our sample test documents (leave, expense, security policies)
DATA = Path(__file__).parent.parent / "data"


@pytest.fixture
def docchat() -> DocChat:
    """A DocChat with fake parts: no model download, no API key, in memory."""
    app = DocChat(
        embedder=HashEmbedder(),  # Deterministic test embedder
        llm=FakeLLM(
            "You get 18 vacation days per year [1]."
        ),  # Deterministic fake reply citing [1]
        db_path=None,  # None forces ChromaDB EphemeralClient (100% in RAM)
        collection=f"test-{uuid.uuid4().hex[:8]}",  # Unique collection prevents test collisions
    )
    # Pre-loads sample test policies so tests have data ready immediately
    app.ingest_folder(DATA)
    return app