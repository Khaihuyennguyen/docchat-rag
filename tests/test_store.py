import uuid

from docchat.chunking import Chunk
from docchat.embeddings import HashEmbedder
from docchat.store import VectorStore


def make_store() -> VectorStore:
    return VectorStore(HashEmbedder(), path=None, name=f"test-{uuid.uuid4().hex[:8]}")


def test_search_returns_most_similar_chunk_first():
    store = make_store()
    store.add([
        Chunk("1", "a.md", 1, "vacation days and sick leave"),
        Chunk("2", "b.md", 1, "laptop password and phishing"),
    ])
    hits = store.search("how many vacation days", k=2)
    assert hits[0][0].id == "1"
    assert hits[0][1] > hits[1][1]


def test_upsert_does_not_duplicate():
    store = make_store()
    chunk = Chunk("1", "a.md", 1, "same text")
    store.add([chunk])
    store.add([chunk])
    assert store.count() == 1


def test_search_on_empty_store():
    assert make_store().search("anything", k=3) == []