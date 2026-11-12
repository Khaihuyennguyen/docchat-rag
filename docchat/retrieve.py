""" Hybrid retrieval: combine vector search and keyword search. """

from collections import defaultdict

from .bm25 import BM25
from .chunking import Chunk
from .store import VectorStore

def rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    """
        Reciprocal Rank Fusion: score = sum of 1 / (k + rank) across lists

        it only uses ranks, so we never have to compare a cosine score with a BM25 sore (they are on totally different scales)
    """

    scores: dict[str, float] = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start = 1):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, store: VectorStore, top_k: int = 4) -> None:
        self.store = store
        self.top_k = top_k
        self.refresh()

    def refresh(self) -> None:
        """Rebuild the keyword index. Call this after adding docuents"""
        self.chunks = self.store.all_chunks()
        self.by_id = {c.id: c for c in self.chunks}
        self.bm25 = BM25([c.text for c in self.chunks])

    def retrieve(self, query: str, k: int | None = None, mode: str = "hybrid") -> list[Chunk]:
        k = k or self.top_k
        if not self.chunks:
            return []
        dense = [c.id for c, _ in self.store.search(query, k * 3)]
        sparse = [self.chunks[i].id for i, _ in self.bm25.top(query, k * 3)]
        if mode == "dense":
            ids = dense[:k]
        elif mode == "sparse":
            ids = sparse[:k]
        elif mode == "hybrid":
            ids = [doc_id for doc_id, _ in rrf([dense, sparse])[:k]]
        else:
            raise ValueError(f"unknown mode: {mode}")
        return [self.by_id[i] for i in ids]
