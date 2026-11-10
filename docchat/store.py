import chromadb
from .chunking import Chunk
from .embeddings import Embedder

class VectorStore:
    def __init__(self, embedder: Embedder, path: str | None = None, name: str = "docchat") -> None:
        self.embedder = embedder
        settings = chromadb.Settings(anonymized_telemetry=False)
        if path: # Production model
            self.client = chromadb.PersistentClient(path=path, settings=settings)
        else: # Test mode
            self.client = chromadb.EphemeralClient(settings=settings)

    
        self.collection = self.client.get_or_create_collection(
            name, metadata={"hnsw:space": "cosine"}
        )

    def add(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        # we use upsert nnot inser to avoid error id exists or create duplicate 
        # it would update instead
        self.collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[{"source": c.source, "page": c.page} for c in chunks],
            embeddings=self.embedder.embed([c.text for c in chunks]),
        )

    def count(self) -> int:
        """
        show us how many chunks inside the database
        
        """
        return self.collection.count()
    def all_chunks(self) -> list[Chunk]:
        got = self.collection.get(include=["documents", "metadatas"])
        return [
            Chunk(i, meta["source"], int(meta["page"]), doc)
            for i, doc, meta in zip(got["ids"], got["documents"], got["metadatas"])
        ]

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        total = self.count()
        if total == 0:
            return []
        # turn the user questions into vector 
        # then if the user ask for the top 4 results, but data only has 2, ask for 2 so it will not complain
        res = self.collection.query(
            query_embeddings = self.embedder.embed([query]),
            n_results = min(k, total),
            include=["documents", "metadatas", "distances"]
        )
        hits = []

        for i, doc, meta, dist in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            similarity = 1.0 - dist
            hits.append((Chunk(i, meta["source"], int(meta["page"]), doc), similarity))
        return hits