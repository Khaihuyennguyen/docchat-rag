"""Docchat wires everything together and the cli and the api both use it"""

from .answer import Answer, answer_question

from .chunking import chunk_pages
from .config import settings
from .embeddings import Embedder, LocalEmbedder
from .llm import LLM, ClaudeLLM
from .loaders import load_folder
from pathlib import Path 
from .retrieve import HybridRetriever
from .store import VectorStore


class DocChat:
    def __init__(
        self,
        embedder: Embedder | None = None,
        llm: LLM | None = None,
        db_path: str | None = settings.db_path,
        collection: str = "docchat",
    ) -> None:
        # if no embedder passed, default to production LocalEmbedder()
        self.store = VectorStore(embedder or LocalEmbedder(), db_path, collection)
        # wire the vector store into the hybrid retriever with top_k from seetings
        self.retriever = HybridRetriever(self.store, settings.top_k)

        self._llm = llm

    @property
    def llm(self) -> LLM:
        if self._llm is None: # Create the real client only when first needed
            self._llm = ClaudeLLM(settings.model)
        return self._llm

    def ingest_folder(self, folder: str | Path) -> int:
        # Step 1: read all supported files into Page objcts

        pages = load_folder(Path(folder))
        # split pages into chunks objects repecting token/char boundaries 
        chunks = chunk_pages(pages, settings.chunk_size, settings.chunk_overlap)
        # upsert chunks into ChromaDB (computes vector emboddings)
        self.store.add(chunks)
        # reindex all chunks into the BM25 keyword search engine
        self.retriever.refresh()
        # return count of ingested chunks
        return len(chunks)
    def ask(self, question: str) -> Answer:
        # hands question, retriever, and LLM to answer question for gorunded genenration
        return answer_question(question, self.retriever, self.llm)