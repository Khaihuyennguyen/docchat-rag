"""Embedders turn text into vectors. Similar meaning -> nearby vectors"""

import hashlib
from typing import Protocol
import numpy as np


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]: ...

class LocalEmbedder:
    """all-MiniLM-L6-v2 (384 numbers per text) running locally via ONNX.
    It ships with chromadb. The first call downloads the model (~80 MB).
    
    """

    def __init__(self) -> None:
        from chromadb.utils.embedding_functions import DefaultEmbeddingFunction

        self._fn = DefaultEmbeddingFunction()
        """We don't import this heavy machine learning library at the top of the file. 
         Because loading neural networks takes 2–3 seconds. 
        We only load it if someone actually hires the real worker."""

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in vector] for vector in self._fn(texts)]


class HashEmbedder:
    def __init__(self, dim: int = 256) -> None:
        """
        Neuron networks are slow. For automated tests, we want our code to run in 0.05 seconds
        so we create a stunt double that turn words into numbers using pure math, zero downloads
        zero ai -> saving money on tokens
        """
        self.dim = dim
    def embed(self, texts:list[str]) -> list[list[float]]:
        """Fast, offline, deterministic embedder for tests.
        Each word is hashed into one of `dim` buckets ("bag of words").
        It knows nothing about meaning, but it is good enough to test plumbing.
        """
        vectors = []
        for text in texts:
            v = np.zeros(self.dim) # for each piece of text, starts an array of 256 zeros
            for word in text.lower().split():
                # take each word, removes punctuation, and hash the words with MD5
                # modules the dim give a bucket number between 0 to 255
                # then update that bucket
                word = word.strip(".,!?;:()[]\"'")
                bucket = int(hashlib.md5(word.encode()).hexdigest(), 16) % self.dim                
                v[bucket] += 1.0
            norm = np.linalg.norm(v)
            if norm == 0: # when the sentence is eimply
                v[0], norm = 1.0, 1.0
            vectors.append((v/norm).tolist())
        return vectors


    

