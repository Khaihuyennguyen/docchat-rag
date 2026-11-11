""" BM25 implementation for document rankin"""

import math
import re
from collections import Counter

TOKEN = re.compile(r"[a-z0-9]+")

def tokenize(text: str) -> list[str]:
    return TOKEN.findall(text.lower())

class BM25:
    def __init__(self, docs: list[str], k1: float = 1.6, b: float = 0.75) -> None:
        self.k1, self.b = k1, b
        self.docs = [tokenize(d) for d in docs]
        self.term_freqs = [Counter(d) for d in self.docs]
        self.avg_len = sum(len(d) for  d in self.docs)/ max(len(self.docs), 1)
        n = len(self.docs)
        doc_freq = Counter(word for d in self.docs for word in set(d))

        # rare word get high idf, common words (the) get a low one
        self.idf = {w: math.log(1 + (n - f + 0.5)/ (f + 0.5)) for w, f in doc_freq.items()}

    def scores(self, query: str) -> list[float]:
        words = tokenize(query)
        out = []
        for tf, doc in zip(self.term_freqs, self.docs):
            score = 0.0
            for w in words:
                if w not in tf:
                    continue
                f = tf[w]
                length_norm = 1 - self.b  + self.b * len(doc)/self.avg_len
                score += self.idf[w] * f * (self.k1 + 1)/ (f + self.k1 * length_norm)
            out.append(score)
        return out
    def top(self, query: str, k: int) -> list[tuple[int, float]]:
        ranked = sorted(enumerate(self.scores(query)), key = lambda x: x[1], reverse=True)
        return [(i, s) for i, s in ranked[:k] if s > 0]
