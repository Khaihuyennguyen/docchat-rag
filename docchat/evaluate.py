"""Measure retrieval quality: does the right document come back for each questions"""

import json
from dataclasses import dataclass
from pathlib import Path
from .retrieve import HybridRetriever


@dataclass
class EvalCase:
    question: str
    expected_source: str
def load_cases(path: str | Path) -> list[EvalCase]:
    # linking the questions with the sources
    cases = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            cases.append(EvalCase(row["question"], row["expected_source"]))
    return cases
def evaluate(retriever: HybridRetriever, cases: list[EvalCase], k: int = 4, mode: str = "hybrid") -> dict:
    """hit_rate: share of questions where the right file is in the top k
    MRR (mean reciporical rank): 1.0 if it is always ranked first, 0.5 if second..."""

    hits, reciprocal_ranks = 0, 0.0
    for case in cases:
        sources = [
            c.source for c in retriever.retrieve(case.question, k=k, mode=mode)
        ]

        if case.expected_source in sources:
            hits += 1
            reciprocal_ranks += 1.0 / (sources.index(case.expected_source) + 1)

    n = max(len(cases), 1)
    return {"mode": mode, "k": k, "n": len(cases), "hit_rate":hits / n, "mrr": reciprocal_ranks/ n}
