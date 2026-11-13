from pathlib import Path

from docchat.evaluate import evaluate, load_cases

CASES = Path(__file__).parent.parent / "evals" / "questions.jsonl"


def test_load_cases():
    cases = load_cases(CASES)
    assert len(cases) == 12
    assert cases[0].expected_source == "leave_policy.md"


def test_metrics_are_between_zero_and_one(docchat):
    report = evaluate(docchat.retriever, load_cases(CASES), k=2, mode="hybrid")
    assert report["n"] == 12
    assert 0.0 <= report["mrr"] <= report["hit_rate"] <= 1.0


def test_perfect_score_when_k_covers_every_file(docchat):
    # there are only 3 files, so with k=50 the right file is always returned
    report = evaluate(docchat.retriever, load_cases(CASES), k=50, mode="hybrid")
    assert report["hit_rate"] == 1.0