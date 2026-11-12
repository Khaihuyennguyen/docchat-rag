import pytest

from docchat.retrieve import rrf


def test_rrf_rewards_items_ranked_high_in_both_lists():
    fused = rrf([["a", "b", "c"], ["b", "a", "d"]])
    ids = [doc_id for doc_id, _ in fused]
    assert set(ids[:2]) == {"a", "b"}
    assert ids[-1] in {"c", "d"}


def test_hybrid_finds_exact_product_name(docchat):
    chunks = docchat.retriever.retrieve("What is Vaultkeep?", k=2)
    assert chunks[0].source == "security_policy.md"


@pytest.mark.parametrize("mode", ["dense", "sparse", "hybrid"])
def test_every_mode_returns_at_most_k(docchat, mode):
    assert len(docchat.retriever.retrieve("vacation", k=2, mode=mode)) <= 2


def test_unknown_mode_raises(docchat):
    with pytest.raises(ValueError):
        docchat.retriever.retrieve("vacation", mode="magic")