
import numpy as np

from docchat.embeddings import HashEmbedder


def test_vectors_have_unit_length():
    vectors = HashEmbedder(dim=64).embed(["hello world", "", "coffee"])
    for v in vectors:
        assert len(v) == 64
        assert abs(np.linalg.norm(v) - 1.0) < 1e-9


def test_same_text_same_vector():
    e = HashEmbedder()
    assert e.embed(["vacation days"]) == e.embed(["vacation days"])


def test_shared_words_mean_higher_similarity():
    a, b, c = HashEmbedder().embed(["vacation days policy", "vacation days", "laptop password"])
    assert np.dot(a, b) > np.dot(a, c)