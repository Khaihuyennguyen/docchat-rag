from docchat.bm25 import BM25, tokenize


def test_tokenize_lowercases_and_drops_punctuation():
    # Verifies punctuation like "!" and "," are stripped and letters are lowercased
    assert tokenize("Hello, World! 2FA") == ["hello", "world", "2fa"]


def test_rare_exact_word_wins():
    # 3 docs: common words like "the" and "sat" appear in multiple docs; "Vaultkeep" appears only in doc 2
    docs = ["the cat sat on the mat", "the dog sat on the log", "use Vaultkeep for passwords"]
    top = BM25(docs).top("vaultkeep", k=3)
    # The top hit must be index 2 (the doc containing 'Vaultkeep')
    assert top[0][0] == 2
    # Only 1 document matched; docs with score 0 are dropped
    assert len(top) == 1


def test_no_match_returns_empty():
    # If the query term does not exist anywhere in the corpus, return an empty list
    assert BM25(["a b c"]).top("zebra", k=5) == []