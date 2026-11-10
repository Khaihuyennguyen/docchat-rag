import pytest

from docchat.chunking import chunk_pages, split_text
from docchat.loaders import Page

TEXT = "One two three. " * 100  # 1500 characters


def test_chunks_respect_max_size():
    pieces = split_text(TEXT, size=200, overlap=50)
    assert len(pieces) > 1
    assert all(len(p) <= 200 for p in pieces)


def test_chunks_overlap():
    pieces = split_text(TEXT, size=200, overlap=50)
    # the end of one chunk appears again at the start of the next
    assert pieces[0][-20:].strip().split()[-1] in pieces[1]


def test_words_are_never_cut_in_half():
    pieces = split_text(TEXT, size=200, overlap=50)
    for p in pieces:
        for word in p.replace(".", "").split():
            assert word in {"One", "two", "three"}


def test_overlap_must_be_smaller_than_size():
    with pytest.raises(ValueError):
        split_text("hello", size=10, overlap=10)


def test_chunk_ids_are_stable():
    pages = [Page("a.md", 1, TEXT)]
    first = [c.id for c in chunk_pages(pages, 200, 50)]
    second = [c.id for c in chunk_pages(pages, 200, 50)]
    assert first == second
    assert len(set(first)) == len(first)  # and unique