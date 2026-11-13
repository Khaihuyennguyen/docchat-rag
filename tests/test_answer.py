from docchat.answer import NO_ANSWER, build_prompt, cited_numbers
from docchat.chunking import Chunk


def test_prompt_numbers_every_source():
    chunks = [Chunk("x", "a.md", 1, "alpha"), Chunk("y", "b.md", 2, "beta")]
    prompt = build_prompt("q?", chunks)
    assert "[1] (a.md, page 1)\nalpha" in prompt
    assert "[2] (b.md, page 2)\nbeta" in prompt
    assert prompt.endswith("Question: q?")


def test_cited_numbers_are_unique_and_sorted():
    assert cited_numbers("A [2]. B [1][2]. C [10].") == [1, 2, 10]


def test_ask_returns_only_cited_sources(docchat):
    result = docchat.ask("How many vacation days?")
    assert "18 vacation days" in result.text
    assert len(result.sources) == 1  # the fake answer cites [1] only
    system, prompt = docchat.llm.calls[-1]
    assert "ONLY the numbered sources" in system
    assert "Question: How many vacation days?" in prompt


def test_made_up_citation_numbers_are_dropped(docchat):
    docchat.llm.reply = "Something [99]."
    assert docchat.ask("vacation?").sources == []


def test_empty_store_says_i_dont_know(docchat):
    docchat.retriever.chunks = []
    assert docchat.ask("anything").text == NO_ANSWER