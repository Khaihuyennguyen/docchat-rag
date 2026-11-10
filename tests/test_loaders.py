from pathlib import Path

import pytest

from docchat.loaders import load_file, load_folder

DATA = Path(__file__).parent.parent / "data"


def test_load_folder_reads_every_markdown_file():
    pages = load_folder(DATA)
    assert {p.source for p in pages} == {"leave_policy.md", "expense_policy.md", "security_policy.md"}
    assert all(p.page == 1 for p in pages)


def test_unsupported_file_type_raises(tmp_path):
    bad = tmp_path / "photo.png"
    bad.write_bytes(b"not text")
    with pytest.raises(ValueError):
        load_file(bad)