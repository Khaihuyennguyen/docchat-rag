""" Turn files on disk into plain text pages. """

from dataclasses import dataclass

from pathlib import Path

from pypdf import PdfReader


SUPPORTED = {".pdf", ".md", ".txt"}

@dataclass
class Page:
    source: str # file name, shown to the users in citeations
    page: int # 1 - based page number (always 1 for md and txt)
    text: str

def load_file(path: Path) -> list[Page]:
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        reader = PdfReader(str(Path))
        return [
            Page(path.name, number, page.extract_text() or "")
            for number, page in enumerate(reader.pages, start=1)
        ]
    if suffix in {".md", ".txt"}:
        return [Page(path.name, 1, path.read_text(encoding="utf-8"))]

    raise ValueError(f"unsupported file: {suffix}")

def load_folder(folder: Path) -> list[Page]:
    pages: list[Page] = []
    for path in sorted(folder.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED:
            pages.extend(load_file(path))
    return pages
