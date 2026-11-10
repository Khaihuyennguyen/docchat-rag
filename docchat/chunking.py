"""split long pages into small overlapping chunks that fit in a prompt"""
import hashlib
from dataclasses import dataclass

from .loaders import Page

@dataclass
class Chunk:
    id: str 
    source: str
    page: int
    text: str

def split_text(text:str, size: int, overlap: int) -> list[str]:
    if overlap >= size:
        raise ValueError("overlap must be smaller than size")

    text = " ".join(text.split()) # collapse newlines and double spaces
    pieces: list[str] = []
    start = 0

    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            # prefer to cut at the end of the sentence, else at a space,
            # so we never chop a word in hafl
            cut = text.rfind(". ", start, end)
            if cut <= start + size // 2:
                cut = text.rfind(" ", start, end)
            if cut > start + size // 2:
                end = cut + 1
        pieces.append(text[start:end].strip())
        if end == len(text):
            break

        next_start = max(end - overlap, start + 1) # alwasy move forward
        if text[next_start - 1] != " ":
            # we landed inside a word: skip ahead to the st art of the next word:
            space = text.find( " ", next_start, end)
            if space != -1:
                next_start = space + 1
        start = next_start

    return [p for p in pieces if p]


def chunk_pages(pages: list[Page], size: int, overlap: int) -> list[Chunk]:
    chunks: list[Chunk] = []
    for page in pages:
        for i, piece in enumerate(split_text(page.text, size, overlap)):
            raw = f"{page.source}|{page.page}|{i}|{piece}"
            chunk_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
            chunks.append(Chunk(chunk_id, page.source, page.page, piece))
    return chunks