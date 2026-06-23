"""Load and chunk the 70-document US EV dataset."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Document:
    doc_id: str
    query: str
    title: str
    content: str
    source_file: str


def parse_document(path: Path) -> Document:
    """Parse a dataset txt file into structured fields."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    query = ""
    title = ""
    content = raw

    qm = re.search(r"^Query:\s*(.+)$", raw, re.MULTILINE)
    tm = re.search(r"^Title:\s*(.+)$", raw, re.MULTILINE)
    if qm:
        query = qm.group(1).strip()
    if tm:
        title = tm.group(1).strip()

    if "Full Content:" in raw:
        content = raw.split("Full Content:", 1)[1].strip()
    elif title:
        content = re.sub(r"^Title:.*$", "", raw, count=1, flags=re.MULTILINE).strip()
        content = re.sub(r"^Link:.*$", "", content, flags=re.MULTILINE).strip()
        content = re.sub(r"^Snippet:.*$", "", content, flags=re.MULTILINE).strip()
        content = re.sub(r"^Query:.*$", "", content, flags=re.MULTILINE).strip()

    # Collapse excessive whitespace
    content = re.sub(r"\n{3,}", "\n\n", content)
    content = re.sub(r"[ \t]+", " ", content)

    return Document(
        doc_id=path.stem,
        query=query,
        title=title,
        content=content,
        source_file=path.name,
    )


def load_dataset(dataset_dir: Path) -> list[Document]:
    """Load all doc_*.txt files from dataset folder."""
    paths = sorted(dataset_dir.glob("doc_*.txt"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    return [parse_document(p) for p in paths]


def merge_corpus_text(docs: list[Document]) -> str:
    """Merge documents into a single corpus string."""
    parts = []
    for doc in docs:
        header = f"=== {doc.doc_id} | {doc.title or doc.query} ==="
        parts.append(f"{header}\n{doc.content}")
    return "\n\n".join(parts)


def extraction_chunks(docs: list[Document], max_chars: int = 2800) -> list[str]:
    """One extraction chunk per document (truncated for token cost)."""
    chunks = []
    for doc in docs:
        text = doc.content[:max_chars]
        if len(text) < 80:
            continue
        chunks.append(
            f"[{doc.doc_id}] Query: {doc.query}\nTitle: {doc.title}\n{text}"
        )
    return chunks


def rag_chunks(docs: list[Document], chunk_size: int = 900) -> list[str]:
    """Split documents into chunks for vector indexing."""
    all_chunks: list[str] = []
    for doc in docs:
        text = doc.content
        if len(text) <= chunk_size:
            if text.strip():
                all_chunks.append(f"[{doc.doc_id}] {doc.title}\n{text}")
            continue
        start = 0
        while start < len(text):
            piece = text[start : start + chunk_size].strip()
            if piece:
                all_chunks.append(f"[{doc.doc_id}] {doc.title}\n{piece}")
            start += chunk_size
    return all_chunks


def prepare_corpus(dataset_dir: Path, merged_path: Path) -> list[Document]:
    """Load dataset and write merged corpus file."""
    docs = load_dataset(dataset_dir)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(merge_corpus_text(docs), encoding="utf-8")
    return docs
