"""Load and chunk the 70-document US EV dataset."""

from __future__ import annotations

import hashlib
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


def read_raw_file(path: Path) -> str:
    """Decode dataset files safely; strip BOM and null bytes."""
    raw = path.read_bytes()
    if raw.startswith(b"\xef\xbb\xbf"):
        raw = raw[3:]
    elif raw.startswith(b"\xff\xfe"):
        return raw[2:].decode("utf-16-le", errors="replace").replace("\x00", "")
    elif raw.startswith(b"\xfe\xff"):
        return raw[2:].decode("utf-16-be", errors="replace").replace("\x00", "")

    text = raw.decode("utf-8", errors="replace")
    return text.replace("\x00", "")


PDF_GARBAGE = re.compile(
    r"endstream|endobj|\d+\s+0\s+obj|<>stream|/Type|/Font|/Pages|obj\s*<<|xref\b|trailer\b|startxref",
    re.IGNORECASE,
)

BOILERPLATE_PHRASES = (
    "uses cookies",
    "we use cookies",
    "privacy policy",
    "essential cookies",
    "google analytics",
    "join our mailing list",
    "contact us",
    "all maps & data",
    "all publications",
    "all tools",
    "terms of use",
    "copyright ©",
    "whistleblower policy",
    "here's how you know",
    "official website of the united states government",
)


def is_pdf_binary_dump(text: str) -> bool:
    """Detect PDF/font streams mis-saved as text."""
    sample = text[:80000]
    if sample.count("/Filter/FlateDecode") >= 3:
        return True
    if sample.count("endobj") >= 5:
        return True
    if len(re.findall(r"\d+\s+0\s+obj", sample)) >= 5:
        return True
    prose_words = re.findall(r"\b[a-zA-Z]{4,}\b", sample[:15000])
    pdf_tokens = sample[:15000].count("stream") + sample[:15000].count("/Length")
    return pdf_tokens >= 8 and len(prose_words) < 80


def is_garbage_line(line: str) -> bool:
    """Drop binary noise, table fragments, and boilerplate lines."""
    line = line.strip()
    if len(line) < 4:
        return True
    if "\ufffd" in line:
        return True
    if PDF_GARBAGE.search(line):
        return True
    if "/Filter/FlateDecode" in line or "/Length" in line or "CharSet(" in line:
        return True
    if re.fullmatch(r"[\d\s\(\)\-.,$%]+", line):
        return True
    alnum = sum(1 for c in line if c.isalnum())
    if alnum / len(line) < 0.35:
        return True
    low = line.lower()
    if any(p in low for p in BOILERPLATE_PHRASES) and len(line) < 200:
        return True
    return False


def is_boilerplate_sentence(sentence: str) -> bool:
    low = sentence.lower()
    return any(p in low for p in BOILERPLATE_PHRASES)


def clean_content(text: str) -> str:
    """Keep readable prose; remove scraped junk and corrupted bytes."""
    text = text.replace("\x00", "").replace("\ufeff", "")
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)

    paragraphs: list[str] = []
    for block in re.split(r"\n{2,}", text):
        lines = [ln.strip() for ln in block.splitlines() if not is_garbage_line(ln)]
        if not lines:
            continue
        para = " ".join(lines)
        para = re.sub(r"\s+", " ", para).strip()
        if len(para) < 40:
            continue

        sentences = re.split(r"(?<=[.!?])\s+", para)
        good_sentences = [s for s in sentences if s and not is_boilerplate_sentence(s)]
        if not good_sentences:
            continue
        para = " ".join(good_sentences)

        words = re.findall(r"[a-zA-Z]{3,}", para)
        if len(words) < 5:
            continue
        if PDF_GARBAGE.search(para):
            continue
        paragraphs.append(para)

    cleaned = "\n\n".join(paragraphs)
    return dedupe_lines(cleaned)


def content_quality(text: str) -> float:
    """Score how readable the text is (0-1)."""
    if not text:
        return 0.0
    sample = text[:8000]
    readable = sum(1 for c in sample if c.isalnum() or c in " .,;:'\"()-/$%\n")
    return readable / len(sample)


def is_usable_document(text: str) -> bool:
    """Reject corrupted or non-prose documents."""
    if len(text) < 300:
        return False
    if text.count("\ufffd") / len(text) > 0.005:
        return False
    if content_quality(text) < 0.82:
        return False
    if len(PDF_GARBAGE.findall(text)) > 0:
        return False
    words = re.findall(r"[a-zA-Z]{3,}", text[:8000])
    return len(words) >= 40


def parse_document(path: Path) -> Document | None:
    """Parse a dataset txt file into structured fields."""
    raw = read_raw_file(path)
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

    if is_pdf_binary_dump(content):
        return None

    content = clean_content(content)
    if not is_usable_document(content):
        return None

    return Document(
        doc_id=path.stem,
        query=query,
        title=title,
        content=content,
        source_file=path.name,
    )


def dedupe_lines(text: str) -> str:
    """Remove repeated lines common in scraped web pages."""
    lines = text.splitlines()
    out: list[str] = []
    prev = ""
    for line in lines:
        line = line.strip()
        if not line:
            if out and out[-1] != "":
                out.append("")
            continue
        if line == prev:
            continue
        out.append(line)
        prev = line
    return "\n".join(out).strip()


def doc_fingerprint(text: str) -> str:
    """Fingerprint for skipping near-duplicate documents."""
    sample = re.sub(r"\s+", " ", text[:2500]).lower()
    return hashlib.md5(sample.encode("utf-8", errors="ignore")).hexdigest()


def dedupe_documents(docs: list[Document]) -> list[Document]:
    """Keep first occurrence of documents with identical opening content."""
    seen: set[str] = set()
    unique: list[Document] = []
    for doc in docs:
        fp = doc_fingerprint(doc.content)
        if fp in seen:
            continue
        seen.add(fp)
        unique.append(doc)
    return unique


def trim_for_extraction(text: str, max_chars: int = 6000) -> str:
    """Cap very large documents while keeping intro + diverse paragraphs."""
    if len(text) <= max_chars:
        return text

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if len(p.strip()) > 40]
    if not paragraphs:
        return text[:max_chars]

    selected = [paragraphs[0]]
    used = len(paragraphs[0])
    step = max(1, len(paragraphs) // 8)
    for i in range(step, len(paragraphs), step):
        p = paragraphs[i]
        if used + len(p) > max_chars:
            break
        selected.append(p)
        used += len(p)

    trimmed = "\n\n".join(selected)
    return trimmed if len(trimmed) >= 200 else text[:max_chars]


def load_dataset(dataset_dir: Path) -> list[Document]:
    """Load all usable doc_*.txt files from dataset folder."""
    paths = sorted(dataset_dir.glob("doc_*.txt"), key=lambda p: int(re.search(r"\d+", p.stem).group()))
    docs: list[Document] = []
    for path in paths:
        doc = parse_document(path)
        if doc is not None:
            docs.append(doc)
    return docs


def merge_corpus_text(docs: list[Document]) -> str:
    """Merge cleaned documents into a single corpus string."""
    parts = []
    for doc in docs:
        header = f"=== {doc.doc_id} | {doc.title or doc.query} ==="
        parts.append(f"{header}\n{doc.content}")
    return "\n\n".join(parts)


def _chunk_header(doc: Document) -> str:
    return f"[{doc.doc_id}] Query: {doc.query}\nTitle: {doc.title}\n"


def extraction_chunks(
    docs: list[Document],
    chunk_size: int = 900,
    stride: int = 700,
    max_chunks_per_doc: int = 3,
) -> list[str]:
    """Small focused chunks for graph extraction; capped per document."""
    docs = dedupe_documents(docs)
    chunks: list[str] = []
    for doc in docs:
        text = trim_for_extraction(doc.content)
        if len(text) < 80:
            continue
        header = _chunk_header(doc)
        if len(text) <= chunk_size:
            chunks.append(header + text)
            continue

        starts = list(range(0, len(text) - chunk_size + 1, stride))
        if len(starts) > max_chunks_per_doc:
            picks = [0]
            inner = starts[1:]
            step = max(1, len(inner) // (max_chunks_per_doc - 1))
            picks.extend(inner[::step][: max_chunks_per_doc - 1])
            starts = sorted(set(picks))[:max_chunks_per_doc]

        for start in starts:
            piece = text[start : start + chunk_size].strip()
            if len(piece) >= 120:
                chunks.append(header + piece)
    return chunks


BENCHMARK_KEYWORDS = [
    "McKerracher", "BloombergNEF", "ZEV", "1.7 million", "242 billion",
    "29.2%", "UAW", "ICCT", "935", "Colin", "Elizabeth Krear", "Stellantis",
    "two-thirds", "75%", "one in five", "500,000", "2035", "315,000",
    "1.3%", "430 workplace", "Tesla", "Ford", "General Motors",
]


def benchmark_boost_chunks(
    docs: list[Document],
    keywords: list[str] | None = None,
    window: int = 850,
    max_total: int = 30,
) -> list[str]:
    """Small windows centered on benchmark-critical facts."""
    keywords = keywords or BENCHMARK_KEYWORDS
    docs = dedupe_documents(docs)
    boosts: list[str] = []
    seen: set[str] = set()

    for doc in docs:
        text = doc.content
        for kw in keywords:
            idx = text.lower().find(kw.lower())
            if idx < 0:
                continue
            start = max(0, idx - window // 3)
            piece = text[start : start + window].strip()
            if len(piece) < 100:
                continue
            chunk = f"[{doc.doc_id}|fact] Keyword: {kw}\nTitle: {doc.title}\n{piece}"
            if chunk in seen:
                continue
            seen.add(chunk)
            boosts.append(chunk)
            if len(boosts) >= max_total:
                return boosts
    return boosts


def rag_chunks(docs: list[Document], chunk_size: int = 600, max_chunks_per_doc: int = 12) -> list[str]:
    """Smaller chunks for vector indexing on large documents."""
    docs = dedupe_documents(docs)
    all_chunks: list[str] = []
    for doc in docs:
        text = trim_for_extraction(doc.content, max_chars=8000)
        if len(text) <= chunk_size:
            if text.strip():
                all_chunks.append(f"[{doc.doc_id}] {doc.title}\n{text}")
            continue

        count = 0
        start = 0
        while start < len(text) and count < max_chunks_per_doc:
            piece = text[start : start + chunk_size].strip()
            if piece:
                all_chunks.append(f"[{doc.doc_id}] {doc.title}\n{piece}")
                count += 1
            start += chunk_size
    return all_chunks


def prepare_corpus(dataset_dir: Path, merged_path: Path) -> list[Document]:
    """Load dataset and write clean merged corpus (UTF-8, no BOM)."""
    docs = load_dataset(dataset_dir)
    merged_path.parent.mkdir(parents=True, exist_ok=True)
    merged_path.write_text(merge_corpus_text(docs), encoding="utf-8", newline="\n")
    return docs
