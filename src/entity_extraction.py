"""Entity and relation extraction using LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from src.config import LLM_MODEL, get_openai_api_key
from src.demo_triples import DEMO_TRIPLES

EXTRACTION_PROMPT = """Bạn là hệ thống trích xuất tri thức. Đọc văn bản và trích xuất các bộ ba (subject, relation, object).

Quy tắc:
- subject và object là THỰC THỂ (người, công ty, sản phẩm, địa điểm, năm).
- relation là QUAN HỆ viết HOA bằng tiếng Anh (FOUNDED_BY, CEO_OF, ACQUIRED, FOUNDED_IN, ...).
- Mỗi câu có thể sinh nhiều triple.
- Không trích xuất thuộc tính mô tả chung làm node riêng nếu có thể gắn vào quan hệ.

Trả về JSON object với key "triples":
{{"triples": [{{"subject": "OpenAI", "relation": "FOUNDED_BY", "object": "Sam Altman"}}, ...]}}

Văn bản:
{text}
"""


@dataclass
class ExtractionResult:
    triples: list[tuple[str, str, str]]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    source: str = "llm"


def normalize_entity(name: str) -> str:
    """Normalize entity names for deduplication."""
    name = name.strip()
    name = re.sub(r"\s+", " ", name)
    return name


def deduplicate_triples(triples: list[tuple[str, str, str]]) -> list[tuple[str, str, str]]:
    """Remove duplicate triples after normalization."""
    seen: set[tuple[str, str, str]] = set()
    unique: list[tuple[str, str, str]] = []
    for s, r, o in triples:
        key = (normalize_entity(s), r.strip().upper(), normalize_entity(o))
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


def extract_triples_from_text(text: str, client: OpenAI | None = None) -> ExtractionResult:
    """Extract triples from a single text chunk using OpenAI."""
    client = client or OpenAI(api_key=get_openai_api_key())
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "You extract knowledge graph triples. Return valid JSON only."},
            {"role": "user", "content": EXTRACTION_PROMPT.format(text=text)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    usage = response.usage
    content = response.choices[0].message.content or "[]"

    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict):
            parsed = parsed.get("triples", parsed.get("data", []))
        triples = [
            (normalize_entity(t["subject"]), t["relation"].strip().upper(), normalize_entity(t["object"]))
            for t in parsed
            if "subject" in t and "relation" in t and "object" in t
        ]
    except (json.JSONDecodeError, KeyError, TypeError):
        triples = []

    triples = deduplicate_triples(triples)
    return ExtractionResult(
        triples=triples,
        prompt_tokens=usage.prompt_tokens if usage else 0,
        completion_tokens=usage.completion_tokens if usage else 0,
        total_tokens=usage.total_tokens if usage else 0,
        source="llm",
    )


def extract_triples_from_corpus(corpus_path, demo: bool = False) -> ExtractionResult:
    """Extract triples from full corpus file."""
    if demo or not get_openai_api_key():
        return ExtractionResult(triples=deduplicate_triples(DEMO_TRIPLES), source="demo")

    text = corpus_path.read_text(encoding="utf-8")
    paragraphs = [p.strip() for p in text.split("\n") if p.strip()]

    all_triples: list[tuple[str, str, str]] = []
    total_prompt = total_completion = 0

    client = OpenAI(api_key=get_openai_api_key())
    for para in paragraphs:
        result = extract_triples_from_text(para, client)
        all_triples.extend(result.triples)
        total_prompt += result.prompt_tokens
        total_completion += result.completion_tokens

    return ExtractionResult(
        triples=deduplicate_triples(all_triples),
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        total_tokens=total_prompt + total_completion,
        source="llm",
    )


def save_triples(triples: list[tuple[str, str, str]], path) -> None:
    data = [{"subject": s, "relation": r, "object": o} for s, r, o in triples]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_triples(path) -> list[tuple[str, str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(t["subject"], t["relation"], t["object"]) for t in data]
