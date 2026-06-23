"""Entity and relation extraction using LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from openai import OpenAI

from src.config import DATASET_DIR, LLM_MODEL, get_openai_api_key
from src.corpus import extraction_chunks, load_dataset, prepare_corpus
from src.demo_triples import DEMO_TRIPLES

EXTRACTION_PROMPT = """You are a knowledge extraction system. Read the text about the US electric vehicle sector and extract triples (subject, relation, object).

Rules:
- subject and object are ENTITIES (companies, people, organizations, metrics, years, locations).
- relation is an UPPERCASE English relation (MARKET_SHARE, PUBLISHED_BY, LOCATED_IN, GROWTH_RATE, ...).
- Extract numeric facts as objects when relevant (e.g. Tesla, MARKET_SHARE, 50%).

Return JSON:
{{"triples": [{{"subject": "Tesla", "relation": "MARKET_LEADER", "object": "US EV market"}}, ...]}}

Text:
{text}
"""


@dataclass
class ExtractionResult:
    triples: list[tuple[str, str, str]]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    source: str = "llm"


def normalize_entity(name: str | int | float) -> str:
    """Normalize entity names for deduplication."""
    name = str(name).strip()
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
            (normalize_entity(t["subject"]), str(t["relation"]).strip().upper(), normalize_entity(t["object"]))
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


def extract_triples_from_corpus(corpus_path=None, demo: bool = False, progress_callback=None) -> ExtractionResult:
    """Extract triples from dataset folder (70 txt files) or merged corpus."""
    if demo or not get_openai_api_key():
        return ExtractionResult(triples=deduplicate_triples(DEMO_TRIPLES), source="demo")

    from src.config import CORPUS_PATH

    docs = prepare_corpus(DATASET_DIR, CORPUS_PATH)
    chunks = extraction_chunks(docs)

    all_triples: list[tuple[str, str, str]] = []
    total_prompt = total_completion = 0
    client = OpenAI(api_key=get_openai_api_key())

    for i, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(f"Extracting doc {i + 1}/{len(chunks)}...", (i + 1) / len(chunks))
        result = extract_triples_from_text(chunk, client)
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
