"""Entity and relation extraction using LLM."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from openai import OpenAI

from src.config import DATASET_DIR, LLM_MODEL, get_openai_api_key
from src.corpus import benchmark_boost_chunks, extraction_chunks, load_dataset, prepare_corpus
from src.demo_triples import DEMO_TRIPLES

EXTRACTION_PROMPT = """Extract knowledge graph triples from US electric vehicle sector text.

CRITICAL: Extract ALL numeric facts, percentages, years, dollar amounts, and named relationships.

Relation examples: MARKET_SHARE, LEAD_AUTHOR_OF, PUBLISHED_BY, PROJECTS_PEAK_YEAR, GROWTH_RATE,
DISPLACES_OIL, MANUFACTURED_IN, TAX_CREDIT, CHARGER_TARGET, SURVEY_RESULT, STRIKE_AGAINST,
PUBLIC_CHARGERS_PER_MILLION, MARKET_SIZE, FOUNDED_BY, CEO_OF, VP_OF.

Use canonical names: BloombergNEF (not BNEF), General Motors (not GM), ICCT, J.D. Power.

Return JSON:
{{"triples": [{{"subject": "Tesla", "relation": "MARKET_SHARE", "object": "51%"}}, ...]}}

Text:
{text}
"""

# Map aliases to canonical entity names for graph linking
ENTITY_CANONICAL = {
    "bnef": "BloombergNEF",
    "bloomberg nef": "BloombergNEF",
    "bloombergnef": "BloombergNEF",
    "gm": "General Motors",
    "international council on clean transportation": "ICCT",
    "the icct": "ICCT",
    "j.d. power": "J.D. Power",
    "jd power": "J.D. Power",
    "uaw": "UAW",
    "united auto workers": "UAW",
    "biden administration": "Biden administration",
    "zero-emission vehicle states": "ZEV states",
    "zev states": "ZEV states",
}


def canonicalize_entity(name: str | int | float) -> str:
    """Normalize and map aliases to canonical entity names."""
    n = normalize_entity(name)
    return ENTITY_CANONICAL.get(n.lower(), n)


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
        key = (canonicalize_entity(s), r.strip().upper(), canonicalize_entity(o))
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
            (canonicalize_entity(t["subject"]), str(t["relation"]).strip().upper(), canonicalize_entity(t["object"]))
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

    from src.config import CORPUS_PATH, TRIPLES_PATH

    docs = prepare_corpus(DATASET_DIR, CORPUS_PATH)
    main_chunks = extraction_chunks(docs)
    boost_chunks = benchmark_boost_chunks(docs)
    seen_main = set(main_chunks)
    chunks = main_chunks + [c for c in boost_chunks if c not in seen_main]

    all_triples: list[tuple[str, str, str]] = []
    total_prompt = total_completion = 0
    client = OpenAI(api_key=get_openai_api_key())

    for i, chunk in enumerate(chunks):
        if progress_callback:
            label = "fact" if "|fact]" in chunk[:40] else "doc"
            progress_callback(f"Extracting {label} {i + 1}/{len(chunks)}...", (i + 1) / len(chunks))
        result = extract_triples_from_text(chunk, client)
        all_triples.extend(result.triples)
        total_prompt += result.prompt_tokens
        total_completion += result.completion_tokens
        if (i + 1) % 20 == 0:
            save_triples(deduplicate_triples(all_triples), TRIPLES_PATH)

    return ExtractionResult(
        triples=deduplicate_triples(all_triples),
        prompt_tokens=total_prompt,
        completion_tokens=total_completion,
        total_tokens=total_prompt + total_completion,
        source="llm",
    )


def save_triples(triples: list[tuple[str, str, str]], path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = [{"subject": s, "relation": r, "object": o} for s, r, o in triples]
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_triples(path) -> list[tuple[str, str, str]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return [(t["subject"], t["relation"], t["object"]) for t in data]
