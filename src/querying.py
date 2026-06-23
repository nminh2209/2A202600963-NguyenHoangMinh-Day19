"""GraphRAG query pipeline with entity extraction and multi-hop graph retrieval."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass

import networkx as nx
from openai import OpenAI

from src.config import LLM_MODEL, get_openai_api_key
from src.graph_construction import get_neighbors_bfs, textualize_subgraph


ENTITY_EXTRACTION_PROMPT = """Extract key entities (companies, people, organizations, metrics) from the question.
Return JSON: {{"entities": ["entity1", "entity2"]}}

Question: {question}
"""

REFUSAL_PATTERN = re.compile(
    r"insufficient information|cannot (answer|provide)|no (related )?information|not (enough|provided)|"
    r"triples do not|do not provide|do not contain|does not provide|does not contain|"
    r"i cannot|unable to|not directly provided|not specified",
    re.I,
)

GRAPH_ANSWER_SYSTEM = (
    "You are a GraphRAG expert. Answer ONLY from the numbered FACT lines in the knowledge graph. "
    "Each FACT shows: Entity --[Relation]--> Value. "
    "For multi-hop questions, chain facts across entities (e.g. BNEF -> author, ZEV states -> 5% vs 1.3%). "
    "You MUST give a direct, complete answer when any FACT is relevant. "
    "Never say facts are missing if a FACT line contains the answer. "
    "Include specific names, numbers, and years from the facts. Answer in English."
)


@dataclass
class QueryResult:
    question: str
    answer: str
    context: str
    entities: list[str]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_sec: float = 0.0


def extract_entities_from_question(question: str, client: OpenAI | None = None) -> tuple[list[str], int, int]:
    """Extract key entities from user question."""
    api_key = get_openai_api_key()
    known = [
        "Tesla", "General Motors", "GM", "Ford", "Stellantis", "BloombergNEF", "BNEF",
        "ICCT", "China", "Biden", "J.D. Power", "UAW", "ZEV", "Colin McKerracher",
        "Elizabeth Krear", "Motor Intelligence", "Bloomberg", "metropolitan areas",
    ]
    found = [k for k in known if k.lower() in question.lower()]

    if not api_key:
        candidates = re.findall(r"\b[A-Z][a-zA-Z]+\b", question)
        return (found or candidates[:3]), 0, 0

    client = client or OpenAI(api_key=api_key)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": "Extract entities from questions. Return JSON only."},
            {"role": "user", "content": ENTITY_EXTRACTION_PROMPT.format(question=question)},
        ],
        temperature=0,
        response_format={"type": "json_object"},
    )
    usage = response.usage
    content = response.choices[0].message.content or "{}"
    try:
        entities = json.loads(content).get("entities", [])
    except json.JSONDecodeError:
        entities = []
    merged = list(dict.fromkeys(found + entities))
    return merged, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


def is_refusal(answer: str) -> bool:
    return bool(REFUSAL_PATTERN.search(answer))


def compose_direct_answer(question: str, triples: list[tuple[str, str, str]]) -> str:
    """Deterministic fallback: stitch top graph facts into an answer."""
    if not triples:
        return "No related information found in the knowledge graph."

    parts = []
    for s, r, o in triples[:10]:
        rel = r.replace("_", " ").lower()
        parts.append(f"{s} {rel} {o}")

    joined = "; ".join(parts)
    if "?" in question:
        return f"From the knowledge graph: {joined}."
    return joined


def _llm_answer_from_graph(
    client: OpenAI,
    question: str,
    context: str,
    strict: bool = False,
) -> tuple[str, int, int]:
    system = GRAPH_ANSWER_SYSTEM
    if strict:
        system += " IMPORTANT: The facts listed ARE sufficient. Compose the answer now."

    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"{context}\n\nQuestion: {question}"},
        ],
        temperature=0,
    )
    usage = response.usage
    answer = response.choices[0].message.content or ""
    return (
        answer,
        usage.prompt_tokens if usage else 0,
        usage.completion_tokens if usage else 0,
    )


def answer_with_graph(
    question: str,
    graph: nx.DiGraph,
    max_hops: int = 4,
    client: OpenAI | None = None,
) -> QueryResult:
    """GraphRAG: entities -> multi-hop retrieval -> LLM with direct-fact fallback."""
    start = time.perf_counter()
    entities, ent_prompt, ent_completion = extract_entities_from_question(question, client)

    subgraph = get_neighbors_bfs(
        graph, entities, max_hops=max_hops, question=question, entities=entities
    )
    triples = subgraph.get("triples", [])
    context = textualize_subgraph(subgraph)

    api_key = get_openai_api_key()
    if not api_key:
        answer = compose_direct_answer(question, triples)
        return QueryResult(
            question=question, answer=answer, context=context, entities=entities,
            latency_sec=time.perf_counter() - start,
        )

    if not triples:
        answer = "No related information found in the knowledge graph."
        return QueryResult(
            question=question, answer=answer, context=context, entities=entities,
            latency_sec=time.perf_counter() - start,
        )

    client = client or OpenAI(api_key=api_key)
    answer, p_tok, c_tok = _llm_answer_from_graph(client, question, context)

    if is_refusal(answer):
        answer, p2, c2 = _llm_answer_from_graph(client, question, context, strict=True)
        p_tok += p2
        c_tok += c2

    if is_refusal(answer):
        answer = compose_direct_answer(question, triples)

    return QueryResult(
        question=question,
        answer=answer,
        context=context,
        entities=entities,
        prompt_tokens=p_tok + ent_prompt,
        completion_tokens=c_tok + ent_completion,
        latency_sec=time.perf_counter() - start,
    )
