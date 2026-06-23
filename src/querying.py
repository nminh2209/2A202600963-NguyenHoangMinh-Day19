"""GraphRAG query pipeline with entity extraction and BFS traversal."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import networkx as nx
from openai import OpenAI

from src.config import LLM_MODEL, get_openai_api_key
from src.graph_construction import get_neighbors_bfs, textualize_subgraph


ENTITY_EXTRACTION_PROMPT = """Trích xuất các thực thể chính (tên công ty, người, sản phẩm) từ câu hỏi.
Trả về JSON: {{"entities": ["entity1", "entity2"]}}

Câu hỏi: {question}
"""


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
    if not api_key:
        # Simple fallback: capitalize words that look like company/person names
        import re
        candidates = re.findall(r"\b[A-Z][a-zA-Z]+\b", question)
        known = [
            "Tesla", "General Motors", "GM", "Ford", "Stellantis", "BloombergNEF", "BNEF",
            "ICCT", "China", "Biden", "J.D. Power", "UAW", "EV", "ZEV", "Colin McKerracher",
            "Elizabeth Krear", "Motor Intelligence", "Bloomberg",
        ]
        found = [k for k in known if k.lower() in question.lower()]
        return (found or candidates[:3]), 0, 0

    client = client or OpenAI(api_key=get_openai_api_key())
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
    return entities, usage.prompt_tokens if usage else 0, usage.completion_tokens if usage else 0


def answer_with_graph(
    question: str,
    graph: nx.DiGraph,
    max_hops: int = 3,
    client: OpenAI | None = None,
) -> QueryResult:
    """GraphRAG: extract entities -> BFS 2-hop -> textualize -> LLM answer."""
    start = time.perf_counter()
    entities, ent_prompt, ent_completion = extract_entities_from_question(question, client)

    subgraph = get_neighbors_bfs(
        graph, entities, max_hops=2, question=question, entities=entities
    )
    context = textualize_subgraph(subgraph)

    api_key = get_openai_api_key()
    if not api_key:
        answer = f"[Demo] Dựa trên đồ thị: {context[:500]}"
        return QueryResult(
            question=question,
            answer=answer,
            context=context,
            entities=entities,
            latency_sec=time.perf_counter() - start,
        )

    client = client or OpenAI(api_key=get_openai_api_key())
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "Answer the question using the knowledge graph context provided. "
                    "Reason over the (subject, relation, object) triples. "
                    "Answer with available facts; only say insufficient information if the graph is unrelated. "
                    "Be concise. Answer in English."
                ),
            },
            {
                "role": "user",
                "content": f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}",
            },
        ],
        temperature=0,
    )
    usage = response.usage
    answer = response.choices[0].message.content or ""

    return QueryResult(
        question=question,
        answer=answer,
        context=context,
        entities=entities,
        prompt_tokens=(usage.prompt_tokens if usage else 0) + ent_prompt,
        completion_tokens=(usage.completion_tokens if usage else 0) + ent_completion,
        latency_sec=time.perf_counter() - start,
    )
