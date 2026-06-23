"""Knowledge graph construction with NetworkX and optional Neo4j."""

from __future__ import annotations

import re

import networkx as nx

from src.entity_extraction import normalize_entity


def build_networkx_graph(triples: list[tuple[str, str, str]]) -> nx.DiGraph:
    """Build directed knowledge graph from triples."""
    graph = nx.DiGraph()
    for subject, relation, obj in triples:
        s = normalize_entity(subject)
        o = normalize_entity(obj)
        graph.add_node(s, label=s, type="entity")
        graph.add_node(o, label=o, type="entity")
        graph.add_edge(s, o, relation=relation)
    return graph


def find_seed_nodes(graph: nx.DiGraph, question: str, entities: list[str]) -> list[str]:
    """Match question entities/keywords to graph nodes."""
    matched: list[str] = []
    node_list = list(graph.nodes)

    for entity in entities:
        el = normalize_entity(entity).lower()
        for node in node_list:
            nl = node.lower()
            if el in nl or nl in el:
                matched.append(node)

    if not matched:
        keywords = re.findall(r"\b[A-Za-z][A-Za-z0-9&.'-]{2,}\b", question)
        stop = {
            "what", "which", "when", "where", "does", "the", "and", "for", "with",
            "from", "that", "this", "have", "were", "was", "are", "how", "many",
            "much", "year", "according", "will", "over", "into", "about",
        }
        for word in keywords:
            wl = word.lower()
            if wl in stop or len(wl) < 4:
                continue
            for node in node_list:
                if wl in node.lower():
                    matched.append(node)

    seen: set[str] = set()
    unique: list[str] = []
    for n in matched:
        if n not in seen:
            seen.add(n)
            unique.append(n)
    return unique[:8]


def rank_and_limit_triples(triples: list[tuple[str, str, str]], question: str, limit: int = 35) -> list[tuple[str, str, str]]:
    """Keep the most question-relevant triples to avoid context overload."""
    if len(triples) <= limit:
        return triples
    qwords = {w.lower() for w in re.findall(r"\b[A-Za-z0-9%$.,]+\b", question) if len(w) > 2}
    scored = []
    for s, r, o in triples:
        text = f"{s} {r} {o}".lower()
        score = sum(1 for w in qwords if w in text)
        scored.append((score, s, r, o))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, r, o) for _, s, r, o in scored[:limit]]


def get_neighbors_bfs(
    graph: nx.DiGraph,
    start_nodes: list[str],
    max_hops: int = 2,
    question: str = "",
    entities: list[str] | None = None,
) -> dict:
    """BFS traversal from seed nodes within max_hops."""
    if entities is not None and question:
        matched_starts = find_seed_nodes(graph, question, entities)
    else:
        matched_starts = []
        node_list = list(graph.nodes)
        for node in node_list:
            for start in [normalize_entity(n) for n in start_nodes]:
                sl = start.lower()
                if sl in node.lower() or node.lower() in sl:
                    matched_starts.append(node)
                    break

    if not matched_starts:
        return {"nodes": [], "edges": [], "triples": []}

    visited: set[str] = set()
    edges_found: list[tuple[str, str, str]] = []
    queue = [(n, 0) for n in matched_starts]

    while queue:
        node, depth = queue.pop(0)
        if node in visited:
            continue
        visited.add(node)
        if depth >= max_hops:
            continue

        for _, neighbor, data in graph.out_edges(node, data=True):
            edges_found.append((node, data.get("relation", "RELATED_TO"), neighbor))
            if neighbor not in visited:
                queue.append((neighbor, depth + 1))

        for predecessor, _, data in graph.in_edges(node, data=True):
            edges_found.append((predecessor, data.get("relation", "RELATED_TO"), node))
            if predecessor not in visited:
                queue.append((predecessor, depth + 1))

    triples = list(dict.fromkeys(edges_found))
    if question:
        triples = rank_and_limit_triples(triples, question)
    return {"nodes": list(visited), "edges": triples, "triples": triples}


def textualize_subgraph(subgraph: dict) -> str:
    """Convert graph neighborhood to natural language context."""
    lines = [f"({s}, {r}, {o})" for s, r, o in subgraph.get("triples", [])]
    if not lines:
        return "No related information found in the knowledge graph."
    return "Knowledge graph relations:\n" + "\n".join(lines)


def push_to_neo4j(triples: list[tuple[str, str, str]], uri: str, user: str, password: str) -> int:
    """Push triples to Neo4j database. Returns number of relationships created."""
    try:
        from neo4j import GraphDatabase
    except ImportError as e:
        raise ImportError("Install neo4j: pip install neo4j") from e

    driver = GraphDatabase.driver(uri, auth=(user, password))
    count = 0

    def create_triple(tx, subject, relation, obj):
        query = """
        MERGE (s:Entity {name: $subject})
        MERGE (o:Entity {name: $object})
        MERGE (s)-[r:RELATION {type: $relation}]->(o)
        """
        tx.run(query, subject=subject, relation=relation, object=obj)

    with driver.session() as session:
        for s, r, o in triples:
            session.execute_write(create_triple, s, r, o)
            count += 1

    driver.close()
    return count


def graph_stats(graph: nx.DiGraph) -> dict:
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "density": round(nx.density(graph), 4),
    }
