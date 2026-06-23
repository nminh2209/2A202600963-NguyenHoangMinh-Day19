"""Knowledge graph construction with NetworkX and optional Neo4j."""

from __future__ import annotations

import re

import networkx as nx

from src.entity_extraction import canonicalize_entity, normalize_entity


def build_networkx_graph(triples: list[tuple[str, str, str]]) -> nx.DiGraph:
    """Build directed knowledge graph from triples."""
    graph = nx.DiGraph()
    for subject, relation, obj in triples:
        s = canonicalize_entity(subject)
        o = canonicalize_entity(obj)
        graph.add_node(s, label=s, type="entity")
        graph.add_node(o, label=o, type="entity")
        graph.add_edge(s, o, relation=relation)
    return graph


ENTITY_QUERY_ALIASES = {
    "bnef": "BloombergNEF",
    "bloombergnef": "BloombergNEF",
    "gm": "General Motors",
    "icct": "ICCT",
    "uaw": "UAW",
    "j.d. power": "J.D. Power",
    "jd power": "J.D. Power",
    "biden": "Biden administration",
    "zev": "ZEV states",
    "metro": "metropolitan areas",
    "metropolitan": "metropolitan areas",
    "author": "Colin McKerracher",
    "mckerracher": "Colin McKerracher",
    "oil": "DISPLACE_OIL",
    "charging": "PUBLIC_CHARGERS",
    "incentive": "CONSUMER_INCENTIVES",
    "incentives": "CONSUMER_INCENTIVES",
    "strike": "STRIKE_AGAINST",
    "tax credit": "TAX_CREDIT",
}


def expand_query_terms(question: str) -> set[str]:
    """Expand question tokens for better triple matching."""
    stop = {
        "what", "which", "when", "where", "does", "the", "and", "for", "with",
        "from", "that", "this", "have", "were", "was", "are", "how", "many",
        "much", "year", "according", "will", "over", "into", "about", "who",
        "per", "than", "more", "most", "recently", "related", "their", "does",
        "did", "been", "being", "would", "could", "should", "very", "likely",
    }
    terms: set[str] = set()
    qlow = question.lower()
    for alias, canonical in ENTITY_QUERY_ALIASES.items():
        if alias in qlow:
            terms.add(canonical.lower())
            terms.add(alias)

    for word in re.findall(r"\b[A-Za-z0-9%$.,]+\b", question):
        wl = word.lower().strip(".,$%")
        if len(wl) > 2 and wl not in stop:
            terms.add(wl)
    return terms


def find_seed_nodes(graph: nx.DiGraph, question: str, entities: list[str]) -> list[str]:
    """Match question entities/keywords to graph nodes."""
    matched: list[str] = []
    node_list = list(graph.nodes)

    search_terms = list(entities)
    for word in re.findall(r"\b[A-Za-z][A-Za-z0-9&.'-]{2,}\b", question):
        wl = word.lower()
        if wl in ENTITY_QUERY_ALIASES:
            search_terms.append(ENTITY_QUERY_ALIASES[wl])
        search_terms.append(word)

    for entity in search_terms:
        el = canonicalize_entity(entity).lower()
        for node in node_list:
            nl = node.lower()
            if el in nl or nl in el or (len(el) > 3 and el[:4] in nl):
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


def search_triples_by_question(
    graph: nx.DiGraph,
    question: str,
    limit: int = 50,
) -> list[tuple[str, str, str]]:
    """Keyword search across all graph edges when BFS misses relevant facts."""
    qwords = expand_query_terms(question)

    scored: list[tuple[int, tuple[str, str, str]]] = []
    for s, o, data in graph.edges(data=True):
        r = data.get("relation", "RELATED_TO")
        text = f"{s} {r} {o}".lower()
        score = 0
        for w in qwords:
            if w in text:
                score += 2
            elif len(w) > 4 and any(w in tok for tok in text.split()):
                score += 1
        if score > 0:
            scored.append((score, (s, r, o)))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored[:limit]]


def find_path_triples(
    graph: nx.DiGraph,
    seeds: list[str],
    max_hops: int = 4,
    max_paths: int = 12,
) -> list[tuple[str, str, str]]:
    """Collect triples along shortest paths between seed entities (multi-hop)."""
    if len(seeds) < 2:
        return []

    undirected = graph.to_undirected()
    path_triples: list[tuple[str, str, str]] = []
    seen_edges: set[tuple[str, str, str]] = set()

    for i, a in enumerate(seeds[:6]):
        for b in seeds[i + 1 : 6]:
            if a not in undirected or b not in undirected:
                continue
            try:
                path = nx.shortest_path(undirected, a, b)
            except nx.NetworkXNoPath:
                continue
            for u, v in zip(path, path[1:]):
                if graph.has_edge(u, v):
                    r = graph[u][v].get("relation", "RELATED_TO")
                    t = (u, r, v)
                elif graph.has_edge(v, u):
                    r = graph[v][u].get("relation", "RELATED_TO")
                    t = (v, r, u)
                else:
                    continue
                if t not in seen_edges:
                    seen_edges.add(t)
                    path_triples.append(t)
            if len(path_triples) >= max_paths * 3:
                break

    return path_triples


def merge_triple_lists(*lists: list[tuple[str, str, str]], limit: int = 60) -> list[tuple[str, str, str]]:
    """Deduplicate and preserve order from multiple retrieval sources."""
    seen: set[tuple[str, str, str]] = set()
    merged: list[tuple[str, str, str]] = []
    for triples in lists:
        for t in triples:
            if t not in seen:
                seen.add(t)
                merged.append(t)
    return merged[:limit]


def rank_and_limit_triples(triples: list[tuple[str, str, str]], question: str, limit: int = 55) -> list[tuple[str, str, str]]:
    """Keep the most question-relevant triples to avoid context overload."""
    if len(triples) <= limit:
        return triples
    qwords = expand_query_terms(question)
    scored = []
    for s, r, o in triples:
        text = f"{s} {r} {o}".lower()
        score = sum(1 for w in qwords if w in text)
        scored.append((score, s, r, o))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(s, r, o) for _, s, r, o in scored[:limit]]


def retrieve_graph_context(
    graph: nx.DiGraph,
    question: str,
    entities: list[str],
    max_hops: int = 4,
) -> dict:
    """Graph-first retrieval: keyword search + BFS + multi-hop paths."""
    matched_starts = find_seed_nodes(graph, question, entities)
    keyword_triples = search_triples_by_question(graph, question, limit=50)
    path_triples = find_path_triples(graph, matched_starts, max_hops=max_hops) if matched_starts else []

    if not matched_starts:
        triples = rank_and_limit_triples(keyword_triples, question, limit=55)
        nodes = list({n for s, _, o in triples for n in (s, o)})
        return {"nodes": nodes, "edges": triples, "triples": triples, "seeds": []}

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

    bfs_triples = list(dict.fromkeys(edges_found))
    # Keyword + path triples first so graph-specific facts are never ranked away
    triples = merge_triple_lists(keyword_triples, path_triples, bfs_triples, limit=60)
    triples = rank_and_limit_triples(triples, question, limit=55)
    return {"nodes": list(visited), "edges": triples, "triples": triples, "seeds": matched_starts}


def get_neighbors_bfs(
    graph: nx.DiGraph,
    start_nodes: list[str],
    max_hops: int = 4,
    question: str = "",
    entities: list[str] | None = None,
) -> dict:
    """BFS traversal from seed nodes, merged with keyword triple search."""
    if entities is not None and question:
        return retrieve_graph_context(graph, question, entities, max_hops=max_hops)

    matched_starts: list[str] = []
    node_list = list(graph.nodes)
    for node in node_list:
        for start in [normalize_entity(n) for n in start_nodes]:
            sl = start.lower()
            if sl in node.lower() or node.lower() in sl:
                matched_starts.append(node)
                break

    keyword_triples = search_triples_by_question(graph, question) if question else []
    if not matched_starts:
        triples = rank_and_limit_triples(keyword_triples, question) if question else keyword_triples
        nodes = list({n for s, _, o in triples for n in (s, o)})
        return {"nodes": nodes, "edges": triples, "triples": triples}

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

    bfs_triples = list(dict.fromkeys(edges_found))
    triples = merge_triple_lists(keyword_triples, bfs_triples, limit=55)
    if question:
        triples = rank_and_limit_triples(triples, question, limit=55)
    return {"nodes": list(visited), "edges": triples, "triples": triples}


def textualize_subgraph(subgraph: dict) -> str:
    """Convert graph neighborhood to natural language context."""
    lines = []
    for i, (s, r, o) in enumerate(subgraph.get("triples", []), 1):
        rel = r.replace("_", " ").title()
        lines.append(f"FACT {i}: {s} --[{rel}]--> {o}")
    if not lines:
        return "No related information found in the knowledge graph."
    return "Knowledge graph facts (use these to answer):\n" + "\n".join(lines)


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
