"""Knowledge graph construction with NetworkX and optional Neo4j."""

from __future__ import annotations

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


def get_neighbors_bfs(graph: nx.DiGraph, start_nodes: list[str], max_hops: int = 2) -> dict:
    """
    BFS traversal from start nodes within max_hops.
    Returns nodes, edges, and triples discovered.
    """
    start_nodes = [normalize_entity(n) for n in start_nodes]
    matched_starts = []
    node_list = list(graph.nodes)
    for node in node_list:
        nl = node.lower()
        for start in start_nodes:
            sl = start.lower()
            if sl in nl or nl in sl:
                matched_starts.append(node)
                break

    # Fallback: match any corpus entity mentioned in question text joined as string
    if not matched_starts and start_nodes:
        query = " ".join(start_nodes).lower()
        for node in node_list:
            if node.lower() in query or any(part in node.lower() for part in query.split() if len(part) > 3):
                matched_starts.append(node)

    if not matched_starts:
        return {"nodes": [], "edges": [], "triples": []}

    visited = set()
    edges_found = []
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
    nodes = list(visited)
    return {"nodes": nodes, "edges": triples, "triples": triples}


def textualize_subgraph(subgraph: dict) -> str:
    """Convert graph neighborhood to natural language context."""
    lines = []
    for s, r, o in subgraph.get("triples", []):
        lines.append(f"({s}, {r}, {o})")
    if not lines:
        return "Không tìm thấy thông tin liên quan trong đồ thị."
    return "Các quan hệ trong đồ thị tri thức:\n" + "\n".join(lines)


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
