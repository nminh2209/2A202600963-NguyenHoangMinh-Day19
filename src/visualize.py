"""Visualize knowledge graph with Matplotlib."""

from __future__ import annotations

import matplotlib.pyplot as plt
import networkx as nx


def visualize_graph(graph: nx.DiGraph, output_path, max_labels: int = 40) -> None:
    """Save knowledge graph visualization to PNG."""
    plt.figure(figsize=(16, 12))

    if graph.number_of_nodes() > max_labels:
        degrees = dict(graph.degree())
        top_nodes = sorted(degrees, key=degrees.get, reverse=True)[:max_labels]
        subgraph = graph.subgraph(top_nodes).copy()
    else:
        subgraph = graph

    pos = nx.spring_layout(subgraph, k=1.5, seed=42)
    nx.draw_networkx_nodes(subgraph, pos, node_color="#4A90D9", node_size=800, alpha=0.9)
    nx.draw_networkx_labels(subgraph, pos, font_size=7, font_weight="bold")

    edge_labels = {(u, v): d.get("relation", "")[:12] for u, v, d in subgraph.edges(data=True)}
    nx.draw_networkx_edges(subgraph, pos, edge_color="#888888", arrows=True, arrowsize=12, alpha=0.6)
    nx.draw_networkx_edge_labels(subgraph, pos, edge_labels=edge_labels, font_size=5)

    plt.title("Tech Company Knowledge Graph (NetworkX)", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
