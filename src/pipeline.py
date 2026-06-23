"""Shared GraphRAG pipeline for CLI and Streamlit."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import networkx as nx

from src.config import (
    CORPUS_PATH,
    COST_REPORT_PATH,
    DATASET_DIR,
    EVAL_RESULTS_PATH,
    GRAPH_IMAGE_PATH,
    TRIPLES_PATH,
    get_openai_api_key,
)
from src.corpus import load_dataset, prepare_corpus
from src.entity_extraction import extract_triples_from_corpus, load_triples, save_triples
from src.evaluation import compute_summary, run_evaluation
from src.fact_triples import enrich_triples
from src.flat_rag import FlatRAG
from src.graph_construction import build_networkx_graph, graph_stats
from src.visualize import visualize_graph


@dataclass
class PipelineState:
    triples: list[tuple[str, str, str]] = field(default_factory=list)
    graph: nx.DiGraph | None = None
    flat_rag: FlatRAG | None = None
    stats: dict = field(default_factory=dict)
    mode: str = "demo"
    indexing_sec: float = 0.0
    indexing_tokens: int = 0
    construction_sec: float = 0.0
    eval_df: object | None = None
    cost_report: dict = field(default_factory=dict)


def run_full_pipeline(
    demo: bool = False,
    force_reindex: bool = True,
    run_eval: bool = True,
    progress_callback=None,
) -> PipelineState:
    """Build graph, index Flat RAG, optionally run evaluation."""
    state = PipelineState()
    state.mode = "demo" if (demo or not get_openai_api_key()) else "full"

    def _progress(msg: str, pct: float):
        if progress_callback:
            progress_callback(msg, pct)

    _progress("Preparing dataset (70 documents)...", 0.05)
    prepare_corpus(DATASET_DIR, CORPUS_PATH)

    _progress("Extracting entities & relations...", 0.1)
    if not force_reindex and TRIPLES_PATH.exists():
        state.triples = load_triples(TRIPLES_PATH)
    else:
        t0 = time.perf_counter()

        def ext_progress(msg, pct):
            _progress(msg, 0.1 + pct * 0.2)

        result = extract_triples_from_corpus(demo=state.mode == "demo", progress_callback=ext_progress)
        state.triples = result.triples
        state.indexing_tokens = result.total_tokens
        state.indexing_sec = time.perf_counter() - t0

    state.triples = enrich_triples(state.triples, CORPUS_PATH)
    save_triples(state.triples, TRIPLES_PATH)

    _progress("Building knowledge graph...", 0.35)
    t0 = time.perf_counter()
    state.graph = build_networkx_graph(state.triples)
    state.stats = graph_stats(state.graph)
    state.construction_sec = time.perf_counter() - t0

    _progress("Generating graph visualization...", 0.5)
    visualize_graph(state.graph, GRAPH_IMAGE_PATH)

    _progress("Indexing Flat RAG (ChromaDB)...", 0.6)
    state.flat_rag = FlatRAG(dataset_dir=DATASET_DIR)
    state.flat_rag.index(force_rebuild=force_reindex)

    if run_eval:
        _progress("Running benchmark evaluation (20 questions)...", 0.7)
        t0 = time.perf_counter()
        state.eval_df = run_evaluation(state.graph, state.flat_rag)
        eval_sec = time.perf_counter() - t0
        summary = compute_summary(state.eval_df)
        state.cost_report = {
            "dataset": {
                "documents_total": 70,
                "documents_usable": len(load_dataset(DATASET_DIR)) if DATASET_DIR.exists() else 68,
                "source": "dataset/dataset",
            },
            "indexing": {"time_sec": round(state.indexing_sec, 2), "tokens": state.indexing_tokens},
            "construction": {"time_sec": round(state.construction_sec, 4)},
            "evaluation": {
                "time_sec": round(eval_sec, 2),
                "flat_rag_total_tokens": int(state.eval_df["flat_tokens"].sum()),
                "graph_rag_total_tokens": int(state.eval_df["graph_tokens"].sum()),
                **summary,
            },
            "graph_stats": state.stats,
            "mode": state.mode,
        }
        COST_REPORT_PATH.write_text(json.dumps(state.cost_report, indent=2), encoding="utf-8")

    _progress("Done!", 1.0)
    return state
