#!/usr/bin/env python3
"""
LAB DAY 19: GraphRAG Pipeline - Tech Company Corpus
Run: python main.py --demo          # No API key needed
     python main.py                 # Full pipeline with OpenAI
     python main.py --neo4j         # Also push to Neo4j
     python main.py --eval-only     # Skip indexing, run evaluation
"""

from __future__ import annotations

import argparse
import json
import time

from src.config import (
    CORPUS_PATH,
    COST_REPORT_PATH,
    EVAL_RESULTS_PATH,
    GRAPH_IMAGE_PATH,
    NEO4J_PASSWORD,
    NEO4J_URI,
    NEO4J_USER,
    TRIPLES_PATH,
    get_openai_api_key,
)
from src.pipeline import run_full_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Lab 19 GraphRAG Pipeline")
    parser.add_argument("--demo", action="store_true", help="Use pre-extracted triples (no API key)")
    parser.add_argument("--neo4j", action="store_true", help="Push graph to Neo4j")
    parser.add_argument("--eval-only", action="store_true", help="Skip indexing, use saved triples")
    parser.add_argument("--question", type=str, help="Ask a single question")
    return parser.parse_args()


def main():
    args = parse_args()

    if args.question:
        from src.entity_extraction import load_triples
        from src.flat_rag import FlatRAG
        from src.graph_construction import build_networkx_graph
        from src.querying import answer_with_graph

        triples = load_triples(TRIPLES_PATH) if TRIPLES_PATH.exists() else []
        if not triples:
            print("No triples found. Run pipeline first: python main.py")
            return
        graph = build_networkx_graph(triples)
        flat_rag = FlatRAG(CORPUS_PATH)
        flat_rag.index()
        print(f"\n[Query] {args.question}")
        print(f"\nFlat RAG:  {flat_rag.answer(args.question).answer}")
        print(f"GraphRAG:  {answer_with_graph(args.question, graph).answer}")
        return

    demo = args.demo or not get_openai_api_key()
    print("=" * 60)
    print("LAB 19: GraphRAG - Tech Company Corpus")
    print("=" * 60)
    print(f"[Mode] {'DEMO' if demo else 'FULL'}")

    state = run_full_pipeline(
        demo=demo,
        force_reindex=not args.eval_only,
        run_eval=True,
    )
    print(f"\n[Step 1] {len(state.triples)} triples | {state.indexing_sec:.2f}s | {state.indexing_tokens} tokens")
    print(f"[Step 2] {state.stats['num_nodes']} nodes, {state.stats['num_edges']} edges")
    print(f"[Step 2c] Graph image: {GRAPH_IMAGE_PATH}")

    if args.neo4j and NEO4J_PASSWORD:
        from src.graph_construction import push_to_neo4j
        count = push_to_neo4j(state.triples, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD)
        print(f"[Neo4j] Pushed {count} relationships")

    if state.eval_df is not None:
        from src.evaluation import print_summary
        print_summary(state.eval_df)
        print(f"\nEvaluation CSV: {EVAL_RESULTS_PATH}")
        print(f"Cost report: {COST_REPORT_PATH}")


if __name__ == "__main__":
    main()
