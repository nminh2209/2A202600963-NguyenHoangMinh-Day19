"""
Streamlit demo UI for Lab 19 GraphRAG.
Run: streamlit run streamlit_app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from src.config import (
    BENCHMARK_PATH,
    CORPUS_PATH,
    COST_REPORT_PATH,
    DATASET_DIR,
    EVAL_RESULTS_PATH,
    GRAPH_IMAGE_PATH,
    LLM_MODEL,
    TRIPLES_PATH,
    get_openai_api_key,
)
from src.corpus import load_dataset
from src.entity_extraction import load_triples
from src.evaluation import compute_summary, run_evaluation
from src.flat_rag import FlatRAG
from src.graph_construction import build_networkx_graph, get_neighbors_bfs, graph_stats, textualize_subgraph
from src.pipeline import run_full_pipeline
from src.querying import answer_with_graph

st.set_page_config(page_title="GraphRAG Lab 19", page_icon="🔗", layout="wide")

st.title("🔗 Lab 19: GraphRAG — US Electric Vehicle Dataset")
st.caption("68 clean docs | Flat RAG (ChromaDB) vs GraphRAG (NetworkX + multi-hop paths)")


def init_session():
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = None


def load_cached_pipeline():
    """Load graph from saved triples without full rebuild."""
    if not TRIPLES_PATH.exists():
        return None
    triples = load_triples(TRIPLES_PATH)
    graph = build_networkx_graph(triples)
    flat_rag = FlatRAG(dataset_dir=DATASET_DIR)
    flat_rag.index(force_rebuild=False)
    return {"triples": triples, "graph": graph, "flat_rag": flat_rag, "stats": graph_stats(graph)}


init_session()

# Sidebar
with st.sidebar:
    st.header("⚙️ Pipeline")
    api_key = get_openai_api_key()
    api_ok = bool(api_key)
    if api_ok:
        st.success(f"Real LLM mode (`{LLM_MODEL}`)")
        st.caption(f"Key: …{api_key[-8:]}")
    else:
        st.error("Demo mode — no API key in `.env`")

    if st.button("🔌 Test API connection", use_container_width=True):
        try:
            from openai import OpenAI
            r = OpenAI(api_key=api_key).chat.completions.create(
                model=LLM_MODEL,
                messages=[{"role": "user", "content": "Reply: OK"}],
                max_tokens=5,
            )
            st.success(f"API works! Tokens used: {r.usage.total_tokens}")
            st.caption("Check OpenAI Usage dashboard in 1-5 minutes.")
        except Exception as e:
            st.error(str(e))

    force_rebuild = st.checkbox("Force rebuild index", value=False)
    run_eval_on_build = st.checkbox("Run 20-question benchmark on build", value=True)

    if st.button("🚀 Build / Rebuild Pipeline", type="primary", use_container_width=True):
        progress = st.progress(0, text="Starting...")
        status = st.empty()

        def on_progress(msg, pct):
            progress.progress(pct, text=msg)
            status.caption(msg)

        with st.spinner("Running full pipeline..."):
            state = run_full_pipeline(
                demo=not api_ok,
                force_reindex=force_rebuild or not TRIPLES_PATH.exists(),
                run_eval=run_eval_on_build,
                progress_callback=on_progress,
            )
            st.session_state.pipeline = state
        progress.progress(1.0, text="Complete!")
        st.success(f"Pipeline ready — {len(state.triples)} triples, {state.stats.get('num_nodes', 0)} nodes")

    if st.button("📂 Load saved artifacts", use_container_width=True):
        cached = load_cached_pipeline()
        if cached:
            st.session_state.pipeline = type("S", (), cached)()
            st.success("Loaded from output/")
        else:
            st.error("No saved triples found. Build pipeline first.")

    st.divider()
    st.markdown("**Quick stats**")
    if TRIPLES_PATH.exists():
        st.metric("Saved triples", len(load_triples(TRIPLES_PATH)))
    if COST_REPORT_PATH.exists():
        cost = json.loads(COST_REPORT_PATH.read_text(encoding="utf-8"))
        st.metric("Mode", cost.get("mode", "—"))

# Resolve pipeline state
pipe = st.session_state.pipeline
if pipe is None:
    pipe = load_cached_pipeline()
    if pipe:
        st.session_state.pipeline = type("S", (), pipe)()

if pipe is None:
    st.info("👈 Click **Build / Rebuild Pipeline** in the sidebar to get started.")
    st.stop()

graph = pipe.graph if hasattr(pipe, "graph") else pipe["graph"]
flat_rag = pipe.flat_rag if hasattr(pipe, "flat_rag") else pipe["flat_rag"]
stats = pipe.stats if hasattr(pipe, "stats") else pipe["stats"]
triples = pipe.triples if hasattr(pipe, "triples") else pipe["triples"]

tab_query, tab_graph, tab_eval, tab_corpus = st.tabs(["💬 Query", "🕸️ Graph", "📊 Evaluation", "📄 Corpus"])

# --- Query tab ---
with tab_query:
    st.subheader("Compare Flat RAG vs GraphRAG")
    question = st.text_input(
        "Your question",
        value="Which company leads the US EV market with more than half of all EV sales?",
        placeholder="Nhập câu hỏi về tech companies...",
    )
    max_hops = st.slider("BFS max hops", 1, 3, 2)
    col1, col2 = st.columns(2)

    if st.button("Ask", type="primary"):
        with st.spinner("Querying..."):
            flat_res = flat_rag.answer(question)
            graph_res = answer_with_graph(question, graph, max_hops=max_hops)

        with col1:
            st.markdown("### Flat RAG")
            st.write(flat_res.answer)
            with st.expander("Retrieved context"):
                st.text(flat_res.context)
            st.caption(f"⏱ {flat_res.latency_sec:.2f}s | 🎫 {flat_res.prompt_tokens + flat_res.completion_tokens} tokens")

        with col2:
            st.markdown("### GraphRAG")
            st.write(graph_res.answer)
            with st.expander("Graph context (BFS)"):
                st.text(graph_res.context)
            st.caption(f"Entities: {graph_res.entities} | ⏱ {graph_res.latency_sec:.2f}s")

# --- Graph tab ---
with tab_graph:
    c1, c2, c3 = st.columns(3)
    c1.metric("Nodes", stats["num_nodes"])
    c2.metric("Edges", stats["num_edges"])
    c3.metric("Density", stats["density"])

    if GRAPH_IMAGE_PATH.exists():
        st.image(str(GRAPH_IMAGE_PATH), caption="Knowledge Graph (NetworkX + Matplotlib)")

    st.subheader("Explore subgraph (BFS)")
    entity = st.text_input("Start entity", value="Tesla")
    if entity:
        sub = get_neighbors_bfs(graph, [entity], max_hops=2)
        st.code(textualize_subgraph(sub), language=None)
        st.write(f"**{len(sub['triples'])}** relations found")

    with st.expander(f"All triples ({len(triples)})"):
        st.dataframe(
            pd.DataFrame(triples, columns=["subject", "relation", "object"]),
            use_container_width=True,
            height=300,
        )

# --- Evaluation tab ---
with tab_eval:
    st.subheader("Benchmark: 20 questions")

    col_run, col_load = st.columns(2)
    with col_run:
        if st.button("▶️ Run evaluation", use_container_width=True):
            prog = st.progress(0)

            def ev_progress(msg, pct):
                prog.progress(pct, text=msg)

            with st.spinner("Evaluating 20 questions (may take a few minutes)..."):
                df = run_evaluation(graph, flat_rag, progress_callback=ev_progress)
            st.session_state.eval_df = df
            prog.progress(1.0)

    eval_df = getattr(st.session_state, "eval_df", None)
    if eval_df is None and EVAL_RESULTS_PATH.exists():
        eval_df = pd.read_csv(EVAL_RESULTS_PATH)
    elif hasattr(pipe, "eval_df") and pipe.eval_df is not None:
        eval_df = pipe.eval_df

    if eval_df is not None and len(eval_df):
        summary = compute_summary(eval_df)
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Flat RAG accuracy", f"{summary['flat_accuracy_pct']}%")
        m2.metric("GraphRAG accuracy", f"{summary['graph_accuracy_pct']}%")
        m3.metric("Graph wins (Flat wrong)", summary["graph_wins_when_flat_wrong"])
        m4.metric("Multi-hop Graph acc.", f"{summary['multi_hop_graph_accuracy_pct']}%")

        st.markdown("#### Cases where GraphRAG fixes Flat RAG errors")
        wins = eval_df[eval_df["graph_wins"] == "Có"]
        if len(wins):
            for _, row in wins.iterrows():
                with st.expander(f"Q{row['id']}: {row['question']}"):
                    st.markdown(f"**Ground truth:** {row['ground_truth']}")
                    st.error(f"Flat RAG: {row['flat_rag_answer']}")
                    st.success(f"GraphRAG: {row['graph_rag_answer']}")
        else:
            st.info("No cases where GraphRAG won alone in this run.")

        st.markdown("#### Full results")
        display_cols = [
            "id", "question", "ground_truth", "flat_correct", "graph_correct",
            "graph_wins", "flat_latency", "graph_latency",
        ]
        st.dataframe(eval_df[display_cols], use_container_width=True, height=400)

        if COST_REPORT_PATH.exists():
            with st.expander("Cost analysis"):
                st.json(json.loads(COST_REPORT_PATH.read_text(encoding="utf-8")))

        csv = eval_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("Download CSV", csv, "evaluation_results.csv", "text/csv")
    else:
        st.info("Run evaluation to see benchmark results.")

# --- Corpus tab ---
with tab_corpus:
    st.subheader("US EV Dataset (70 documents)")
    st.metric("Documents", len(load_dataset(DATASET_DIR)))
    if CORPUS_PATH.exists():
        st.text_area("Merged corpus preview", CORPUS_PATH.read_text(encoding="utf-8")[:8000], height=400)
    if BENCHMARK_PATH.exists():
        st.subheader("Benchmark questions")
        st.dataframe(pd.read_json(BENCHMARK_PATH), use_container_width=True)
