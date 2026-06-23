"""Configuration and paths for Lab 19 GraphRAG."""

from pathlib import Path

import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASET_DIR = PROJECT_ROOT / "dataset" / "dataset"
OUTPUT_DIR = PROJECT_ROOT / "output"

CORPUS_PATH = OUTPUT_DIR / "merged_corpus.txt"
BENCHMARK_PATH = DATA_DIR / "benchmark_questions.json"
TRIPLES_PATH = OUTPUT_DIR / "triples.json"
GRAPH_IMAGE_PATH = OUTPUT_DIR / "knowledge_graph.png"
EVAL_RESULTS_PATH = OUTPUT_DIR / "evaluation_results.csv"
COST_REPORT_PATH = OUTPUT_DIR / "cost_analysis.json"
CHROMA_DIR = OUTPUT_DIR / "chroma_db"

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")

LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def get_openai_api_key() -> str:
    """Read API key fresh from .env (avoids stale cache in long-running Streamlit)."""
    load_dotenv(override=True)
    return os.getenv("OPENAI_API_KEY", "").strip()


# Back-compat alias; refreshed at import time
OPENAI_API_KEY = get_openai_api_key()
