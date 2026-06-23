"""Evaluation: compare Flat RAG vs GraphRAG on benchmark questions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import networkx as nx
import pandas as pd
from openai import OpenAI

from src.config import BENCHMARK_PATH, EVAL_RESULTS_PATH, LLM_MODEL, OPENAI_API_KEY
from src.flat_rag import FlatRAG
from src.querying import answer_with_graph


@dataclass
class EvalRow:
    id: int
    question: str
    question_type: str
    ground_truth: str
    flat_rag_answer: str
    graph_rag_answer: str
    flat_correct: str
    graph_correct: str
    flat_judge_reason: str
    graph_judge_reason: str
    graph_wins: str
    flat_latency: float
    graph_latency: float
    flat_tokens: int
    graph_tokens: int
    notes: str


JUDGE_PROMPT = """Bạn là giám khảo đánh giá câu trả lời QA.

Nhiệm vụ: Xác định câu trả lời có chứa đủ thông tin đúng theo ground truth không.
- Chấp nhận diễn đạt khác nếu ý nghĩa đúng.
- Với câu multi-hop, cần đủ các mắt xích quan trọng trong ground truth.
- Không chấp nhận nếu thiếu thông tin cốt lõi hoặc mâu thuẫn.

Trả về JSON:
{{"correct": true hoặc false, "reason": "giải thích ngắn bằng tiếng Việt"}}

Ground truth: {ground_truth}
Câu trả lời: {answer}
"""


def judge_answer(answer: str, ground_truth: str, client: OpenAI | None = None) -> tuple[bool, str]:
    """Use LLM to judge answer correctness."""
    if not OPENAI_API_KEY:
        gt_words = {w for w in ground_truth.lower().split() if len(w) > 2}
        ans_lower = answer.lower()
        overlap = sum(1 for w in gt_words if w in ans_lower)
        correct = overlap >= max(1, len(gt_words) // 3)
        return correct, "demo keyword match"

    client = client or OpenAI(api_key=OPENAI_API_KEY)
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(ground_truth=ground_truth, answer=answer)}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        result = json.loads(content)
        return bool(result.get("correct", False)), result.get("reason", "")
    except json.JSONDecodeError:
        return False, "parse error"


def compute_summary(df: pd.DataFrame) -> dict:
    """Return summary metrics dict for CLI/Streamlit."""
    flat_acc = (df["flat_correct"] == "Đúng").mean() * 100
    graph_acc = (df["graph_correct"] == "Đúng").mean() * 100
    graph_wins = int((df["graph_wins"] == "Có").sum())
    both_correct = int(((df["flat_correct"] == "Đúng") & (df["graph_correct"] == "Đúng")).sum())
    both_wrong = int(((df["flat_correct"] == "Sai") & (df["graph_correct"] == "Sai")).sum())

    multi = df[df["question_type"] == "multi_hop"]
    multi_flat = (multi["flat_correct"] == "Đúng").mean() * 100 if len(multi) else 0
    multi_graph = (multi["graph_correct"] == "Đúng").mean() * 100 if len(multi) else 0

    return {
        "flat_accuracy_pct": round(flat_acc, 1),
        "graph_accuracy_pct": round(graph_acc, 1),
        "graph_wins_when_flat_wrong": graph_wins,
        "both_correct": both_correct,
        "both_wrong": both_wrong,
        "multi_hop_flat_accuracy_pct": round(multi_flat, 1),
        "multi_hop_graph_accuracy_pct": round(multi_graph, 1),
        "avg_latency_flat_sec": round(df["flat_latency"].mean(), 3),
        "avg_latency_graph_sec": round(df["graph_latency"].mean(), 3),
        "total_tokens_flat": int(df["flat_tokens"].sum()),
        "total_tokens_graph": int(df["graph_tokens"].sum()),
    }


def run_evaluation(
    graph: nx.DiGraph,
    flat_rag: FlatRAG,
    benchmark_path=BENCHMARK_PATH,
    output_path=EVAL_RESULTS_PATH,
    progress_callback=None,
) -> pd.DataFrame:
    """Run full benchmark and save results CSV."""
    questions = json.loads(benchmark_path.read_text(encoding="utf-8"))
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
    rows: list[EvalRow] = []
    total = len(questions)

    for i, item in enumerate(questions):
        if progress_callback:
            progress_callback(f"Evaluating Q{item['id']}/{total}...", (i + 1) / total)

        qid = item["id"]
        question = item["question"]
        ground_truth = item["ground_truth"]
        qtype = item.get("type", "unknown")
        notes = item.get("notes", "")

        flat_result = flat_rag.answer(question)
        graph_result = answer_with_graph(question, graph, client=client)

        flat_ok, flat_reason = judge_answer(flat_result.answer, ground_truth, client)
        graph_ok, graph_reason = judge_answer(graph_result.answer, ground_truth, client)
        wins = not flat_ok and graph_ok

        rows.append(
            EvalRow(
                id=qid,
                question=question,
                question_type=qtype,
                ground_truth=ground_truth,
                flat_rag_answer=flat_result.answer,
                graph_rag_answer=graph_result.answer,
                flat_correct="Đúng" if flat_ok else "Sai",
                graph_correct="Đúng" if graph_ok else "Sai",
                flat_judge_reason=flat_reason,
                graph_judge_reason=graph_reason,
                graph_wins="Có" if wins else "Không",
                flat_latency=round(flat_result.latency_sec, 3),
                graph_latency=round(graph_result.latency_sec, 3),
                flat_tokens=flat_result.prompt_tokens + flat_result.completion_tokens,
                graph_tokens=graph_result.prompt_tokens + graph_result.completion_tokens,
                notes=notes if wins else "",
            )
        )
        time.sleep(0.15)

    df = pd.DataFrame([r.__dict__ for r in rows])
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    return df


def print_summary(df: pd.DataFrame) -> None:
    s = compute_summary(df)
    print("\n" + "=" * 60)
    print("BANG SO SANH FLAT RAG vs GRAPHRAG")
    print("=" * 60)
    print(f"Flat RAG accuracy:        {s['flat_accuracy_pct']}%")
    print(f"GraphRAG accuracy:        {s['graph_accuracy_pct']}%")
    print(f"Multi-hop Flat accuracy:  {s['multi_hop_flat_accuracy_pct']}%")
    print(f"Multi-hop Graph accuracy: {s['multi_hop_graph_accuracy_pct']}%")
    print(f"GraphRAG wins (Flat wrong): {s['graph_wins_when_flat_wrong']} questions")
    print(f"Avg latency Flat:         {s['avg_latency_flat_sec']}s")
    print(f"Avg latency Graph:        {s['avg_latency_graph_sec']}s")
    print(f"Total tokens Flat:        {s['total_tokens_flat']}")
    print(f"Total tokens Graph:       {s['total_tokens_graph']}")
    print("=" * 60)
