"""Evaluation: compare Flat RAG vs GraphRAG on benchmark questions."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass

import networkx as nx
import pandas as pd
from openai import OpenAI

from src.config import BENCHMARK_PATH, EVAL_RESULTS_PATH, LLM_MODEL, get_openai_api_key
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


JUDGE_PROMPT = """Bạn là giám khảo QA. Đánh giá câu trả lời có ĐÚNG ý chính so với ground truth không.

Quy tắc:
- correct=true nếu câu trả lời chứa đủ thông tin CỐT LÕI được hỏi (tên, năm, số liệu chính).
- Không yêu cầu thêm chi tiết ngoài câu hỏi (ví dụ hỏi ai sáng lập thì không cần nêu thêm công ty khác).
- Chấp nhận diễn đạt khác, đơn vị khác (tỷ đô la = USD).
- correct=false chỉ khi SAI fact hoặc THIẾU hoàn toàn thông tin được hỏi.

JSON: {{"correct": true/false, "reason": "ngắn gọn tiếng Việt"}}

Ground truth: {ground_truth}
Câu trả lời: {answer}
"""


def _keyword_overlap_score(answer: str, ground_truth: str) -> float:
    """Heuristic overlap for key terms (names, years, numbers)."""
    import re
    ans = answer.lower()
    gt_tokens = re.findall(r"[a-zA-ZÀ-ỹ0-9]+", ground_truth.lower())
    if not gt_tokens:
        return 0.0
    hits = 0
    for tok in gt_tokens:
        if len(tok) <= 2 and not tok.isdigit():
            continue
        if tok in ans:
            hits += 1
    significant = [t for t in gt_tokens if len(t) > 2 or t.isdigit()]
    if not significant:
        significant = gt_tokens
    return hits / len(significant)


def judge_answer(answer: str, ground_truth: str, client: OpenAI | None = None) -> tuple[bool, str]:
    """Use LLM + keyword heuristic to judge answer correctness."""
    overlap = _keyword_overlap_score(answer, ground_truth)
    if overlap >= 0.6:
        return True, f"keyword overlap {overlap:.0%}"

    if not get_openai_api_key():
        return overlap >= 0.4, "demo keyword match"

    client = client or OpenAI(api_key=get_openai_api_key())
    response = client.chat.completions.create(
        model=LLM_MODEL,
        messages=[{"role": "user", "content": JUDGE_PROMPT.format(ground_truth=ground_truth, answer=answer)}],
        temperature=0,
        response_format={"type": "json_object"},
    )
    content = response.choices[0].message.content or "{}"
    try:
        result = json.loads(content)
        llm_ok = bool(result.get("correct", False))
        reason = result.get("reason", "")
        # Trust heuristic if LLM disagrees strongly with obvious overlap
        if not llm_ok and overlap >= 0.5:
            return True, f"override: {overlap:.0%} keyword overlap ({reason})"
        return llm_ok, reason
    except json.JSONDecodeError:
        return overlap >= 0.5, "parse error, used keyword fallback"


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
    client = OpenAI(api_key=get_openai_api_key()) if get_openai_api_key() else None
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
