# LAB DAY 19: GraphRAG — US Electric Vehicle Dataset

## Dataset

- **Source:** `dataset/dataset/` — 70 txt files (`doc_1.txt` … `doc_70.txt`)
- **Topic:** US electric vehicle sector (market, sentiment, policy, charging, stocks)
- **Cleaned corpus:** `output/merged_corpus.txt` — 68 usable documents after PDF/binary filtering (`doc_50`, `doc_60` rejected)
- **Benchmark:** 20 questions (12 multi-hop + 8 single-hop) in `data/benchmark_questions.json`

## Cấu trúc dự án

```
├── main.py
├── streamlit_app.py
├── verify_api.py
├── graphrag_lab19.ipynb
├── dataset/dataset/          # 70 source documents
├── data/benchmark_questions.json
├── src/
│   ├── corpus.py             # Load, clean & chunk documents
│   ├── entity_extraction.py  # LLM triple extraction
│   ├── fact_triples.py       # Corpus-mined fact enrichment
│   ├── graph_construction.py # NetworkX + multi-hop retrieval
│   ├── querying.py           # GraphRAG answer pipeline
│   ├── flat_rag.py           # ChromaDB vector RAG
│   ├── evaluation.py
│   └── pipeline.py
└── output/
    ├── merged_corpus.txt
    ├── triples.json
    ├── knowledge_graph.png
    ├── evaluation_results.csv
    ├── cost_analysis.json
    └── pipeline_run.log
```

## Cài đặt & chạy

```bash
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY

# Full pipeline (extract → enrich graph → ChromaDB → 20-question eval)
python main.py

# Re-run evaluation only (uses saved triples + latest graph logic)
python main.py --eval-only

# Streamlit demo UI
streamlit run streamlit_app.py
```

---

## Kết quả đánh giá (Full LLM — `gpt-4o-mini`, cleaned corpus)

### Đồ thị tri thức

| Metric | Giá trị |
|--------|---------|
| Source documents | 70 (68 usable after cleaning) |
| Triples (LLM + fact enrichment) | **810** |
| Nodes | **888** |
| Edges | **761** |
| Indexing time | ~887s (~105,927 tokens) |

### So sánh Flat RAG vs GraphRAG (20 câu hỏi benchmark)

| Metric | Flat RAG (ChromaDB) | GraphRAG (NetworkX + multi-hop) |
|--------|---------------------|----------------------------------|
| **Overall accuracy** | **80.0%** | **100.0%** |
| **Multi-hop accuracy** | **85.7%** | **100.0%** |
| **Graph wins (Flat sai → Graph đúng)** | — | **4** |
| **Both correct** | 16 | 16 |
| **Both wrong** | 0 | 0 |
| **Avg latency** | 2.22s | 2.26s |
| **Eval tokens** | ~16,595 | ~27,883 |

### GraphRAG thắng khi Flat RAG sai (4 câu)

| ID | Chủ đề | Flat RAG lỗi | GraphRAG đúng |
|----|--------|--------------|---------------|
| Q6 | J.D. Power VP + 29.2% | Không tìm được Elizabeth Krear | Elizabeth Krear @ J.D. Power |
| Q12 | Biden chargers 2030 | Nói ~1 triệu | **500,000** |
| Q13 | EV growth H1 2023 | Nói 58% (nhầm chunk) | **51%** |
| Q17 | J.D. Power 2026 scope | Nói 50% | **75%** |

### Phân tích chi phí

| Giai đoạn | Thời gian | Tokens |
|-----------|-----------|--------|
| Indexing (LLM extraction + enrichment) | ~887s | ~105,927 |
| Evaluation (20 questions × 2 systems) | ~104s | ~44,478 |
| **Tổng ước tính** | ~17 min | **~150,000** |

### Kết luận

| Hệ thống | Điểm mạnh | Điểm yếu |
|----------|-----------|----------|
| **Flat RAG** | Retrieval nhanh từ văn bản gốc; tốt single-hop | Dễ nhầm số liệu gần nghĩa (58% vs 51%, 1M vs 500k) |
| **GraphRAG** | **100% accuracy**; chain multi-hop (BNEF→McKerracher, J.D. Power→29.2%→Krear) | Cần extraction + fact enrichment; nhiều token hơn khi eval |

**Pipeline GraphRAG (graph-first):**
1. Clean corpus — loại PDF binary, cookie boilerplate
2. LLM triple extraction (small chunks) + `fact_triples` enrichment
3. Multi-hop retrieval — keyword search + 4-hop BFS + shortest paths
4. Answer — numbered FACT lines → strict prompt → direct triple fallback

---

## PHẦN 1: Nghiên cứu (Research Answers)

### Entity vs Attribute
- **Node:** Tesla, BloombergNEF, ZEV states, Colin McKerracher
- **Relation:** MARKET_SHARE, LEAD_AUTHOR_OF, STRIKE_AGAINST, SURVEY_RESULT
- **Attribute:** gắn vào edge/object (51%, 2027, $242 billion)

### Deduplication
- Gộp "GM" / "General Motors", "BNEF" / "BloombergNEF"
- Loại document trùng nội dung; dedupe lines trong scraped pages

### BFS vs Vector Search
- **Vector (Flat RAG):** similarity ngữ nghĩa — mạnh khi fact nằm trong 1 chunk
- **BFS + paths (GraphRAG):** duyệt quan hệ có cấu trúc — mạnh multi-hop khi đồ thị đầy đủ

---

## Deliverables

1. **Mã nguồn:** `main.py`, `streamlit_app.py`, `src/`, `graphrag_lab19.ipynb`
2. **Ảnh đồ thị:** `output/knowledge_graph.png`
3. **Bảng 20 câu hỏi:** `output/evaluation_results.csv`
4. **Phân tích chi phí:** `output/cost_analysis.json`
5. **Demo UI:** `streamlit run streamlit_app.py`
