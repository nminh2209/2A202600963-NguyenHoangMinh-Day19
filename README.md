# LAB DAY 19: GraphRAG — US Electric Vehicle Dataset (70 documents)

## Dataset

- **Source:** `dataset/dataset/` — 70 txt files (`doc_1.txt` … `doc_70.txt`)
- **Topic:** US electric vehicle sector (market, sentiment, policy, charging, stocks)
- **Merged corpus:** `output/merged_corpus.txt` (auto-generated)

## Cấu trúc dự án

```
├── main.py
├── streamlit_app.py
├── verify_api.py
├── graphrag_lab19.ipynb
├── dataset/dataset/          # 70 source documents
├── data/benchmark_questions.json
├── src/
│   ├── corpus.py             # Load & chunk 70 documents
│   ├── entity_extraction.py
│   ├── graph_construction.py
│   ├── querying.py
│   ├── flat_rag.py
│   ├── evaluation.py
│   └── pipeline.py
└── output/
    ├── merged_corpus.txt
    ├── triples.json
    ├── knowledge_graph.png
    ├── evaluation_results.csv
    └── cost_analysis.json
```

## Cài đặt & chạy

```bash
pip install -r requirements.txt
cp .env.example .env   # add OPENAI_API_KEY

# Full pipeline (70 docs → graph → ChromaDB → 20-question eval)
python main.py

# Streamlit demo UI
streamlit run streamlit_app.py

# Re-run evaluation only (uses saved triples)
python main.py --eval-only
```

---

## Kết quả đánh giá (Full LLM — `gpt-4o-mini`)

### Đồ thị tri thức

| Metric | Giá trị |
|--------|---------|
| Documents | 70 |
| Triples extracted | 670 |
| Nodes | 861 |
| Edges | 656 |
| Indexing time | ~488s (~81,671 tokens) |

### So sánh Flat RAG vs GraphRAG (20 câu hỏi benchmark)

| Metric | Flat RAG (ChromaDB) | GraphRAG (NetworkX + BFS) |
|--------|---------------------|---------------------------|
| **Overall accuracy** | **75.0%** | **25.0%** |
| **Multi-hop accuracy** | **75.0%** | **25.0%** |
| **Graph wins (Flat sai → Graph đúng)** | — | **0** |
| **Avg latency** | 2.6s | 2.0s |
| **Eval tokens** | ~24,000 | ~11,000 |

### Phân tích chi phí

| Giai đoạn | Thời gian | Tokens |
|-----------|-----------|--------|
| Indexing (70 docs × LLM) | ~488s | ~81,671 |
| Evaluation | ~118s | ~35,000 |
| **Tổng ước tính** | ~10 min | **~115,000+** |

### Kết luận

| Hệ thống | Điểm mạnh trên dataset này | Điểm yếu |
|----------|---------------------------|----------|
| **Flat RAG** | Truy xuất trực tiếp từ văn bản gốc (75% accuracy) | Khó multi-hop khi fact nằm rải nhiều chunk |
| **GraphRAG** | Tốt khi triple có trong đồ thị (Tesla, ZEV 5%, Biden 500k chargers) | Phụ thuộc extraction — nhiều fact bị bỏ sót → 25% accuracy |

**GraphRAG trả lời đúng khi:** triple đã được trích xuất (Q2 Tesla, Q8 Biden chargers, Q12 ZEV 5%, Q18 315k sales).

**GraphRAG thất bại khi:** fact có trong văn bản nhưng không có trong đồ thị (Q13 BNEF 2027 oil peak, Q14 Colin McKerracher, Q19 $242B charging market).

---

## PHẦN 1: Nghiên cứu (Research Answers)

### Entity vs Attribute
- **Node:** Tesla, BNEF, ZEV regulations, 2020
- **Relation:** MARKET_SHARE, PUBLISHED_BY, GROWTH_RATE
- **Attribute:** gắn vào edge/object, không tách node nếu không cần multi-hop

### Deduplication
- Gộp "GM" / "General Motors", chuẩn hóa numeric objects
- Tránh phình đồ thị (861 nodes từ 670 triples)

### BFS vs Vector Search
- **Vector:** tìm đoạn văn liên quan ngữ nghĩa — mạnh trên corpus 70 file
- **BFS:** duyệt quan hệ — chỉ hiệu quả khi extraction đầy đủ

---

## Deliverables

1. **Mã nguồn:** `main.py`, `streamlit_app.py`, `src/`, `graphrag_lab19.ipynb`
2. **Ảnh đồ thị:** `output/knowledge_graph.png`
3. **Bảng 20 câu hỏi:** `output/evaluation_results.csv`
4. **Phân tích chi phí:** `output/cost_analysis.json`
5. **Demo UI:** `streamlit run streamlit_app.py`
