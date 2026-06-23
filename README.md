# LAB DAY 19: GraphRAG với Tech Company Corpus

## Cấu trúc dự án

```
├── main.py                      # Pipeline chính
├── graphrag_lab19.ipynb         # Notebook báo cáo
├── noderag_setup.py             # Tích hợp NodeRAG (tùy chọn)
├── requirements.txt
├── data/
│   ├── tech_company_corpus.txt  # Corpus văn bản
│   └── benchmark_questions.json # 20 câu hỏi benchmark
├── src/
│   ├── entity_extraction.py     # Bước 1: Trích xuất triple
│   ├── graph_construction.py    # Bước 2: NetworkX + Neo4j
│   ├── querying.py              # Bước 3: GraphRAG query
│   ├── flat_rag.py              # Flat RAG (ChromaDB)
│   ├── evaluation.py            # Bước 4: So sánh
│   └── visualize.py             # Vẽ đồ thị Matplotlib
└── output/
    ├── knowledge_graph.png      # Ảnh đồ thị (deliverable)
    ├── evaluation_results.csv   # Bảng 20 câu hỏi
    └── cost_analysis.json       # Phân tích chi phí
```

## Cài đặt

```bash
pip install -r requirements.txt
cp .env.example .env
# Thêm OPENAI_API_KEY vào file .env
```

## Chạy pipeline

```bash
# Demo (không cần API key - dùng triples mẫu)
python main.py --demo

# Full pipeline với OpenAI
python main.py

# Streamlit demo UI
streamlit run streamlit_app.py

# Hỏi 1 câu
python main.py --question "Ai là CEO của công ty sở hữu DeepMind?"

# Đẩy đồ thị lên Neo4j (cần Neo4j Desktop/Docker)
python main.py --neo4j
```

---

## PHẦN 1: NGHIÊN CỨU (Research Answers)

### 2.1.1 Entity Extraction: Phân biệt Node và Thuộc tính?

LLM phân biệt thực thể (Node) và thuộc tính qua **prompt engineering** và **schema**:

| Loại | Ví dụ | Cách xử lý |
|------|-------|------------|
| **Node (Thực thể)** | OpenAI, Sam Altman, 2015 | Tên riêng, có thể có nhiều quan hệ |
| **Thuộc tính** | "lớn", "nổi tiếng" | Gắn vào edge hoặc bỏ qua |
| **Quan hệ (Edge)** | FOUNDED_BY, CEO_OF | Động từ/cụm quan hệ giữa 2 node |

**Quy tắc:** Nếu một khái niệm có thể **kết nối với nhiều thực thể khác** → Node. Nếu chỉ **mô tả một thực thể** → thuộc tính của node đó.

Ví dụ: `"OpenAI được thành lập năm 2015"` → `(OpenAI, FOUNDED_IN, 2015)` — năm 2015 có thể là node hoặc property tùy schema.

### 2.1.2 Graph Construction: Tại sao Deduplication quan trọng?

- **"Google" vs "google" vs "Google Inc."** → nếu không gộp sẽ tạo nhiều node trùng nghĩa
- Đồ thị phình to, BFS trả về context rối
- Quan hệ bị phân mảnh: `(Google, ACQUIRED, DeepMind)` và `(google, bought, DeepMind)` không nối được
- Giải pháp: normalize (lowercase, trim), entity resolution, merge aliases

### 2.1.3 Query Answering: BFS vs Vector Search?

| | **Vector Search (Flat RAG)** | **BFS Graph Traversal (GraphRAG)** |
|---|---|---|
| Cơ chế | Embedding similarity | Duyệt cạnh theo quan hệ |
| Điểm mạnh | Tìm đoạn văn liên quan ngữ nghĩa | Multi-hop reasoning có cấu trúc |
| Điểm yếu | Khó nối A→B→C nếu không cùng chunk | Phụ thuộc chất lượng đồ thị |
| Ví dụ | "CEO Google" có thể trả chunk về Larry Page | BFS: DeepMind → Google → Sundar Pichai |

---

## So sánh công cụ

| Tool | Ưu điểm | Phù hợp khi |
|------|---------|-------------|
| **NetworkX** | Offline, prototype nhanh, thuật toán đồ thị | Notebook, nghiên cứu thuật toán |
| **Neo4j** | Cypher, Bloom visualization, production | Cần nhìn thấy đồ thị trực quan |
| **NodeRAG** | All-in-one, tích hợp sẵn retrieval | Muốn giải pháp trọn gói |

---

## Deliverables

1. **Mã nguồn**: `main.py`, `src/`, `graphrag_lab19.ipynb`
2. **Ảnh đồ thị**: `output/knowledge_graph.png`
3. **Bảng 20 câu hỏi**: `output/evaluation_results.csv`
4. **Phân tích chi phí**: `output/cost_analysis.json`
