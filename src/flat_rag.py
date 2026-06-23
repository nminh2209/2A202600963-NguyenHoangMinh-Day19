"""Flat RAG using ChromaDB vector search."""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass

import chromadb
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction
from openai import OpenAI

from src.config import CHROMA_DIR, EMBEDDING_MODEL, LLM_MODEL, get_openai_api_key


@dataclass
class FlatRAGResult:
    question: str
    answer: str
    context: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_sec: float = 0.0


def chunk_corpus(text: str, chunk_size: int = 300) -> list[str]:
    """Split corpus into sentence-based chunks."""
    sentences = [s.strip() for s in text.replace("\n", " ").split(".") if s.strip()]
    chunks = []
    current = ""
    for sent in sentences:
        if len(current) + len(sent) < chunk_size:
            current += sent + ". "
        else:
            if current:
                chunks.append(current.strip())
            current = sent + ". "
    if current:
        chunks.append(current.strip())
    return chunks


class FlatRAG:
    def __init__(self, corpus_path, collection_name: str = "tech_corpus"):
        self.corpus_path = corpus_path
        self.collection_name = collection_name
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        self._collection = None

    def _get_embedding_function(self):
        api_key = get_openai_api_key()
        if not api_key:
            return None
        return OpenAIEmbeddingFunction(api_key=api_key, model_name=EMBEDDING_MODEL)

    def index(self, force_rebuild: bool = False) -> int:
        """Index corpus into ChromaDB. Returns number of chunks."""
        if force_rebuild:
            try:
                self.client.delete_collection(self.collection_name)
            except Exception:
                pass

        ef = self._get_embedding_function()
        self._collection = self.client.get_or_create_collection(
            name=self.collection_name,
            embedding_function=ef,
        )

        if self._collection.count() > 0 and not force_rebuild:
            return self._collection.count()

        text = self.corpus_path.read_text(encoding="utf-8")
        chunks = chunk_corpus(text)

        api_key = get_openai_api_key()
        if not api_key:
            # Chroma default embedding when no API key
            self._collection = self.client.get_or_create_collection(name=self.collection_name)

        ids = [str(uuid.uuid4()) for _ in chunks]
        self._collection.add(documents=chunks, ids=ids)
        return len(chunks)

    def retrieve(self, question: str, top_k: int = 3) -> str:
        if self._collection is None:
            self.index()
        results = self._collection.query(query_texts=[question], n_results=top_k)
        docs = results.get("documents", [[]])[0]
        return "\n\n".join(docs) if docs else ""

    def answer(self, question: str, top_k: int = 3) -> FlatRAGResult:
        start = time.perf_counter()
        context = self.retrieve(question, top_k=top_k)

        api_key = get_openai_api_key()
        if not api_key:
            return FlatRAGResult(
                question=question,
                answer=f"[Demo] Dựa trên vector search: {context[:400]}",
                context=context,
                latency_sec=time.perf_counter() - start,
            )

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Trả lời câu hỏi dựa trên ngữ cảnh được cung cấp. "
                        "Trả lời ngắn gọn bằng tiếng Việt."
                    ),
                },
                {"role": "user", "content": f"Ngữ cảnh:\n{context}\n\nCâu hỏi: {question}"},
            ],
            temperature=0,
        )
        usage = response.usage
        return FlatRAGResult(
            question=question,
            answer=response.choices[0].message.content or "",
            context=context,
            prompt_tokens=usage.prompt_tokens if usage else 0,
            completion_tokens=usage.completion_tokens if usage else 0,
            latency_sec=time.perf_counter() - start,
        )
