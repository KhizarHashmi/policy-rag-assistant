"""The RAG pipeline: query understanding -> hybrid retrieval -> grounded generation."""
import time
from functools import lru_cache

from .chunking import load_corpus
from .config import CHROMA_DIR, DOCS_DIR, settings
from .embeddings import Embedder, GeminiEmbedder
from .generation import generate
from .llm import get_llm
from .retrieval import HybridRetriever, detect_plan


class RAGPipeline:
    def __init__(self, embedder: Embedder | None = None, llm=None, persist_dir=CHROMA_DIR):
        self.retriever = HybridRetriever(embedder or GeminiEmbedder(), persist_dir, settings.collection)
        self._llm = llm

    @property
    def llm(self):
        return self._llm or get_llm()

    def ingest(self, docs_dir=DOCS_DIR) -> int:
        return self.retriever.index(load_corpus(docs_dir, settings.chunk_max_chars))

    def ask(self, question: str, plan: str | None = None, k: int | None = None, mode: str = "hybrid") -> dict:
        t0 = time.perf_counter()
        plan = plan or detect_plan(question)
        hits = self.retriever.search(question, k or settings.top_k, plan, mode)
        t1 = time.perf_counter()
        answer = generate(self.llm, question, hits)
        t2 = time.perf_counter()
        label = {f"S{i}": h for i, h in enumerate(hits, start=1)}
        return {
            "answer": answer.answer,
            "answerable": answer.answerable,
            "plan_filter": plan,
            "citations": [{"label": s, "chunk_id": label[s].id, "document": label[s].metadata["title"],
                           "section": label[s].metadata["section"]} for s in answer.cited_sources if s in label],
            "retrieved": [{"chunk_id": h.id, "score": h.score, "section": h.metadata["section"]} for h in hits],
            "timings_ms": {"retrieval": round(1000 * (t1 - t0)), "generation": round(1000 * (t2 - t1))},
        }


@lru_cache(maxsize=1)
def get_pipeline() -> RAGPipeline:
    return RAGPipeline()
