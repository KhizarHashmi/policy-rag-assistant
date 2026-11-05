"""Hybrid retrieval: dense (ChromaDB, cosine HNSW) + sparse (BM25), fused with Reciprocal Rank Fusion.

Why hybrid: dense vectors capture meaning ("having a baby" ~ "maternity"), while BM25 nails exact
tokens that embeddings blur: amounts, plan names, codes ("ICU", "SGD 1,800", "Premier").
Why RRF: dense cosine scores and BM25 scores live on incomparable scales. RRF fuses by rank only,
score(d) = sum over retrievers of 1 / (k + rank(d)), so no score calibration is needed.
"""
import re
from dataclasses import dataclass
from pathlib import Path

import chromadb
from rank_bm25 import BM25Okapi

from .chunking import Chunk
from .embeddings import Embedder

PLANS = ("essential", "plus", "premier")
RRF_K = 60


@dataclass
class Hit:
    id: str
    text: str
    metadata: dict
    score: float


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:[.,][0-9]+)*", text.lower())


def rrf(rankings: list[list[str]], k: int = RRF_K) -> dict[str, float]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    return scores


def detect_plan(question: str) -> str | None:
    """Lightweight query understanding: if the user names exactly one plan, scope retrieval to it."""
    found = {p for p in PLANS if re.search(rf"\b{p}\b", question, re.I)}
    return found.pop() if len(found) == 1 else None


class HybridRetriever:
    def __init__(self, embedder: Embedder, persist_dir: Path, collection: str = "policies"):
        self.embedder = embedder
        self.client = chromadb.PersistentClient(path=str(persist_dir))
        self.collection = self.client.get_or_create_collection(collection, metadata={"hnsw:space": "cosine"})
        self._load_sparse()

    # ---------- indexing ----------
    def index(self, chunks: list[Chunk]) -> int:
        """Idempotent full rebuild: drop and re-create the collection, then upsert all chunks."""
        name, meta = self.collection.name, self.collection.metadata
        self.client.delete_collection(name)
        self.collection = self.client.create_collection(name, metadata=meta)
        texts = [c.text for c in chunks]
        self.collection.add(ids=[c.id for c in chunks], documents=texts,
                            metadatas=[c.metadata() for c in chunks], embeddings=self.embedder.embed_documents(texts))
        self._load_sparse()
        return len(chunks)

    def _load_sparse(self) -> None:
        data = self.collection.get(include=["documents", "metadatas"])
        self.docs = {i: (d, m) for i, d, m in zip(data["ids"], data["documents"], data["metadatas"])}
        self.ids = list(self.docs)
        self.bm25 = BM25Okapi([tokenize(self.docs[i][0]) for i in self.ids]) if self.ids else None

    # ---------- search ----------
    def _allowed(self, doc_id: str, plan: str | None) -> bool:
        return plan is None or self.docs[doc_id][1]["plan"] in (plan, "all")

    def dense(self, query: str, n: int, plan: str | None = None) -> list[str]:
        where = {"plan": {"$in": [plan, "all"]}} if plan else None
        res = self.collection.query(query_embeddings=[self.embedder.embed_query(query)], n_results=n, where=where)
        return res["ids"][0]

    def sparse(self, query: str, n: int, plan: str | None = None) -> list[str]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(self.ids)), key=lambda i: scores[i], reverse=True)
        return [self.ids[i] for i in ranked if scores[i] > 0 and self._allowed(self.ids[i], plan)][:n]

    def search(self, query: str, k: int = 5, plan: str | None = None, mode: str = "hybrid") -> list[Hit]:
        if not self.ids:
            raise RuntimeError("Index is empty - run ingestion first.")
        pool = max(k * 4, 20)  # retrieve deeper than k so fusion has candidates to re-rank
        if mode == "dense":
            ranked = {d: 1.0 / (RRF_K + r) for r, d in enumerate(self.dense(query, pool, plan), 1)}
        elif mode == "bm25":
            ranked = {d: 1.0 / (RRF_K + r) for r, d in enumerate(self.sparse(query, pool, plan), 1)}
        else:
            ranked = rrf([self.dense(query, pool, plan), self.sparse(query, pool, plan)])
        top = sorted(ranked.items(), key=lambda kv: kv[1], reverse=True)[:k]
        return [Hit(i, self.docs[i][0], self.docs[i][1], round(s, 5)) for i, s in top]
