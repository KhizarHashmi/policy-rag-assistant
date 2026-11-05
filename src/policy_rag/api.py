"""FastAPI service.

    uvicorn policy_rag.api:app --reload
    POST /ingest              rebuild the index from data/policies
    POST /ask                 {"question": "...", "plan": "plus" | null, "top_k": 5, "mode": "hybrid"}
    GET  /health
"""
from typing import Literal

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from .pipeline import RAGPipeline, get_pipeline

app = FastAPI(title="Policy RAG Assistant", version="0.1.0",
              description="Grounded Q&A over health-insurance policy documents, with citations.")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=500)
    plan: Literal["essential", "plus", "premier"] | None = None
    top_k: int = Field(5, ge=1, le=20)
    mode: Literal["hybrid", "dense", "bm25"] = "hybrid"


class Citation(BaseModel):
    label: str
    chunk_id: str
    document: str
    section: str


class AskResponse(BaseModel):
    answer: str
    answerable: bool
    plan_filter: str | None
    citations: list[Citation]
    retrieved: list[dict]
    timings_ms: dict[str, int]


@app.get("/health")
def health(pipe: RAGPipeline = Depends(get_pipeline)) -> dict:
    return {"status": "ok", "indexed_chunks": len(pipe.retriever.ids)}


@app.post("/ingest")
def ingest(pipe: RAGPipeline = Depends(get_pipeline)) -> dict:
    return {"indexed_chunks": pipe.ingest()}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest, pipe: RAGPipeline = Depends(get_pipeline)) -> dict:
    if not pipe.retriever.ids:
        raise HTTPException(status_code=409, detail="Index is empty. POST /ingest first.")
    return pipe.ask(req.question, req.plan, req.top_k, req.mode)
