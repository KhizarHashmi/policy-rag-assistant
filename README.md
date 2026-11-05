# Policy RAG Assistant

Grounded question-answering over health-insurance policy documents. It uses **hybrid retrieval**
(ChromaDB dense vectors + BM25, fused with Reciprocal Rank Fusion), plan-aware metadata filtering,
**citation-enforced generation** with an explicit "not in the documents" path, a **FastAPI** service
and Docker packaging. A retrieval ablation shows which component earns its keep.

## Architecture

```
 INGEST   policies/*.md ─► heading-aware chunker ─► contextual header ─► Gemini embeddings ─► ChromaDB (cosine HNSW)
                                                     "Plus Plan | plan: plus | section: Maternity"        └─► BM25 index

 QUERY    question ─► plan detection ("...on the Plus plan" ─► filter plan ∈ {plus, all})
                   ─► dense top-20 ┐
                   ─► BM25 top-20  ┴─► Reciprocal Rank Fusion ─► top-5 ─► LLM (structured: answer, citations, answerable)
                                                                          └─► citations validated against retrieved set
```

| Concern | Choice | Why |
|---|---|---|
| Chunking | Split on markdown headings, then pack paragraphs to ≤900 chars | One policy section usually answers one question. Fixed-size windows cut tables and clauses in half |
| Context | Prefix every chunk with title, plan and section | "Covered up to SGD 450 per day" is useless without knowing *which plan* |
| Dense | `gemini-embedding-001`, 768-d (Matryoshka truncation, re-normalised), asymmetric task types | Separate document and query embeddings improve retrieval. 768-d keeps the index small |
| Sparse | BM25 (`rank_bm25`) | Exact tokens embeddings blur: amounts, "ICU", plan names |
| Fusion | RRF, k=60 | Rank-based, so cosine and BM25 scores never need calibrating |
| Filtering | Plan detected from the query, Chroma `where` filter plus the same filter on BM25 | Three plans share near-identical wording; filtering removes cross-plan confusion |
| Generation | Structured output (`answerable`, `answer`, `cited_sources`); citations not in the retrieved set are dropped | Faithfulness you can check in code; refusal is a first-class outcome |
| Serving | FastAPI + Pydantic request/response models, dependency-injected pipeline | Typed contract, 422 on bad input, easy to test with `TestClient` |

## Results (17 answerable + 3 unanswerable questions, `eval/qa.json`)

**Retrieval ablation** (a hit means a chunk from the expected document *and* section is retrieved):

| Retriever | hit@1 | hit@5 | MRR |
|---|---|---|---|
| BM25 | 0.824 | 1.000 | 0.865 |
| BM25 + plan filter | 0.824 | 1.000 | 0.897 |
| Dense | 1.000 | 1.000 | 1.000 |
| Dense + plan filter | 1.000 | 1.000 | 1.000 |
| Hybrid (RRF) | 0.882 | 1.000 | 0.941 |
| **Hybrid + plan filter (default)** | **1.000** | **1.000** | **1.000** |

**End to end** (hybrid + plan filter, k=5): answer accuracy **100%**, citation rate **100%**, correct refusal on unanswerable questions **3/3**.

**What the ablation shows:** on this small, clean corpus, dense retrieval alone is already perfect. Unfiltered RRF *lowered* hit@1 (1.00 → 0.88), because BM25's weaker ranking pulled near-duplicate sections from other plans upward. Hybrid is not automatically better. The plan filter removed the cross-plan confusion and restored 1.00. Hybrid stays the default because it is the robust choice once queries contain codes, amounts and rare terms (larger real corpora). That claim should be re-checked with this same ablation on real data.

> Small benchmark: 20 questions over 5 documents (45 chunks). Treat it as a regression suite, not a leaderboard.

## Quick start

```bash
# Python 3.11+. On Windows, use python.org / uv-managed Python, not Anaconda (see Troubleshooting)
python -m venv .venv && source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
cp .env.example .env                                     # add GOOGLE_API_KEY

pytest -q                                                # 8 offline tests (hash embedder + fake LLM)
python -m policy_rag ingest                              # chunk, embed, index (45 chunks)
python -m policy_rag ask "What is the ICU daily limit under the Plus plan?"
python -m policy_rag eval                                # ablation table + answer metrics
python -m policy_rag serve                               # http://127.0.0.1:8000/docs
```

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
     -d '{"question": "Is maternity covered on Plus, and what is the caesarean limit?"}'
```

Docker:

```bash
docker build -t policy-rag .
docker run -p 8000:8000 -e GOOGLE_API_KEY=$GOOGLE_API_KEY policy-rag
```

## Troubleshooting

**Segfault in `chromadb` / `onnxruntime` on Windows with Anaconda.** Anaconda ships older MSVC runtime
DLLs (`msvcp140.dll`) that load ahead of the system ones and crash newer native wheels (Chroma's Rust
core, onnxruntime). Create the venv from a python.org or uv-managed interpreter instead:
`uv python install 3.12 && uv venv --python 3.12 --python-preference only-managed`.

## Roadmap

- Cross-encoder re-ranking (e.g. `bge-reranker`) on the fused top-20
- Query rewriting for multi-plan comparison questions, plus HyDE for vague queries
- Faithfulness scoring (claim-level entailment against cited chunks) in the eval
- PDF ingestion with layout-aware parsing of benefit tables
