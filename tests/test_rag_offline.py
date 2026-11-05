"""Offline tests: deterministic hash embeddings + a fake LLM. No API key or network needed."""
import pytest
from fastapi.testclient import TestClient

from policy_rag import api
from policy_rag.chunking import chunk_markdown, load_corpus
from policy_rag.config import DOCS_DIR
from policy_rag.embeddings import HashEmbedder
from policy_rag.generation import Answer
from policy_rag.pipeline import RAGPipeline
from policy_rag.retrieval import detect_plan, rrf

DOC = """---
title: Test Plan
plan: plus
---
# Test Plan
## Maternity
Normal delivery up to SGD 4,000.

## Hospital Room and Board
Single room up to SGD 450 per day.
"""


class FakeLLM:
    def with_structured_output(self, schema):
        class R:
            def invoke(self, prompt):
                return Answer(answerable=True, answer="Up to SGD 1,800 per day [S1].", cited_sources=["S1", "S9"])
        return R()


@pytest.fixture(scope="module")
def pipe(tmp_path_factory):
    p = RAGPipeline(embedder=HashEmbedder(), llm=FakeLLM(), persist_dir=tmp_path_factory.mktemp("chroma"))
    p.ingest(DOCS_DIR)
    return p


def test_chunks_follow_headings_and_carry_context():
    chunks = chunk_markdown(DOC, "test_plan")
    assert [c.section for c in chunks] == ["Test Plan > Maternity", "Test Plan > Hospital Room and Board"]
    assert chunks[0].text.startswith("Test Plan | plan: plus | section: Test Plan > Maternity")
    assert chunks[1].id == "test_plan::hospital-room-and-board::0"


def test_long_sections_split_on_paragraphs():
    body = "\n\n".join(f"Paragraph {i} " + "x" * 300 for i in range(5))
    chunks = chunk_markdown(f"# T\n## S\n{body}", "d", max_chars=700)
    assert len(chunks) == 3 and all(len(c.body) <= 700 for c in chunks)


def test_rrf_rewards_agreement():
    scores = rrf([["a", "b", "c"], ["b", "a", "d"]])
    assert max(scores, key=scores.get) in {"a", "b"} and scores["a"] > scores["c"] > 0
    assert scores["d"] < scores["b"]


def test_plan_detection():
    assert detect_plan("ICU limit on the Plus plan?") == "plus"
    assert detect_plan("Compare Essential and Premier") is None
    assert detect_plan("Is cosmetic surgery covered?") is None


def test_plan_filter_keeps_plan_and_general_docs(pipe):
    hits = pipe.retriever.search("room and board limit per day", k=10, plan="essential")
    assert hits and all(h.metadata["plan"] in ("essential", "all") for h in hits)


def test_bm25_finds_exact_terms(pipe):
    top = pipe.retriever.search("intensive care unit ICU Plus", k=3, plan="plus", mode="bm25")
    assert top[0].id.startswith("plan_plus::hospital-room-and-board")


def test_ask_drops_hallucinated_citation_labels(pipe):
    out = pipe.ask("What is the ICU daily limit under the Plus plan?")
    assert out["plan_filter"] == "plus"
    assert [c["label"] for c in out["citations"]] == ["S1"]  # S9 was never retrieved


def test_api_contract(pipe, monkeypatch):
    api.app.dependency_overrides[api.get_pipeline] = lambda: pipe
    client = TestClient(api.app)
    assert client.get("/health").json()["indexed_chunks"] == len(load_corpus(DOCS_DIR))
    r = client.post("/ask", json={"question": "ICU limit on Plus?", "top_k": 3})
    assert r.status_code == 200 and r.json()["answerable"] and len(r.json()["retrieved"]) == 3
    assert client.post("/ask", json={"question": "x"}).status_code == 422          # too short
    assert client.post("/ask", json={"question": "ICU?", "plan": "gold"}).status_code == 422  # unknown plan
    api.app.dependency_overrides.clear()
