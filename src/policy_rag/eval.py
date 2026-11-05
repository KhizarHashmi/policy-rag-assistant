"""Evaluate retrieval (ablation across retrievers) and end-to-end answers.

Retrieval metrics over answerable questions (a hit = a retrieved chunk from the expected document
and section):
  hit@1, hit@k (recall@k), MRR (mean reciprocal rank of the first relevant chunk)
Answer metrics:
  answer accuracy (expected fact present), citation rate, refusal accuracy on unanswerable questions
"""
import json
from functools import lru_cache

from .config import EVAL_DIR
from .pipeline import RAGPipeline
from .retrieval import detect_plan


def load_qa() -> list[dict]:
    return json.loads((EVAL_DIR / "qa.json").read_text(encoding="utf-8"))


def is_relevant(hit_meta: dict, hit_id: str, q: dict) -> bool:
    docs = q["doc"] if isinstance(q["doc"], list) else [q["doc"]]
    return hit_id.split("::")[0] in docs and hit_meta["section"].split(" > ")[-1] == q["section"]


def retrieval_metrics(pipe: RAGPipeline, qa: list[dict], mode: str, use_plan: bool, k: int = 5) -> dict:
    hits1 = hitsk = rr = 0.0
    for q in qa:
        hits = pipe.retriever.search(q["question"], k, detect_plan(q["question"]) if use_plan else None, mode)
        ranks = [i for i, h in enumerate(hits, start=1) if is_relevant(h.metadata, h.id, q)]
        if ranks:
            hits1 += ranks[0] == 1
            hitsk += 1
            rr += 1 / ranks[0]
    n = len(qa)
    return {"hit@1": round(hits1 / n, 3), f"hit@{k}": round(hitsk / n, 3), "mrr": round(rr / n, 3)}


def run_eval(pipe: RAGPipeline | None = None, answers: bool = True) -> dict:
    pipe = pipe or RAGPipeline()
    embedder = pipe.retriever.embedder
    embedder.embed_query = lru_cache(maxsize=256)(embedder.embed_query)  # one embedding call per question
    qa = load_qa()
    answerable = [q for q in qa if not q.get("unanswerable")]

    print(f"Retrieval ablation ({len(answerable)} answerable questions, k=5)")
    print(f"  {'retriever':<24}{'hit@1':>8}{'hit@5':>8}{'MRR':>8}")
    ablation = {}
    for mode in ("bm25", "dense", "hybrid"):
        for use_plan in (False, True):
            name = f"{mode}{' + plan filter' if use_plan else ''}"
            m = retrieval_metrics(pipe, answerable, mode, use_plan)
            ablation[name] = m
            print(f"  {name:<24}{m['hit@1']:>8}{m['hit@5']:>8}{m['mrr']:>8}")

    result = {"retrieval": ablation}
    if answers:
        rows = []
        for q in qa:
            out = pipe.ask(q["question"])
            if q.get("unanswerable"):
                ok = not out["answerable"]
            else:
                ok = out["answerable"] and any(e.lower() in out["answer"].lower() for e in q["expect_any"])
            rows.append({"id": q["id"], "ok": ok, "cited": bool(out["citations"]), "answer": out["answer"],
                         "unanswerable": bool(q.get("unanswerable"))})
            print(f"  [{'OK ' if ok else 'BAD'}] {q['id']}: {out['answer'][:110]}")
        ans = [r for r in rows if not r["unanswerable"]]
        una = [r for r in rows if r["unanswerable"]]
        result["answers"] = {
            "answer_accuracy": round(sum(r["ok"] for r in ans) / len(ans), 3),
            "citation_rate": round(sum(r["cited"] for r in ans) / len(ans), 3),
            "refusal_accuracy": round(sum(r["ok"] for r in una) / max(1, len(una)), 3),
        }
        result["rows"] = rows
        print("\nAnswers:", result["answers"])
    (EVAL_DIR / "results.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result
