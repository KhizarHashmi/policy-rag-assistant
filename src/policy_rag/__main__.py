"""CLI entry point.

    python -m policy_rag ingest
    python -m policy_rag ask "What is the ICU limit on the Plus plan?"
    python -m policy_rag eval
    python -m policy_rag serve            # FastAPI on http://127.0.0.1:8000/docs
"""
import argparse
import json
import sys


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(prog="policy_rag")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("ingest")
    a = sub.add_parser("ask")
    a.add_argument("question")
    a.add_argument("--plan", choices=["essential", "plus", "premier"])
    a.add_argument("--mode", default="hybrid", choices=["hybrid", "dense", "bm25"])
    e = sub.add_parser("eval")
    e.add_argument("--retrieval-only", action="store_true")
    s = sub.add_parser("serve")
    s.add_argument("--port", type=int, default=8000)
    args = p.parse_args()

    if args.cmd == "serve":
        import uvicorn
        uvicorn.run("policy_rag.api:app", host="0.0.0.0", port=args.port)
        return

    from .pipeline import RAGPipeline
    pipe = RAGPipeline()
    if args.cmd == "ingest":
        print(f"Indexed {pipe.ingest()} chunks.")
    elif args.cmd == "ask":
        print(json.dumps(pipe.ask(args.question, args.plan, mode=args.mode), indent=2, ensure_ascii=False))
    else:
        from .eval import run_eval
        run_eval(pipe, answers=not args.retrieval_only)


if __name__ == "__main__":
    main()
