"""Command line: Python -m docchat ingest ask eval"""
import argparse
import sys

from .evaluate import evaluate, load_cases
from .llm import MissingAPIKey
from .pipeline import DocChat

def main(argv: list[str] | None = None) -> None:
    # Subparse container so users can run distinct commands: ingest, ask, eval
    parser = argparse.ArgumentParser(prog="docchat", description="Chat with your documents.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="index a filder of pdf, md, txt").add_argument("folder")
    sub.add_parser("ask", help="ask a question").add_argument("question")
    ev = sub.add_parser("eval", help="score retrieval on a JSONL file")
    ev.add_argument("cases")
    ev.add_argument("--k", type=int, default=4)

    args = parser.parse_args(argv)

    app = DocChat() # instantiate the complete Docchat engine
    if args.command == "ingest":
        n = app.ingest_folder(args.folder)
        print(f"Indexed {n} chunks. Total in store: {app.store.count()}")
    elif args.command == "ask":
        try:
            result = app.ask(args.question)
        except MissingAPIKey as exc:
            # gracefully exit: do not class with no handsome Python stack trace
            sys.exit(f"error: {exc}")

        print(result.text)

        print("\nSources:")
        # then loop over the verified cited chunks and print human readable
        for n, chunk in enumerate(result.sources, start = 1):
            print(f"  - {chunk.source} (page {chunk.page})")

    elif args.command == "eval":
        cases = load_cases(args.cases)
        print(f"{'mode':<8} {'hit@k':>6} {'mrr':>6}")
        for mode in ("dense", "sparse", "hybrid"):
            r = evaluate(app.retriever, cases, k=args.k, mode=mode)
            print(f"{mode:<8} {r['hit_rate']:>6.2f} {r['mrr']:>6.2f}")



