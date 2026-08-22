"""Command line entry points.

    python -m app.cli init-db
    python -m app.cli ingest data/filings
    python -m app.cli build-index
    python -m app.cli query "What does Apple say about supply chain risk?"
    python -m app.cli stats
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from app.config import get_settings
from app.db import build_ivfflat_index, connect, count_chunks, init_schema
from app.embeddings import get_embedder
from app.generation import get_provider
from app.ingest import ingest_directory, ingest_file
from app.service import answer_question


def cmd_init_db(args: argparse.Namespace) -> int:
    with connect() as conn:
        init_schema(conn)
    print("schema applied")
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    settings = get_settings()
    path = Path(args.path)
    if not path.exists():
        print(f"no such path: {path}", file=sys.stderr)
        return 1

    embedder = get_embedder(args.embedder or settings.embedder)
    started = time.perf_counter()
    with connect() as conn:
        init_schema(conn)
        reports = (
            [ingest_file(conn, embedder, path)]
            if path.is_file()
            else ingest_directory(conn, embedder, path)
        )
        total_chunks = count_chunks(conn)

    for report in reports:
        print(report)
    elapsed = time.perf_counter() - started
    print(
        f"\n{len(reports)} document(s), "
        f"{sum(r.chunk_count for r in reports)} chunks embedded in {elapsed:.1f}s "
        f"(corpus now holds {total_chunks} chunks)"
    )
    if reports:
        print("next: python -m app.cli build-index")
    return 0


def cmd_build_index(args: argparse.Namespace) -> int:
    with connect() as conn:
        rows = count_chunks(conn)
        lists = build_ivfflat_index(conn, args.lists)
    print(f"ivfflat index built over {rows} chunks with lists={lists}")
    return 0


def cmd_query(args: argparse.Namespace) -> int:
    settings = get_settings()
    embedder = get_embedder(args.embedder or settings.embedder)
    provider = get_provider(args.provider)
    with connect() as conn:
        result = answer_question(
            conn, embedder, provider, args.question, k=args.top_k or settings.top_k
        )

    if args.json:
        print(json.dumps(
            {
                "question": result.question,
                "answer": result.answer,
                "sources": [vars(s) for s in result.sources],
            },
            indent=2,
        ))
        return 0

    print(f"\nQ: {result.question}\n")
    print(result.answer)
    print("\nSources:")
    for i, source in enumerate(result.sources, start=1):
        label = source.title or source.source
        print(f"  [{i}] {label}")
        print(f"      {source.source} (chunk {source.chunk_index}, "
              f"cosine {source.similarity:.3f})")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    with connect() as conn:
        rows = conn.execute(
            """
            SELECT d.title, d.source, count(c.id)
            FROM documents d LEFT JOIN chunks c ON c.document_id = d.id
            GROUP BY d.id ORDER BY d.id
            """
        ).fetchall()
        total = count_chunks(conn)
    for title, source, chunk_count in rows:
        print(f"{chunk_count:>6}  {title or source}")
    print(f"{total:>6}  TOTAL across {len(rows)} document(s)")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="docs-rag", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("init-db", help="apply the schema")
    p.set_defaults(func=cmd_init_db)

    p = sub.add_parser("ingest", help="chunk, embed and store a file or directory")
    p.add_argument("path", nargs="?", default="data/filings")
    p.add_argument("--embedder", choices=["sentence-transformers", "hash"])
    p.set_defaults(func=cmd_ingest)

    p = sub.add_parser("build-index", help="(re)build the IVFFlat cosine index")
    p.add_argument("--lists", type=int, default=None)
    p.set_defaults(func=cmd_build_index)

    p = sub.add_parser("query", help="ask a question against the corpus")
    p.add_argument("question")
    p.add_argument("--top-k", type=int, default=None)
    p.add_argument("--provider", choices=["anthropic", "openai", "ollama", "echo"])
    p.add_argument("--embedder", choices=["sentence-transformers", "hash"])
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_query)

    p = sub.add_parser("stats", help="chunk counts per document")
    p.set_defaults(func=cmd_stats)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
