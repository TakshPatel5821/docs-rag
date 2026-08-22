#!/usr/bin/env python3
"""Retrieval evaluation: does the right passage actually land in the top k?

    python eval/run_eval.py                          # in-memory, current settings
    python eval/run_eval.py --backend pgvector       # against the live database
    python eval/run_eval.py --sweep                  # compare chunk-size configs

Two metrics, over the questions in ``eval/questions.json``:

* **source recall@k** -- at least one of the top k chunks comes from the filing
  that actually contains the answer. This is the number that matters: if the
  right document never makes the context window, the generator cannot be right
  except by luck.
* **keyword hit@k** -- at least one retrieved chunk from the right document also
  contains one of the expected terms, i.e. we retrieved the relevant *passage*
  and not merely the relevant *document*.

The ``memory`` backend runs the same chunker and the same embedder as production
and swaps only the store (exact cosine in-process instead of pgvector), so a
sweep can be run without a database. ``pgvector`` measures the real system,
including IVFFlat's approximation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # run without installing

from app.chunking import chunk_text
from app.config import get_settings
from app.embeddings import get_embedder
from app.ingest import iter_corpus, read_document

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "eval" / "questions.json"


@dataclass
class Hit:
    source: str
    chunk_index: int
    content: str
    similarity: float


class MemoryIndex:
    """Exact cosine search over in-process vectors. Same chunker, same embedder."""

    def __init__(self, corpus_dir: Path, embedder, chunk_size: int, overlap: int):
        self.embedder = embedder
        self.sources: list[str] = []
        self.indexes: list[int] = []
        self.contents: list[str] = []
        for path in iter_corpus(corpus_dir):
            text = read_document(path)
            for chunk in chunk_text(text, chunk_size, overlap):
                self.sources.append(path.name)
                self.indexes.append(chunk.index)
                self.contents.append(chunk.content)
        started = time.perf_counter()
        self.vectors = embedder.embed_documents(self.contents)
        self.embed_seconds = time.perf_counter() - started

    def __len__(self) -> int:
        return len(self.contents)

    def search(self, question: str, k: int) -> list[Hit]:
        query = self.embedder.embed_query(question)
        scored = [
            (sum(a * b for a, b in zip(query, vector)), i)
            for i, vector in enumerate(self.vectors)
        ]
        scored.sort(reverse=True)
        return [
            Hit(self.sources[i], self.indexes[i], self.contents[i], score)
            for score, i in scored[:k]
        ]


class PgVectorIndex:
    def __init__(self, embedder):
        from app.db import connect

        self.embedder = embedder
        self._ctx = connect()
        self.conn = self._ctx.__enter__()

    def close(self) -> None:
        self._ctx.__exit__(None, None, None)

    def search(self, question: str, k: int) -> list[Hit]:
        from app.retrieval import search

        return [
            Hit(Path(c.source).name, c.chunk_index, c.content, c.similarity)
            for c in search(self.conn, self.embedder, question, k=k)
        ]


def evaluate(index, questions: list[dict], k: int, verbose: bool = True) -> dict:
    source_hits = 0
    keyword_hits = 0
    reciprocal_ranks = 0.0

    for item in questions:
        hits = index.search(item["question"], k)
        ranks = [
            rank
            for rank, hit in enumerate(hits, start=1)
            if hit.source == item["expected_source"]
        ]
        found_source = bool(ranks)
        found_keyword = any(
            hit.source == item["expected_source"]
            and any(kw.lower() in hit.content.lower() for kw in item["expected_keywords"])
            for hit in hits
        )
        source_hits += found_source
        keyword_hits += found_keyword
        reciprocal_ranks += 1.0 / ranks[0] if ranks else 0.0

        if verbose:
            mark = "PASS" if found_keyword else ("DOC " if found_source else "MISS")
            rank = f"rank {ranks[0]}" if ranks else "not in top k"
            print(f"  [{mark}] {item['question'][:68]:<68} {rank}")

    total = len(questions)
    return {
        "questions": total,
        "source_recall_at_k": source_hits / total,
        "keyword_hit_at_k": keyword_hits / total,
        "mrr": reciprocal_ranks / total,
    }


def ask(args) -> int:
    """Answer one question without a database.

    Retrieval is exact cosine over in-process vectors; everything after it --
    prompt assembly, the grounding rule, attribution, the empty-context refusal
    -- is ``app.service.answer_from_chunks``, the same code the API runs.
    """
    from app.generation import get_provider
    from app.retrieval import RetrievedChunk
    from app.service import answer_from_chunks

    embedder = get_embedder(args.embedder)
    index = MemoryIndex(Path(args.corpus), embedder, args.chunk_size, args.overlap)
    print(f"{len(index)} chunks | embedded in {index.embed_seconds:.1f}s "
          f"| embedder={args.embedder}\n")

    hits = index.search(args.ask, args.k)
    chunks = [
        RetrievedChunk(
            chunk_id=i,
            chunk_index=hit.chunk_index,
            content=hit.content,
            source=hit.source,
            title=hit.source,
            distance=1.0 - hit.similarity,
        )
        for i, hit in enumerate(hits)
    ]
    result = answer_from_chunks(args.ask, chunks, get_provider(args.provider))

    print(f"Q: {result.question}\n")
    print(result.answer)
    print("\nSources:")
    for i, source in enumerate(result.sources, start=1):
        print(f"  [{i}] {source.source} (cosine {source.similarity:.3f})")
        print(f"      {source.excerpt[:160]}")
    return 0


def main() -> int:
    settings = get_settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["memory", "pgvector"], default="memory")
    parser.add_argument("--corpus", default="data/filings")
    parser.add_argument("--embedder", choices=["sentence-transformers", "hash"],
                        default=settings.embedder)
    parser.add_argument("-k", type=int, default=settings.top_k)
    parser.add_argument("--chunk-size", type=int, default=settings.chunk_size)
    parser.add_argument("--overlap", type=int, default=settings.chunk_overlap)
    parser.add_argument("--sweep", action="store_true",
                        help="compare several chunk-size/overlap configurations")
    parser.add_argument("--ask", metavar="QUESTION",
                        help="answer one question and show the passages used")
    parser.add_argument("--provider", help="override LLM_PROVIDER for --ask")
    args = parser.parse_args()

    if args.ask:
        return ask(args)

    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))
    embedder = get_embedder(args.embedder)
    print(f"{len(questions)} questions | k={args.k} | embedder={args.embedder} | "
          f"backend={args.backend}\n")

    if args.backend == "pgvector":
        index = PgVectorIndex(embedder)
        try:
            report = evaluate(index, questions, args.k)
        finally:
            index.close()
        print(f"\nsource recall@{args.k}: {report['source_recall_at_k']:.0%}   "
              f"keyword hit@{args.k}: {report['keyword_hit_at_k']:.0%}   "
              f"MRR: {report['mrr']:.3f}")
        return 0

    configs = (
        [(400, 75), (800, 150), (1200, 200), (2000, 200)]
        if args.sweep
        else [(args.chunk_size, args.overlap)]
    )
    rows = []
    for chunk_size, overlap in configs:
        index = MemoryIndex(Path(args.corpus), embedder, chunk_size, overlap)
        print(f"chunk_size={chunk_size} overlap={overlap}: {len(index)} chunks, "
              f"embedded in {index.embed_seconds:.1f}s")
        report = evaluate(index, questions, args.k, verbose=not args.sweep)
        rows.append((chunk_size, overlap, len(index), report))
        print(f"  source recall@{args.k}: {report['source_recall_at_k']:.0%}   "
              f"keyword hit@{args.k}: {report['keyword_hit_at_k']:.0%}   "
              f"MRR: {report['mrr']:.3f}\n")

    if args.sweep:
        print(f"{'chunk':>6} {'overlap':>8} {'chunks':>8} {'recall@k':>9} "
              f"{'keyword@k':>10} {'MRR':>6}")
        for chunk_size, overlap, count, report in rows:
            print(f"{chunk_size:>6} {overlap:>8} {count:>8} "
                  f"{report['source_recall_at_k']:>8.0%} "
                  f"{report['keyword_hit_at_k']:>9.0%} {report['mrr']:>6.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
