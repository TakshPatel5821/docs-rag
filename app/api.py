"""FastAPI surface. Interactive docs at /docs (OpenAPI schema at /openapi.json)."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from app import __version__
from app.config import get_settings
from app.db import connect, count_chunks
from app.embeddings import get_embedder
from app.generation import get_provider
from app.generation.base import GenerationError
from app.service import answer_question

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    # Load the embedding model once at startup rather than per request -- the
    # first encode() otherwise pays a several-second model load.
    _state["embedder"] = get_embedder(settings.embedder)
    _state["provider"] = get_provider(settings.llm_provider)
    yield
    _state.clear()


app = FastAPI(
    title="docs-rag",
    version=__version__,
    description=(
        "Retrieval-augmented question answering over a corpus of SEC 10-K "
        "filings. Answers are grounded in retrieved passages and always carry "
        "source attribution."
    ),
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, examples=["What risk factors does Apple disclose about supply chain concentration?"])
    top_k: int | None = Field(None, ge=1, le=20)


class SourceModel(BaseModel):
    source: str
    title: str | None
    chunk_index: int
    similarity: float
    excerpt: str


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceModel]
    provider: str


@app.get("/health")
def health() -> dict:
    """Liveness plus a corpus size, so a green check means something."""
    settings = get_settings()
    try:
        with connect() as conn:
            chunks = count_chunks(conn)
    except Exception as exc:  # noqa: BLE001 - surfaced verbatim to the caller
        raise HTTPException(status_code=503, detail=f"database unavailable: {exc}")
    return {
        "status": "ok",
        "version": __version__,
        "chunks": chunks,
        "embedding_model": settings.embedding_model,
        "provider": settings.llm_provider,
    }


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    settings = get_settings()
    embedder = _state.get("embedder") or get_embedder(settings.embedder)
    provider = _state.get("provider") or get_provider(settings.llm_provider)

    try:
        with connect() as conn:
            result = answer_question(
                conn,
                embedder,
                provider,
                request.question,
                k=request.top_k or settings.top_k,
            )
    except GenerationError as exc:
        raise HTTPException(status_code=502, detail=str(exc))

    return QueryResponse(
        question=result.question,
        answer=result.answer,
        sources=[SourceModel(**vars(s)) for s in result.sources],
        provider=provider.name,
    )
